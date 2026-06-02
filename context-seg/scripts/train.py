from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image
from PIL import ImageDraw
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from context_seg import (
    HungarianMatcher,
    Mask2FormerLiteHead,
    MaskRefinementHead,
    MaskTargets,
    PseudoMaskFilter,
    SetCriterion,
    StreamingLingbotTokenExtractor,
    VideoFrameDataset,
    collate_single_clip,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Context token segmentation training")
    parser.add_argument("--config", type=str, default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--mask-root", type=str, default=None)
    parser.add_argument("--lingbot-checkpoint", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None, help="Resume segmentation head/optimizer from a checkpoint.")
    parser.add_argument("--no-resume-optimizer", action="store_true", help="Only load head weights from --resume.")
    parser.add_argument("--overfit", action="store_true", help="Use small defaults for a quick overfit experiment")
    parser.add_argument("--dry-run", action="store_true", help="Run a dummy forward/loss shape check")
    parser.add_argument("--streaming-dry-run", action="store_true", help="Run streaming extractor with a fake LingBot model")
    return parser.parse_args()


def dry_run() -> None:
    batch, tokens, context_dim = 2, 37 * 37, 1024
    queries, hidden_dim = 16, 256
    image_size = (518, 518)
    patch_grid = (37, 37)

    patch_tokens = torch.randn(batch, tokens, context_dim)
    seg_head = Mask2FormerLiteHead(
        context_dim=context_dim,
        hidden_dim=hidden_dim,
        pixel_dim=hidden_dim,
        num_queries=queries,
        num_heads=8,
        decoder_layers=2,
    )
    criterion = SetCriterion()

    output = seg_head(patch_tokens, patch_grid=patch_grid, image_size=image_size)
    refiner = MaskRefinementHead(refine_dim=4, hidden_dim=8, query_chunk_size=4)
    refined_logits = refiner(torch.rand(1, 3, *image_size), output.mask_logits[0])
    targets = MaskTargets(masks=torch.randint(0, 2, (8, *image_size)).float())
    aux_outputs = tuple({"pred_masks": aux["pred_masks"][0], "pred_logits": aux["pred_logits"][0]} for aux in output.aux_outputs)
    losses = criterion(refined_logits, targets, output.pred_logits[0], aux_outputs=aux_outputs)

    print("dry-run ok")
    print(f"query_embeddings: {tuple(output.query_embeddings.shape)}")
    print(f"objectness_logits: {tuple(output.objectness_logits.shape)}")
    print(f"mask_logits: {tuple(output.mask_logits.shape)}")
    print(f"refined_logits: {tuple(refined_logits.shape)}")
    print(f"loss: {losses['loss'].item():.4f}")


def streaming_dry_run() -> None:
    batch, frames, channels, height, width = 1, 5, 3, 56, 56
    patch_size, context_dim, hidden_dim = 14, 32, 16
    num_queries = 4

    fake_lingbot = _FakeStreamingLingbot(patch_size=patch_size, context_dim=context_dim)
    extractor = StreamingLingbotTokenExtractor(fake_lingbot, require_camera_only_cache=True)
    seg_head = Mask2FormerLiteHead(
        context_dim=context_dim,
        hidden_dim=hidden_dim,
        pixel_dim=hidden_dim,
        num_queries=num_queries,
        num_heads=4,
        decoder_layers=1,
    )

    outputs = list(
        extractor.stream_sequence(
            torch.randn(batch, frames, channels, height, width),
            num_scale_frames=2,
            keyframe_interval=2,
        )
    )
    step_tokens = outputs[-1]
    head_output = seg_head(step_tokens.patch_tokens, step_tokens.patch_grid, step_tokens.image_size)

    print("streaming dry-run ok")
    print(f"num_outputs: {len(outputs)}")
    print(f"scale_patch_tokens: {tuple(outputs[0].patch_tokens.shape)}")
    print(f"step_patch_tokens: {tuple(step_tokens.patch_tokens.shape)}")
    print(f"objectness_logits: {tuple(head_output.objectness_logits.shape)}")
    print(f"mask_logits: {tuple(head_output.mask_logits.shape)}")
    print(f"clean_calls: {fake_lingbot.clean_calls}")
    print(f"skip_append_events: {fake_lingbot.skip_append_events}")


def train(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    apply_overrides(cfg, args)

    device = torch.device(cfg["train"]["device"] if torch.cuda.is_available() else "cpu")
    output_dir = resolve_path(cfg["train"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "vis").mkdir(exist_ok=True)

    dataset = VideoFrameDataset(
        root=resolve_path(cfg["data"]["root"]),
        mask_root=resolve_path(cfg["data"]["mask_root"]),
        clip_length=cfg["data"]["clip_length"],
        image_size=cfg["model"]["mask_size"],
        clip_stride=cfg["data"].get("clip_stride", 1),
        limit_scenes=cfg["data"].get("limit_scenes"),
        limit_clips=cfg["data"].get("limit_clips"),
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=cfg["train"].get("num_workers", 0),
        collate_fn=collate_single_clip,
    )

    lingbot_ckpt = resolve_path(cfg["lingbot"]["checkpoint"])
    extractor = StreamingLingbotTokenExtractor.from_checkpoint(
        checkpoint_path=str(lingbot_ckpt),
        device=device,
        token_layer=cfg["model"]["token_layer"],
        freeze_backbone=True,
        use_no_grad=True,
        kv_cache_camera_only=cfg["lingbot"].get("kv_cache_camera_only", True),
        kv_cache_sliding_window=cfg["lingbot"]["kv_cache_sliding_window"],
        kv_cache_scale_frames=cfg["lingbot"]["kv_cache_scale_frames"],
        kv_cache_cross_frame_special=cfg["lingbot"]["kv_cache_cross_frame_special"],
        kv_cache_include_scale_frames=cfg["lingbot"]["kv_cache_include_scale_frames"],
        img_size=cfg["model"]["mask_size"][0],
        patch_size=cfg["model"]["patch_size"],
        enable_3d_rope=cfg["lingbot"].get("enable_3d_rope", True),
        max_frame_num=cfg["lingbot"].get("max_frame_num", 1024),
        use_sdpa=cfg["lingbot"].get("use_sdpa", False),
    )

    use_refinement = bool(cfg["model"].get("use_refinement", False))
    seg_head = None
    mask_refiner = build_mask_refiner(cfg).to(device) if use_refinement else None
    optimizer = None
    matcher = HungarianMatcher(
        cost_class=cfg["matcher"].get("cost_class", cfg["matcher"].get("cost_objectness", 2.0)),
        cost_mask=cfg["matcher"].get("cost_mask", cfg["matcher"].get("cost_bce", 5.0)),
        cost_dice=cfg["matcher"]["cost_dice"],
        cost_size=cfg["matcher"]["cost_size"],
    )
    criterion = SetCriterion(
        class_weight=cfg["loss"].get("class_weight", cfg["loss"].get("objectness_weight", 2.0)),
        mask_weight=cfg["loss"].get("mask_weight", cfg["loss"].get("mask_bce_weight", 5.0)),
        dice_weight=cfg["loss"].get("dice_weight", cfg["loss"].get("mask_dice_weight", 5.0)),
        no_object_weight=cfg["loss"].get("no_object_weight", 0.1),
        num_points=cfg["loss"].get("num_points", 12544),
        oversample_ratio=cfg["loss"].get("oversample_ratio", 3.0),
        importance_sample_ratio=cfg["loss"].get("importance_sample_ratio", 0.75),
        matcher=matcher,
    )
    mask_filter = PseudoMaskFilter(
        min_area_ratio=cfg.get("pseudo_mask", {}).get("min_area_ratio", 0.001),
        max_area_ratio=cfg.get("pseudo_mask", {}).get("max_area_ratio", 0.6),
        nms_iou=cfg.get("pseudo_mask", {}).get("nms_iou", 0.8),
        max_masks=cfg["train"].get("max_target_masks", 20),
    )

    max_steps = int(cfg["train"]["max_steps"])
    log_path = output_dir / "train_log.txt"
    if not log_path.exists() or log_path.stat().st_size == 0:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("step scene loss loss_cls loss_mask loss_dice loss_aux pred_mean pred_max object_mean object_max active_queries num_targets num_matches\n")

    resume_state = load_resume_state(args.resume, device) if args.resume is not None else None
    resume_loaded = False

    print_training_summary(cfg, device, output_dir, lingbot_ckpt, len(dataset), args.resume)
    step = 0
    with tqdm(total=max_steps, desc="training", unit="step") as progress:
        while step < max_steps:
            for sample in loader:
                if step >= max_steps:
                    break
                images = sample.images.to(device)
                frame_outputs = list(
                    extractor.stream_sequence(
                        images,
                        num_scale_frames=cfg["data"]["num_scale_frames"],
                        keyframe_interval=cfg["lingbot"].get("keyframe_interval", 1),
                    )
                )

                if seg_head is None or optimizer is None:
                    context_dim = int(frame_outputs[0].patch_tokens.shape[-1])
                    seg_head = Mask2FormerLiteHead(
                        context_dim=context_dim,
                        hidden_dim=cfg["model"]["hidden_dim"],
                        pixel_dim=cfg["model"].get("pixel_dim", cfg["model"]["hidden_dim"]),
                        num_queries=cfg["model"]["num_queries"],
                        num_heads=cfg["model"]["num_attention_heads"],
                        decoder_layers=cfg["model"].get("decoder_layers", cfg["model"].get("num_predictor_layers", 6)),
                        ffn_dim=cfg["model"].get("ffn_dim", 4 * cfg["model"]["hidden_dim"]),
                    ).to(device)
                    trainable_params = list(seg_head.parameters())
                    if mask_refiner is not None:
                        trainable_params.extend(mask_refiner.parameters())
                    optimizer = torch.optim.AdamW(
                        trainable_params,
                        lr=cfg["train"]["lr"],
                        weight_decay=cfg["train"]["weight_decay"],
                    )
                    if resume_state is not None and not resume_loaded:
                        load_training_state(
                            resume_state,
                            seg_head,
                            mask_refiner,
                            optimizer,
                            load_optimizer=not args.no_resume_optimizer,
                        )
                        resume_loaded = True

                assert seg_head is not None and optimizer is not None
                optimizer.zero_grad(set_to_none=True)

                loss_dicts = []
                cursor = 0
                last_logits = None
                last_coarse_logits = None
                last_pred_logits = None
                last_target = None
                last_rgb = None
                for context in frame_outputs:
                    head_output = seg_head(context.patch_tokens, context.patch_grid, context.image_size)
                    coarse_mask_logits = head_output.mask_logits
                    pred_logits = head_output.pred_logits
                    _, chunk_frames = coarse_mask_logits.shape[:2]
                    for local_idx in range(chunk_frames):
                        target = sample.target_masks[cursor].to(device)
                        target = mask_filter(target, max_masks=cfg["train"].get("max_target_masks"))
                        coarse_logits = coarse_mask_logits[0, local_idx]
                        frame_pred_logits = pred_logits[0, local_idx]
                        matched_indices = None
                        if mask_refiner is not None:
                            # Refine only matched queries during training. Unmatched queries are
                            # supervised by object/no-object classification, so keeping RGB-refine
                            # graphs for every query wastes a large amount of high-res memory.
                            match = matcher(coarse_logits.detach(), target, frame_pred_logits.detach())
                            matched_indices = (match.pred_indices, match.target_indices)
                            if match.pred_indices.numel() > 0:
                                refined_logits = mask_refiner(
                                    images[cursor : cursor + 1],
                                    coarse_logits[match.pred_indices],
                                )
                                primary_logits = coarse_logits.clone()
                                primary_logits[match.pred_indices] = refined_logits
                            else:
                                primary_logits = coarse_logits
                        else:
                            primary_logits = coarse_logits
                        aux_outputs = tuple(
                            {
                                "pred_masks": aux["pred_masks"][0, local_idx],
                                "pred_logits": aux["pred_logits"][0, local_idx],
                            }
                            for aux in head_output.aux_outputs
                        )
                        loss_dict = criterion(
                            primary_logits,
                            MaskTargets(masks=target),
                            frame_pred_logits,
                            matched_indices=matched_indices,
                            aux_outputs=aux_outputs,
                        )
                        loss_dicts.append(loss_dict)
                        last_logits = primary_logits.detach().cpu()
                        last_coarse_logits = coarse_logits.detach().cpu()
                        last_pred_logits = pred_logits[0, local_idx].detach().cpu()
                        last_target = target.detach().cpu()
                        last_rgb = images[cursor].detach().cpu()
                        cursor += 1

                metrics = average_loss_dicts(loss_dicts)
                loss = metrics["loss"]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    trainable_params,
                    cfg["train"].get("grad_clip", 1.0),
                )
                optimizer.step()

                step += 1
                log_values = {
                    "loss": float(metrics["loss"].detach().cpu()),
                    "loss_cls": float(metrics["loss_cls"].detach().cpu()),
                    "loss_mask": float(metrics["loss_mask"].detach().cpu()),
                    "loss_dice": float(metrics["loss_dice"].detach().cpu()),
                    "loss_aux": float(metrics["loss_aux"].detach().cpu()),
                    "pred_mean": float(metrics["pred_prob_mean"].detach().cpu()),
                    "pred_max": float(metrics["pred_prob_max"].detach().cpu()),
                    "object_mean": float(metrics["object_prob_mean"].detach().cpu()),
                    "object_max": float(metrics["object_prob_max"].detach().cpu()),
                    "active_queries": float(metrics["active_queries"].detach().cpu()),
                    "num_targets": float(metrics["num_targets"].detach().cpu()),
                    "num_matches": float(metrics["num_matches"].detach().cpu()),
                }
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"{step} {sample.scene} {log_values['loss']:.6f} {log_values['loss_cls']:.6f} "
                        f"{log_values['loss_mask']:.6f} {log_values['loss_dice']:.6f} {log_values['loss_aux']:.6f} "
                        f"{log_values['pred_mean']:.6f} {log_values['pred_max']:.6f} "
                        f"{log_values['object_mean']:.6f} {log_values['object_max']:.6f} "
                        f"{log_values['active_queries']:.1f} {log_values['num_targets']:.1f} {log_values['num_matches']:.1f}\n"
                    )

                progress.set_postfix(
                    scene=sample.scene,
                    loss=f"{log_values['loss']:.4f}",
                    cls=f"{log_values['loss_cls']:.4f}",
                    mask=f"{log_values['loss_mask']:.4f}",
                    dice=f"{log_values['loss_dice']:.4f}",
                    aux=f"{log_values['loss_aux']:.4f}",
                    omean=f"{log_values['object_mean']:.3f}",
                    omax=f"{log_values['object_max']:.3f}",
                    active=f"{log_values['active_queries']:.0f}",
                    targets=f"{log_values['num_targets']:.0f}",
                )
                progress.update(1)

                if (
                    step % int(cfg["train"]["vis_interval"]) == 0
                    and last_logits is not None
                    and last_coarse_logits is not None
                    and last_target is not None
                    and last_pred_logits is not None
                ):
                    save_mask_visualization(
                        output_dir / "vis" / f"step_{step:06d}.png",
                        last_logits,
                        last_target,
                        last_pred_logits,
                        rgb_image=last_rgb,
                        coarse_logits=last_coarse_logits,
                        refinement_enabled=mask_refiner is not None,
                        matcher=matcher,
                        objectness_threshold=cfg["train"].get("objectness_threshold", 0.5),
                        max_active_queries=cfg["train"].get("max_active_queries"),
                        min_mask_area_ratio=cfg["train"].get("min_vis_mask_area_ratio", 0.001),
                    )
                if step % int(cfg["train"]["save_interval"]) == 0:
                    save_checkpoint(output_dir / f"checkpoint_step_{step:06d}.pt", seg_head, mask_refiner, optimizer, cfg, step)

    if seg_head is not None and optimizer is not None:
        save_checkpoint(output_dir / "checkpoint_last.pt", seg_head, mask_refiner, optimizer, cfg, step)


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    if args.data_root is not None:
        cfg["data"]["root"] = args.data_root
    if args.mask_root is not None:
        cfg["data"]["mask_root"] = args.mask_root
    if args.lingbot_checkpoint is not None:
        cfg["lingbot"]["checkpoint"] = args.lingbot_checkpoint
    if args.output_dir is not None:
        cfg["train"]["output_dir"] = args.output_dir
    if args.max_steps is not None:
        cfg["train"]["max_steps"] = args.max_steps
    if args.overfit:
        cfg["data"]["limit_clips"] = cfg["data"].get("overfit_limit_clips", 4)
        cfg["train"]["max_steps"] = args.max_steps if args.max_steps is not None else cfg["train"].get("overfit_steps", 20)


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute() or path.exists():
        return path
    candidate = REPO_ROOT / path
    if candidate.exists() or not str(path).startswith(".."):
        return candidate
    return PROJECT_ROOT / path


def build_mask_refiner(cfg: dict[str, Any]) -> MaskRefinementHead:
    refinement_cfg = cfg["model"].get("refinement", {})
    return MaskRefinementHead(
        refine_dim=refinement_cfg.get("refine_dim", 8),
        hidden_dim=refinement_cfg.get("hidden_dim", 16),
        query_chunk_size=refinement_cfg.get("query_chunk_size", 10),
        residual_scale=refinement_cfg.get("residual_scale", 1.0),
    )


def save_checkpoint(
    path: Path,
    seg_head: Mask2FormerLiteHead,
    mask_refiner: MaskRefinementHead | None,
    optimizer: torch.optim.Optimizer,
    cfg: dict[str, Any],
    step: int,
) -> None:
    state = {
        "step": step,
        "seg_head": seg_head.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": cfg,
    }
    if mask_refiner is not None:
        state["mask_refiner"] = mask_refiner.state_dict()
    torch.save(state, path)


def load_resume_state(path: str | Path, device: torch.device) -> dict[str, Any]:
    ckpt_path = resolve_path(path)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "seg_head" not in state:
        raise KeyError(f"Checkpoint does not contain 'seg_head': {ckpt_path}")
    print(f"Resume checkpoint: {ckpt_path} (saved step={state.get('step', 'unknown')})")
    return state


def load_training_state(
    state: dict[str, Any],
    seg_head: Mask2FormerLiteHead,
    mask_refiner: MaskRefinementHead | None,
    optimizer: torch.optim.Optimizer,
    load_optimizer: bool = True,
) -> None:
    seg_head.load_state_dict(state["seg_head"], strict=True)
    if mask_refiner is not None:
        if "mask_refiner" in state:
            mask_refiner.load_state_dict(state["mask_refiner"], strict=True)
            print("Loaded mask refiner state from checkpoint.")
        else:
            print("Checkpoint has no mask_refiner; using randomly initialized refiner.")
    if load_optimizer and "optimizer" in state:
        try:
            optimizer.load_state_dict(state["optimizer"])
            print("Loaded segmentation head/refiner and optimizer state from checkpoint.")
        except ValueError as exc:
            print(f"Could not load optimizer state ({exc}); optimizer starts fresh.")
    else:
        print("Loaded model weights from checkpoint; optimizer starts fresh.")


def average_loss_dicts(loss_dicts: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not loss_dicts:
        raise ValueError("No losses were produced for this training step")
    keys = (
        "loss",
        "loss_cls",
        "loss_mask",
        "loss_dice",
        "loss_aux",
        "pred_prob_mean",
        "pred_prob_max",
        "object_prob_mean",
        "object_prob_max",
        "active_queries",
        "num_targets",
        "num_matches",
    )
    averaged = {}
    for key in keys:
        values = torch.stack([item[key] for item in loss_dicts])
        averaged[key] = values.sum() if key in {"active_queries", "num_matches"} else values.mean()
    return averaged


def print_training_summary(
    cfg: dict[str, Any],
    device: torch.device,
    output_dir: Path,
    lingbot_ckpt: Path,
    num_clips: int,
    resume: str | None = None,
) -> None:
    print("Training configuration")
    print(f"  device: {device}")
    print(f"  output_dir: {output_dir}")
    print(f"  data.root: {resolve_path(cfg['data']['root'])}")
    print(f"  data.mask_root: {resolve_path(cfg['data']['mask_root'])}")
    print(f"  lingbot.checkpoint: {lingbot_ckpt}")
    if resume is not None:
        print(f"  resume: {resolve_path(resume)}")
    print(f"  clips: {num_clips}")
    print(f"  max_steps: {cfg['train']['max_steps']}")
    print(f"  head_type: {cfg['model'].get('head_type', 'mask2former_lite')}")
    print(f"  num_queries: {cfg['model']['num_queries']}")
    print(f"  refinement: {bool(cfg['model'].get('use_refinement', False))}")
    print(
        "  loss: "
        f"class_weight={cfg['loss'].get('class_weight', 2.0)}, "
        f"mask_weight={cfg['loss'].get('mask_weight', 5.0)}, "
        f"dice_weight={cfg['loss'].get('dice_weight', 5.0)}, "
        f"no_object_weight={cfg['loss'].get('no_object_weight', 0.1)}, "
        f"num_points={cfg['loss'].get('num_points', 12544)}"
    )


def save_mask_visualization(
    path: Path,
    pred_logits: torch.Tensor,
    target_masks: torch.Tensor,
    class_logits: torch.Tensor,
    rgb_image: torch.Tensor | None = None,
    coarse_logits: torch.Tensor | None = None,
    refinement_enabled: bool = False,
    matcher: HungarianMatcher | None = None,
    top_k: int = 4,
    objectness_threshold: float = 0.5,
    max_active_queries: int | None = None,
    min_mask_area_ratio: float = 0.001,
    mask_threshold: float = 0.5,
) -> None:
    pred_prob = pred_logits.sigmoid()
    coarse_prob = coarse_logits.sigmoid() if coarse_logits is not None else pred_prob
    objectness_prob = class_logits.softmax(dim=-1)[:, 1]
    active_indices = torch.nonzero(objectness_prob >= float(objectness_threshold), as_tuple=False).flatten()
    raw_active_indices = active_indices.clone()
    if active_indices.numel() > 0 and min_mask_area_ratio > 0:
        bin_masks = pred_prob[active_indices] >= float(mask_threshold)
        areas = bin_masks.flatten(1).float().mean(dim=1)
        active_indices = active_indices[areas >= float(min_mask_area_ratio)]
    if max_active_queries is not None and active_indices.numel() > int(max_active_queries):
        top_active = torch.argsort(objectness_prob[active_indices], descending=True)[: int(max_active_queries)]
        active_indices = active_indices[top_active]
    if active_indices.numel() > 0:
        pred = pred_prob[active_indices].max(dim=0).values.numpy()
        coarse_pred = coarse_prob[active_indices].max(dim=0).values.numpy()
    else:
        pred = np.zeros(pred_prob.shape[-2:], dtype=np.float32)
        coarse_pred = np.zeros(pred_prob.shape[-2:], dtype=np.float32)
    if target_masks.numel() > 0:
        target = target_masks.max(dim=0).values.numpy()
    else:
        target = np.zeros_like(pred)

    match = matcher(pred_logits, target_masks, class_logits) if matcher is not None and target_masks.numel() > 0 else None
    if match is not None and match.pred_indices.numel() > 0:
        matched_union = pred_prob[match.pred_indices].max(dim=0).values.numpy()
        matched_order = _sort_matches_by_iou(pred_prob, target_masks, match.pred_indices, match.target_indices)
    else:
        matched_union = np.zeros_like(pred)
        matched_order = []

    rgb = _rgb_uint8(rgb_image, pred.shape)
    query_scores = objectness_prob
    top_indices = torch.argsort(query_scores, descending=True)[:top_k]
    row1 = [
        _panel("rgb", rgb),
        _panel("target_overlay", _overlay_masks(rgb, target=target)),
        _panel("active_overlay", _overlay_masks(rgb, pred=pred, target=target)),
        _panel("matched_overlay", _overlay_masks(rgb, pred=matched_union, target=target)),
        _panel("coarse_active", coarse_pred),
        _panel("refined_active", pred),
    ]
    for idx in top_indices:
        row1.append(_panel(f"top_obj q{int(idx)} {float(query_scores[int(idx)]):.2f}", pred_prob[int(idx)].numpy()))

    row2 = []
    for pred_idx, target_idx, iou in matched_order[: 3 + top_k]:
        row2.append(_panel(f"match q{pred_idx}->t{target_idx} iou{iou:.2f}", pred_prob[pred_idx].numpy()))
    while len(row2) < len(row1):
        row2.append(_panel("", np.zeros_like(pred)))

    panel_h, panel_w = row1[0].shape[:2]
    row1 = _pad_panels(row1, panel_h, panel_w)
    row2 = _pad_panels(row2, panel_h, panel_w)
    canvas = np.concatenate([np.concatenate(row1, axis=1), np.concatenate(row2, axis=1)], axis=0)
    Image.fromarray(canvas).save(path)
    stats_path = path.with_suffix(".txt")
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write(f"pred_min {float(pred_prob.min()):.6f}\n")
        f.write(f"pred_mean {float(pred_prob.mean()):.6f}\n")
        f.write(f"pred_max {float(pred_prob.max()):.6f}\n")
        f.write(f"coarse_pred_min {float(coarse_prob.min()):.6f}\n")
        f.write(f"coarse_pred_mean {float(coarse_prob.mean()):.6f}\n")
        f.write(f"coarse_pred_max {float(coarse_prob.max()):.6f}\n")
        f.write(f"refinement_enabled {bool(refinement_enabled)}\n")
        f.write(f"object_prob_min {float(objectness_prob.min()):.6f}\n")
        f.write(f"object_prob_mean {float(objectness_prob.mean()):.6f}\n")
        f.write(f"object_prob_max {float(objectness_prob.max()):.6f}\n")
        f.write(f"raw_active_query_count {int(raw_active_indices.numel())}\n")
        f.write(f"area_filtered_active_query_count {int(active_indices.numel())}\n")
        f.write(f"active_query_indices {[int(i) for i in active_indices]}\n")
        f.write(f"top_query_indices {[int(i) for i in top_indices]}\n")
        f.write(f"top_query_scores {[float(query_scores[int(i)]) for i in top_indices]}\n")
        if match is not None:
            f.write(f"matched_query_indices {[int(i) for i in match.pred_indices]}\n")
            f.write(f"matched_target_indices {[int(i) for i in match.target_indices]}\n")
            f.write(f"matched_iou_sorted {matched_order}\n")


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    return (arr * 255).clip(0, 255).astype(np.uint8)


def _panel(title: str, arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        image = Image.fromarray(_to_uint8(arr)).convert("RGB")
    elif arr.ndim == 3 and arr.shape[-1] == 3:
        image = Image.fromarray(arr.astype(np.uint8)).convert("RGB")
    else:
        raise ValueError(f"Expected panel array [H,W] or [H,W,3], got {arr.shape}")
    label_h = 18
    canvas = Image.new("RGB", (image.width, image.height + label_h), color=(0, 0, 0))
    canvas.paste(image, (0, label_h))
    if title:
        ImageDraw.Draw(canvas).text((4, 2), title, fill=255)
    return np.asarray(canvas)


def _rgb_uint8(rgb_image: torch.Tensor | None, fallback_shape: tuple[int, int]) -> np.ndarray:
    if rgb_image is None:
        return np.zeros((*fallback_shape, 3), dtype=np.uint8)
    rgb = rgb_image.detach().cpu()
    if rgb.ndim == 4:
        rgb = rgb[0]
    if rgb.ndim != 3 or rgb.shape[0] != 3:
        raise ValueError(f"Expected rgb image [3,H,W] or [1,3,H,W], got {tuple(rgb_image.shape)}")
    arr = rgb.permute(1, 2, 0).numpy()
    return _to_uint8(arr)


def _overlay_masks(
    rgb: np.ndarray,
    pred: np.ndarray | None = None,
    target: np.ndarray | None = None,
    alpha: float = 0.55,
    threshold: float = 0.5,
) -> np.ndarray:
    out = (rgb.astype(np.float32) * 0.65).clip(0, 255)
    pred_mask = pred >= threshold if pred is not None else np.zeros(rgb.shape[:2], dtype=bool)
    target_mask = target >= threshold if target is not None else np.zeros(rgb.shape[:2], dtype=bool)
    target_only = target_mask & ~pred_mask
    pred_only = pred_mask & ~target_mask
    both = target_mask & pred_mask
    out[target_only] = (1 - alpha) * out[target_only] + alpha * np.array([0, 255, 0], dtype=np.float32)
    out[pred_only] = (1 - alpha) * out[pred_only] + alpha * np.array([255, 0, 0], dtype=np.float32)
    out[both] = (1 - alpha) * out[both] + alpha * np.array([255, 255, 0], dtype=np.float32)
    return out.clip(0, 255).astype(np.uint8)


def _pad_panels(panels: list[np.ndarray], height: int, width: int) -> list[np.ndarray]:
    padded = []
    for panel in panels:
        if panel.shape[:2] == (height, width):
            padded.append(panel)
            continue
        out = np.zeros((height, width, panel.shape[2] if panel.ndim == 3 else 1), dtype=np.uint8)
        h = min(height, panel.shape[0])
        w = min(width, panel.shape[1])
        out[:h, :w] = panel[:h, :w]
        if out.shape[-1] == 1:
            out = np.repeat(out, 3, axis=-1)
        padded.append(out)
    return padded


def _sort_matches_by_iou(
    pred_prob: torch.Tensor,
    target_masks: torch.Tensor,
    pred_indices: torch.Tensor,
    target_indices: torch.Tensor,
    threshold: float = 0.5,
) -> list[tuple[int, int, float]]:
    items = []
    for pred_idx, target_idx in zip(pred_indices.tolist(), target_indices.tolist()):
        pred = pred_prob[pred_idx] >= threshold
        target = target_masks[target_idx] > 0.5
        intersection = torch.logical_and(pred, target).sum().float()
        union = torch.logical_or(pred, target).sum().float().clamp_min(1.0)
        items.append((int(pred_idx), int(target_idx), float((intersection / union).cpu())))
    return sorted(items, key=lambda item: item[2], reverse=True)


def main() -> None:
    args = parse_args()
    if args.dry_run:
        dry_run()
        return
    if args.streaming_dry_run:
        streaming_dry_run()
        return
    train(args)


class _FakeStreamingLingbot(torch.nn.Module):
    def __init__(self, patch_size: int, context_dim: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.context_dim = context_dim
        self.num_frame_for_scale = 2
        self.kv_cache_camera_only = True
        self.clean_calls = 0
        self.skip_append_events: list[bool] = []

    def clean_kv_cache(self) -> None:
        self.clean_calls += 1

    def _set_skip_append(self, skip: bool) -> None:
        self.skip_append_events.append(skip)

    def _aggregate_features(
        self,
        images: torch.Tensor,
        num_frame_for_scale: int | None = None,
        num_frame_per_block: int = 1,
        **kwargs,
    ) -> tuple[list[torch.Tensor], int]:
        batch, frames, _, height, width = images.shape
        patch_h, patch_w = height // self.patch_size, width // self.patch_size
        patch_start_idx = 3
        tokens = patch_start_idx + patch_h * patch_w
        out = torch.randn(batch, frames, tokens, self.context_dim, device=images.device)
        return [out], patch_start_idx


if __name__ == "__main__":
    main()
