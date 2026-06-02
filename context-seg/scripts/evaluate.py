from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from context_seg import (  # noqa: E402
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
from train import (  # noqa: E402
    average_loss_dicts,
    build_mask_refiner,
    load_config,
    resolve_path,
    save_mask_visualization,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate context-seg checkpoints on held-out scenes.")
    parser.add_argument("--config", type=str, default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--mask-root", type=str, default=None)
    parser.add_argument("--lingbot-checkpoint", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--limit-scenes", nargs="*", default=None, help="Exact scene folder names, e.g. ai_018_001_scene_cam_00.")
    parser.add_argument("--limit-clips", type=int, default=None)
    parser.add_argument("--vis-interval", type=int, default=10)
    parser.add_argument("--max-vis", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    apply_eval_overrides(cfg, args)

    device = torch.device(cfg["train"]["device"] if torch.cuda.is_available() else "cpu")
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "vis").mkdir(exist_ok=True)

    limit_scenes = args.limit_scenes
    if limit_scenes is None:
        limit_scenes = cfg["data"].get("val_limit_scenes")
    dataset = VideoFrameDataset(
        root=resolve_path(cfg["data"]["root"]),
        mask_root=resolve_path(cfg["data"]["mask_root"]),
        clip_length=cfg["data"]["clip_length"],
        image_size=cfg["model"]["mask_size"],
        clip_stride=cfg["data"].get("clip_stride", 1),
        limit_scenes=limit_scenes,
        limit_clips=args.limit_clips,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_single_clip)

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
        boundary_weight=cfg["loss"].get("boundary_weight", 0.0),
        boundary_size=cfg["loss"].get("boundary_size", 128),
        boundary_kernel_size=cfg["loss"].get("boundary_kernel_size", 3),
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

    extractor = StreamingLingbotTokenExtractor.from_checkpoint(
        checkpoint_path=str(resolve_path(cfg["lingbot"]["checkpoint"])),
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

    checkpoint = torch.load(resolve_path(args.checkpoint), map_location=device, weights_only=False)
    if "seg_head" not in checkpoint:
        raise KeyError(f"Checkpoint does not contain 'seg_head': {args.checkpoint}")

    seg_head: Mask2FormerLiteHead | None = None
    mask_refiner: MaskRefinementHead | None = build_mask_refiner(cfg).to(device) if cfg["model"].get("use_refinement", False) else None
    if mask_refiner is not None:
        if "mask_refiner" in checkpoint:
            mask_refiner.load_state_dict(checkpoint["mask_refiner"], strict=True)
            mask_refiner.eval()
        else:
            print("Checkpoint has no mask_refiner; evaluation will use the randomly initialized refiner.")

    log_path = output_dir / "eval_log.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("clip scene loss loss_cls loss_mask loss_dice loss_boundary loss_aux pred_mean pred_max object_mean object_max active_queries num_targets num_matches\n")

    all_metrics: list[dict[str, torch.Tensor]] = []
    vis_count = 0
    print(f"Evaluation configuration: clips={len(dataset)} output_dir={output_dir} scenes={limit_scenes}")
    with torch.no_grad(), tqdm(total=len(dataset), desc="evaluating", unit="clip") as progress:
        for clip_idx, sample in enumerate(loader, start=1):
            images = sample.images.to(device)
            frame_outputs = list(
                extractor.stream_sequence(
                    images,
                    num_scale_frames=cfg["data"]["num_scale_frames"],
                    keyframe_interval=cfg["lingbot"].get("keyframe_interval", 1),
                )
            )

            if seg_head is None:
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
                seg_head.load_state_dict(checkpoint["seg_head"], strict=True)
                seg_head.eval()

            assert seg_head is not None
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
                    primary_logits = mask_refiner(images[cursor : cursor + 1], coarse_logits) if mask_refiner is not None else coarse_logits
                    frame_pred_logits = pred_logits[0, local_idx]
                    aux_outputs = tuple(
                        {
                            "pred_masks": aux["pred_masks"][0, local_idx],
                            "pred_logits": aux["pred_logits"][0, local_idx],
                        }
                        for aux in head_output.aux_outputs
                    )
                    loss_dicts.append(
                        criterion(
                            primary_logits,
                            MaskTargets(masks=target),
                            frame_pred_logits,
                            aux_outputs=aux_outputs,
                        )
                    )
                    last_logits = primary_logits.detach().cpu()
                    last_coarse_logits = coarse_logits.detach().cpu()
                    last_pred_logits = frame_pred_logits.detach().cpu()
                    last_target = target.detach().cpu()
                    last_rgb = images[cursor].detach().cpu()
                    cursor += 1

            metrics = average_loss_dicts(loss_dicts)
            all_metrics.append(metrics)
            log_values = tensor_metrics_to_float(metrics)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(
                    f"{clip_idx} {sample.scene} {log_values['loss']:.6f} {log_values['loss_cls']:.6f} "
                    f"{log_values['loss_mask']:.6f} {log_values['loss_dice']:.6f} "
                    f"{log_values['loss_boundary']:.6f} {log_values['loss_aux']:.6f} "
                    f"{log_values['pred_mean']:.6f} {log_values['pred_max']:.6f} "
                    f"{log_values['object_mean']:.6f} {log_values['object_max']:.6f} "
                    f"{log_values['active_queries']:.1f} {log_values['num_targets']:.1f} {log_values['num_matches']:.1f}\n"
                )

            if (
                vis_count < int(args.max_vis)
                and clip_idx % max(1, int(args.vis_interval)) == 0
                and last_logits is not None
                and last_coarse_logits is not None
                and last_pred_logits is not None
                and last_target is not None
            ):
                save_mask_visualization(
                    output_dir / "vis" / f"clip_{clip_idx:06d}.png",
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
                vis_count += 1

            progress.set_postfix(
                scene=sample.scene,
                loss=f"{log_values['loss']:.4f}",
                dice=f"{log_values['loss_dice']:.4f}",
                bdry=f"{log_values['loss_boundary']:.4f}",
                active=f"{log_values['active_queries']:.0f}",
            )
            progress.update(1)

    summary = average_loss_dicts(all_metrics)
    summary_values = tensor_metrics_to_float(summary)
    with open(output_dir / "eval_summary.txt", "w", encoding="utf-8") as f:
        for key, value in summary_values.items():
            f.write(f"{key} {value:.6f}\n")
    print("Evaluation summary")
    for key in ("loss", "loss_mask", "loss_dice", "loss_boundary", "loss_aux", "active_queries", "num_targets", "num_matches"):
        print(f"  {key}: {summary_values[key]:.6f}")


def apply_eval_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    if args.data_root is not None:
        cfg["data"]["root"] = args.data_root
    if args.mask_root is not None:
        cfg["data"]["mask_root"] = args.mask_root
    if args.lingbot_checkpoint is not None:
        cfg["lingbot"]["checkpoint"] = args.lingbot_checkpoint


def tensor_metrics_to_float(metrics: dict[str, torch.Tensor]) -> dict[str, float]:
    return {
        "loss": float(metrics["loss"].detach().cpu()),
        "loss_cls": float(metrics["loss_cls"].detach().cpu()),
        "loss_mask": float(metrics["loss_mask"].detach().cpu()),
        "loss_dice": float(metrics["loss_dice"].detach().cpu()),
        "loss_boundary": float(metrics["loss_boundary"].detach().cpu()),
        "loss_aux": float(metrics["loss_aux"].detach().cpu()),
        "pred_mean": float(metrics["pred_prob_mean"].detach().cpu()),
        "pred_max": float(metrics["pred_prob_max"].detach().cpu()),
        "object_mean": float(metrics["object_prob_mean"].detach().cpu()),
        "object_max": float(metrics["object_prob_max"].detach().cpu()),
        "active_queries": float(metrics["active_queries"].detach().cpu()),
        "num_targets": float(metrics["num_targets"].detach().cpu()),
        "num_matches": float(metrics["num_matches"].detach().cpu()),
    }


if __name__ == "__main__":
    main()
