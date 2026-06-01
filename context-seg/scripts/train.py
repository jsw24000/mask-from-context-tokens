from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from context_seg import (
    HungarianMatcher,
    MaskTargets,
    QueryInstanceHead,
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
    query_head = QueryInstanceHead(
        context_dim=context_dim,
        hidden_dim=hidden_dim,
        num_queries=queries,
        num_heads=8,
        num_layers=2,
    )
    criterion = SetCriterion()

    output = query_head(patch_tokens, patch_grid=patch_grid, image_size=image_size)
    targets = MaskTargets(masks=torch.randint(0, 2, (8, *image_size)).float())
    losses = criterion(output.mask_logits[0], targets, output.objectness_logits[0])

    print("dry-run ok")
    print(f"query_embeddings: {tuple(output.query_embeddings.shape)}")
    print(f"objectness_logits: {tuple(output.objectness_logits.shape)}")
    print(f"mask_logits: {tuple(output.mask_logits.shape)}")
    print(f"loss: {losses['loss'].item():.4f}")


def streaming_dry_run() -> None:
    batch, frames, channels, height, width = 1, 5, 3, 56, 56
    patch_size, context_dim, hidden_dim = 14, 32, 16
    num_queries = 4

    fake_lingbot = _FakeStreamingLingbot(patch_size=patch_size, context_dim=context_dim)
    extractor = StreamingLingbotTokenExtractor(fake_lingbot, require_camera_only_cache=True)
    query_head = QueryInstanceHead(
        context_dim=context_dim,
        hidden_dim=hidden_dim,
        num_queries=num_queries,
        num_heads=4,
        num_layers=1,
    )

    outputs = list(
        extractor.stream_sequence(
            torch.randn(batch, frames, channels, height, width),
            num_scale_frames=2,
            keyframe_interval=2,
        )
    )
    step_tokens = outputs[-1]
    head_output = query_head(step_tokens.patch_tokens, step_tokens.patch_grid, step_tokens.image_size)

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
        kv_cache_camera_only=True,
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

    query_head = None
    optimizer = None
    matcher = HungarianMatcher(
        cost_bce=cfg["matcher"]["cost_bce"],
        cost_dice=cfg["matcher"]["cost_dice"],
        cost_objectness=cfg["matcher"].get("cost_objectness", 1.0),
        cost_size=cfg["matcher"]["cost_size"],
    )
    criterion = SetCriterion(
        bce_weight=cfg["loss"].get("mask_bce_weight", cfg["loss"].get("bce_weight", 0.2)),
        dice_weight=cfg["loss"].get("mask_dice_weight", cfg["loss"].get("dice_weight", 2.0)),
        objectness_weight=cfg["loss"].get("objectness_weight", 1.0),
        no_object_weight=cfg["loss"].get("no_object_weight", 0.1),
        objectness_threshold=cfg["train"].get("objectness_threshold", 0.5),
        balanced_bce=cfg["loss"].get("balanced_bce", True),
        matcher=matcher,
    )

    max_steps = int(cfg["train"]["max_steps"])
    log_path = output_dir / "train_log.txt"
    if not log_path.exists() or log_path.stat().st_size == 0:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(
                "step scene loss loss_mask_bce loss_mask_dice loss_objectness "
                "pred_mean pred_max objectness_mean objectness_max active_queries num_matches\n"
            )

    print_training_summary(cfg, device, output_dir, lingbot_ckpt, len(dataset))
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

                if query_head is None or optimizer is None:
                    context_dim = int(frame_outputs[0].patch_tokens.shape[-1])
                    query_head = QueryInstanceHead(
                        context_dim=context_dim,
                        hidden_dim=cfg["model"]["hidden_dim"],
                        num_queries=cfg["model"]["num_queries"],
                        num_heads=cfg["model"]["num_attention_heads"],
                        num_layers=cfg["model"]["num_predictor_layers"],
                        mask_hidden_dim=cfg["model"].get("mask_hidden_dim"),
                    ).to(device)
                    optimizer = torch.optim.AdamW(
                        query_head.parameters(),
                        lr=cfg["train"]["lr"],
                        weight_decay=cfg["train"]["weight_decay"],
                    )

                assert query_head is not None and optimizer is not None
                optimizer.zero_grad(set_to_none=True)

                loss_dicts = []
                cursor = 0
                last_logits = None
                last_objectness = None
                last_target = None
                for context in frame_outputs:
                    head_output = query_head(context.patch_tokens, context.patch_grid, context.image_size)
                    mask_logits = head_output.mask_logits
                    objectness_logits = head_output.objectness_logits
                    _, chunk_frames = mask_logits.shape[:2]
                    for local_idx in range(chunk_frames):
                        target = sample.target_masks[cursor].to(device)
                        max_targets = cfg["train"].get("max_target_masks")
                        if max_targets is not None:
                            target = target[: int(max_targets)]
                        loss_dict = criterion(
                            mask_logits[0, local_idx],
                            MaskTargets(masks=target),
                            objectness_logits[0, local_idx],
                        )
                        loss_dicts.append(loss_dict)
                        last_logits = mask_logits[0, local_idx].detach().cpu()
                        last_objectness = objectness_logits[0, local_idx].detach().cpu()
                        last_target = target.detach().cpu()
                        cursor += 1

                metrics = average_loss_dicts(loss_dicts)
                loss = metrics["loss"]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    query_head.parameters(),
                    cfg["train"].get("grad_clip", 1.0),
                )
                optimizer.step()

                step += 1
                log_values = {
                    "loss": float(metrics["loss"].detach().cpu()),
                    "loss_mask_bce": float(metrics["loss_mask_bce"].detach().cpu()),
                    "loss_mask_dice": float(metrics["loss_mask_dice"].detach().cpu()),
                    "loss_objectness": float(metrics["loss_objectness"].detach().cpu()),
                    "pred_mean": float(metrics["pred_prob_mean"].detach().cpu()),
                    "pred_max": float(metrics["pred_prob_max"].detach().cpu()),
                    "objectness_mean": float(metrics["objectness_mean"].detach().cpu()),
                    "objectness_max": float(metrics["objectness_max"].detach().cpu()),
                    "active_queries": float(metrics["active_queries"].detach().cpu()),
                    "num_matches": float(metrics["num_matches"].detach().cpu()),
                }
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"{step} {sample.scene} {log_values['loss']:.6f} {log_values['loss_mask_bce']:.6f} "
                        f"{log_values['loss_mask_dice']:.6f} {log_values['loss_objectness']:.6f} "
                        f"{log_values['pred_mean']:.6f} {log_values['pred_max']:.6f} "
                        f"{log_values['objectness_mean']:.6f} {log_values['objectness_max']:.6f} "
                        f"{log_values['active_queries']:.1f} {log_values['num_matches']:.1f}\n"
                    )

                progress.set_postfix(
                    scene=sample.scene,
                    loss=f"{log_values['loss']:.4f}",
                    mbce=f"{log_values['loss_mask_bce']:.4f}",
                    mdice=f"{log_values['loss_mask_dice']:.4f}",
                    obj=f"{log_values['loss_objectness']:.4f}",
                    omean=f"{log_values['objectness_mean']:.3f}",
                    omax=f"{log_values['objectness_max']:.3f}",
                    active=f"{log_values['active_queries']:.0f}",
                    matches=f"{log_values['num_matches']:.0f}",
                )
                progress.update(1)

                if (
                    step % int(cfg["train"]["vis_interval"]) == 0
                    and last_logits is not None
                    and last_target is not None
                    and last_objectness is not None
                ):
                    save_mask_visualization(
                        output_dir / "vis" / f"step_{step:06d}.png",
                        last_logits,
                        last_target,
                        last_objectness,
                        objectness_threshold=cfg["train"].get("objectness_threshold", 0.5),
                        max_active_queries=cfg["train"].get("max_active_queries"),
                    )
                if step % int(cfg["train"]["save_interval"]) == 0:
                    save_checkpoint(output_dir / f"checkpoint_step_{step:06d}.pt", query_head, optimizer, cfg, step)

    if query_head is not None and optimizer is not None:
        save_checkpoint(output_dir / "checkpoint_last.pt", query_head, optimizer, cfg, step)


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


def save_checkpoint(
    path: Path,
    query_head: QueryInstanceHead,
    optimizer: torch.optim.Optimizer,
    cfg: dict[str, Any],
    step: int,
) -> None:
    torch.save(
        {
            "step": step,
            "query_head": query_head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg,
        },
        path,
    )


def average_loss_dicts(loss_dicts: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not loss_dicts:
        raise ValueError("No losses were produced for this training step")
    keys = (
        "loss",
        "loss_mask_bce",
        "loss_mask_dice",
        "loss_objectness",
        "pred_prob_mean",
        "pred_prob_max",
        "objectness_mean",
        "objectness_max",
        "active_queries",
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
) -> None:
    print("Training configuration")
    print(f"  device: {device}")
    print(f"  output_dir: {output_dir}")
    print(f"  data.root: {resolve_path(cfg['data']['root'])}")
    print(f"  data.mask_root: {resolve_path(cfg['data']['mask_root'])}")
    print(f"  lingbot.checkpoint: {lingbot_ckpt}")
    print(f"  clips: {num_clips}")
    print(f"  max_steps: {cfg['train']['max_steps']}")
    print(f"  num_queries: {cfg['model']['num_queries']}")
    print(
        "  loss: "
        f"mask_bce_weight={cfg['loss'].get('mask_bce_weight', cfg['loss'].get('bce_weight', 0.2))}, "
        f"mask_dice_weight={cfg['loss'].get('mask_dice_weight', cfg['loss'].get('dice_weight', 2.0))}, "
        f"objectness_weight={cfg['loss'].get('objectness_weight', 1.0)}, "
        f"no_object_weight={cfg['loss'].get('no_object_weight', 0.1)}, "
        f"balanced_bce={cfg['loss'].get('balanced_bce', True)}"
    )


def save_mask_visualization(
    path: Path,
    pred_logits: torch.Tensor,
    target_masks: torch.Tensor,
    objectness_logits: torch.Tensor,
    top_k: int = 4,
    objectness_threshold: float = 0.5,
    max_active_queries: int | None = None,
) -> None:
    pred_prob = pred_logits.sigmoid()
    objectness_prob = objectness_logits.sigmoid()
    active_indices = torch.nonzero(objectness_prob >= float(objectness_threshold), as_tuple=False).flatten()
    if max_active_queries is not None and active_indices.numel() > int(max_active_queries):
        top_active = torch.argsort(objectness_prob[active_indices], descending=True)[: int(max_active_queries)]
        active_indices = active_indices[top_active]
    if active_indices.numel() > 0:
        pred = pred_prob[active_indices].max(dim=0).values.numpy()
    else:
        pred = np.zeros(pred_prob.shape[-2:], dtype=np.float32)
    if target_masks.numel() > 0:
        target = target_masks.max(dim=0).values.numpy()
    else:
        target = np.zeros_like(pred)
    panels = [_to_uint8(pred), _to_uint8(target)]
    query_scores = objectness_prob
    top_indices = torch.argsort(query_scores, descending=True)[:top_k]
    for idx in top_indices:
        panels.append(_to_uint8(pred_prob[int(idx)].numpy()))
    while len(panels) < 2 + top_k:
        panels.append(np.zeros_like(panels[0]))
    canvas = np.concatenate(panels, axis=1)
    Image.fromarray(canvas).save(path)
    stats_path = path.with_suffix(".txt")
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write(f"pred_min {float(pred_prob.min()):.6f}\n")
        f.write(f"pred_mean {float(pred_prob.mean()):.6f}\n")
        f.write(f"pred_max {float(pred_prob.max()):.6f}\n")
        f.write(f"objectness_min {float(objectness_prob.min()):.6f}\n")
        f.write(f"objectness_mean {float(objectness_prob.mean()):.6f}\n")
        f.write(f"objectness_max {float(objectness_prob.max()):.6f}\n")
        f.write(f"active_query_indices {[int(i) for i in active_indices]}\n")
        f.write(f"top_query_indices {[int(i) for i in top_indices]}\n")
        f.write(f"top_query_scores {[float(query_scores[int(i)]) for i in top_indices]}\n")


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    return (arr * 255).clip(0, 255).astype(np.uint8)


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
