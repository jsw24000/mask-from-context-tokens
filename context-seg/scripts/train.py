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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from context_seg import (
    HungarianMatcher,
    InstanceQueryPredictor,
    MaskDecoder,
    MaskTargets,
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
    predictor = InstanceQueryPredictor(
        context_dim=context_dim,
        hidden_dim=hidden_dim,
        num_queries=queries,
        num_heads=8,
        num_layers=2,
    )
    decoder = MaskDecoder(query_dim=hidden_dim, patch_dim=context_dim, hidden_dim=hidden_dim)
    criterion = SetCriterion()

    query_embeddings = predictor(patch_tokens)
    mask_logits = decoder(query_embeddings, patch_tokens, patch_grid=patch_grid, image_size=image_size)
    targets = MaskTargets(masks=torch.randint(0, 2, (8, *image_size)).float())
    losses = criterion(mask_logits[0], targets)

    print("dry-run ok")
    print(f"query_embeddings: {tuple(query_embeddings.shape)}")
    print(f"mask_logits: {tuple(mask_logits.shape)}")
    print(f"loss: {losses['loss'].item():.4f}")


def streaming_dry_run() -> None:
    batch, frames, channels, height, width = 1, 5, 3, 56, 56
    patch_size, context_dim, hidden_dim = 14, 32, 16
    num_queries = 4

    fake_lingbot = _FakeStreamingLingbot(patch_size=patch_size, context_dim=context_dim)
    extractor = StreamingLingbotTokenExtractor(fake_lingbot, require_camera_only_cache=True)
    predictor = InstanceQueryPredictor(
        context_dim=context_dim,
        hidden_dim=hidden_dim,
        num_queries=num_queries,
        num_heads=4,
        num_layers=1,
    )
    decoder = MaskDecoder(query_dim=hidden_dim, patch_dim=context_dim, hidden_dim=hidden_dim)

    outputs = list(
        extractor.stream_sequence(
            torch.randn(batch, frames, channels, height, width),
            num_scale_frames=2,
            keyframe_interval=2,
        )
    )
    step_tokens = outputs[-1]
    query_embeddings = predictor(step_tokens.patch_tokens)
    mask_logits = decoder(query_embeddings, step_tokens.patch_tokens, step_tokens.patch_grid, step_tokens.image_size)

    print("streaming dry-run ok")
    print(f"num_outputs: {len(outputs)}")
    print(f"scale_patch_tokens: {tuple(outputs[0].patch_tokens.shape)}")
    print(f"step_patch_tokens: {tuple(step_tokens.patch_tokens.shape)}")
    print(f"mask_logits: {tuple(mask_logits.shape)}")
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

    predictor = None
    decoder = None
    optimizer = None
    matcher = HungarianMatcher(
        cost_bce=cfg["matcher"]["cost_bce"],
        cost_dice=cfg["matcher"]["cost_dice"],
        cost_size=cfg["matcher"]["cost_size"],
    )
    criterion = SetCriterion(
        bce_weight=cfg["loss"]["bce_weight"],
        dice_weight=cfg["loss"]["dice_weight"],
        matcher=matcher,
    )

    max_steps = int(cfg["train"]["max_steps"])
    log_path = output_dir / "train_log.txt"
    step = 0
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

            if predictor is None or decoder is None or optimizer is None:
                context_dim = int(frame_outputs[0].patch_tokens.shape[-1])
                predictor = InstanceQueryPredictor(
                    context_dim=context_dim,
                    hidden_dim=cfg["model"]["hidden_dim"],
                    num_queries=cfg["model"]["num_queries"],
                    num_heads=cfg["model"]["num_attention_heads"],
                    num_layers=cfg["model"]["num_predictor_layers"],
                ).to(device)
                decoder = MaskDecoder(
                    query_dim=cfg["model"]["hidden_dim"],
                    patch_dim=context_dim,
                    hidden_dim=cfg["model"]["hidden_dim"],
                ).to(device)
                optimizer = torch.optim.AdamW(
                    list(predictor.parameters()) + list(decoder.parameters()),
                    lr=cfg["train"]["lr"],
                    weight_decay=cfg["train"]["weight_decay"],
                )

            assert predictor is not None and decoder is not None and optimizer is not None
            optimizer.zero_grad(set_to_none=True)

            losses = []
            cursor = 0
            last_logits = None
            last_target = None
            for context in frame_outputs:
                query_embeddings = predictor(context.patch_tokens)
                mask_logits = decoder(query_embeddings, context.patch_tokens, context.patch_grid, context.image_size)
                _, chunk_frames = mask_logits.shape[:2]
                for local_idx in range(chunk_frames):
                    target = sample.target_masks[cursor].to(device)
                    max_targets = cfg["train"].get("max_target_masks")
                    if max_targets is not None:
                        target = target[: int(max_targets)]
                    loss_dict = criterion(mask_logits[0, local_idx], MaskTargets(masks=target))
                    losses.append(loss_dict["loss"])
                    last_logits = mask_logits[0, local_idx].detach().cpu()
                    last_target = target.detach().cpu()
                    cursor += 1

            loss = torch.stack(losses).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(predictor.parameters()) + list(decoder.parameters()),
                cfg["train"].get("grad_clip", 1.0),
            )
            optimizer.step()

            step += 1
            msg = f"step={step} scene={sample.scene} loss={loss.item():.6f}"
            print(msg)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(msg + "\n")

            if step % int(cfg["train"]["vis_interval"]) == 0 and last_logits is not None and last_target is not None:
                save_mask_visualization(output_dir / "vis" / f"step_{step:06d}.png", last_logits, last_target)
            if step % int(cfg["train"]["save_interval"]) == 0:
                save_checkpoint(output_dir / f"checkpoint_step_{step:06d}.pt", predictor, decoder, optimizer, cfg, step)

    if predictor is not None and decoder is not None and optimizer is not None:
        save_checkpoint(output_dir / "checkpoint_last.pt", predictor, decoder, optimizer, cfg, step)


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
    predictor: InstanceQueryPredictor,
    decoder: MaskDecoder,
    optimizer: torch.optim.Optimizer,
    cfg: dict[str, Any],
    step: int,
) -> None:
    torch.save(
        {
            "step": step,
            "predictor": predictor.state_dict(),
            "decoder": decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg,
        },
        path,
    )


def save_mask_visualization(path: Path, pred_logits: torch.Tensor, target_masks: torch.Tensor) -> None:
    pred = pred_logits.sigmoid().max(dim=0).values.numpy()
    if target_masks.numel() > 0:
        target = target_masks.max(dim=0).values.numpy()
    else:
        target = np.zeros_like(pred)
    pred_img = (pred * 255).clip(0, 255).astype(np.uint8)
    target_img = (target * 255).clip(0, 255).astype(np.uint8)
    canvas = np.concatenate([pred_img, target_img], axis=1)
    Image.fromarray(canvas).save(path)


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

