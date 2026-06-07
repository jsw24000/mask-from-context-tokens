from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from context_seg import (  # noqa: E402
    Mask2FormerLiteHead,
    MaskRefinementHead,
    PseudoMaskFilter,
    StreamingLingbotTokenExtractor,
    VideoFrameDataset,
    collate_single_clip,
)
from train import build_mask_refiner, load_config, resolve_path  # noqa: E402


PALETTE = np.asarray(
    [
        (230, 25, 75),
        (60, 180, 75),
        (0, 130, 200),
        (245, 130, 48),
        (145, 30, 180),
        (70, 240, 240),
        (240, 50, 230),
        (210, 245, 60),
        (250, 190, 190),
        (0, 128, 128),
        (230, 190, 255),
        (170, 110, 40),
        (255, 250, 200),
        (128, 0, 0),
        (170, 255, 195),
        (128, 128, 0),
        (255, 215, 180),
        (0, 0, 128),
        (128, 128, 128),
        (255, 255, 255),
    ],
    dtype=np.uint8,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create report-ready prediction-vs-GT instance segmentation demo figures."
    )
    parser.add_argument("--config", type=str, default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--checkpoint", type=str, required=True, help="Training checkpoint containing seg_head weights.")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--mask-root", type=str, default=None)
    parser.add_argument("--lingbot-checkpoint", type=str, default=None)
    parser.add_argument(
        "--split",
        choices=("val", "train", "all"),
        default="val",
        help="Scene split to sample when --scenes is not provided.",
    )
    parser.add_argument("--scenes", nargs="*", default=None, help="Exact scene folder names. Defaults to val_limit_scenes.")
    parser.add_argument("--limit-clips", type=int, default=4)
    parser.add_argument("--num-frames", type=int, default=8, help="Maximum demo frames to save.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--objectness-threshold", type=float, default=None)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument("--min-area-ratio", type=float, default=None)
    parser.add_argument("--raw-gt", action="store_true", help="Show all GT masks instead of config-filtered GT masks.")
    parser.add_argument("--disable-refinement", action="store_true", help="Ignore mask_refiner even if the checkpoint has one.")
    parser.add_argument("--save-panels", action="store_true", help="Also save separate rgb/pred/gt panel images.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    apply_overrides(cfg, args)

    device = torch.device(cfg["train"]["device"] if torch.cuda.is_available() else "cpu")
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_panels:
        (output_dir / "panels").mkdir(exist_ok=True)

    limit_scenes = args.scenes if args.scenes is not None else scenes_for_split(cfg, args.split)
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

    extractor = build_extractor(cfg, device)
    checkpoint = torch.load(resolve_path(args.checkpoint), map_location=device, weights_only=False)
    if "seg_head" not in checkpoint:
        raise KeyError(f"Checkpoint does not contain 'seg_head': {args.checkpoint}")

    mask_refiner = build_refiner_for_demo(cfg, checkpoint, device, disabled=args.disable_refinement)
    gt_filter = PseudoMaskFilter(
        min_area_ratio=cfg.get("pseudo_mask", {}).get("min_area_ratio", 0.001),
        max_area_ratio=cfg.get("pseudo_mask", {}).get("max_area_ratio", 0.6),
        nms_iou=cfg.get("pseudo_mask", {}).get("nms_iou", 0.8),
        max_masks=cfg["train"].get("max_target_masks", 20),
    )

    objectness_threshold = (
        float(args.objectness_threshold)
        if args.objectness_threshold is not None
        else float(cfg["train"].get("objectness_threshold", 0.5))
    )
    max_instances = int(args.max_instances or cfg["train"].get("max_active_queries", 20))
    min_area_ratio = (
        float(args.min_area_ratio)
        if args.min_area_ratio is not None
        else float(cfg["train"].get("min_vis_mask_area_ratio", 0.001))
    )

    rng = random.Random(args.seed)
    saved = 0
    seg_head: Mask2FormerLiteHead | None = None
    print(
        "Demo configuration: "
        f"device={device}, clips={len(dataset)}, scenes={limit_scenes}, "
        f"objectness_threshold={objectness_threshold}, output_dir={output_dir}"
    )

    with torch.no_grad(), tqdm(total=len(dataset), desc="rendering demo", unit="clip") as progress:
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
                seg_head = build_seg_head(cfg, context_dim, device)
                seg_head.load_state_dict(checkpoint["seg_head"], strict=True)
                seg_head.eval()

            frame_indices = list(range(len(sample.image_ids)))
            rng.shuffle(frame_indices)
            keep_for_clip = set(frame_indices[: max(1, min(2, len(frame_indices)))])

            cursor = 0
            assert seg_head is not None
            for context in frame_outputs:
                head_output = seg_head(context.patch_tokens, context.patch_grid, context.image_size)
                mask_logits = head_output.mask_logits
                pred_logits = head_output.pred_logits
                _, chunk_frames = mask_logits.shape[:2]

                for local_idx in range(chunk_frames):
                    if cursor not in keep_for_clip:
                        cursor += 1
                        continue

                    rgb = tensor_to_rgb(images[cursor].detach().cpu())
                    coarse_logits = mask_logits[0, local_idx]
                    if mask_refiner is not None:
                        final_logits = mask_refiner(images[cursor : cursor + 1], coarse_logits)
                    else:
                        final_logits = coarse_logits

                    pred_masks, pred_scores = select_pred_instances(
                        final_logits.detach().cpu(),
                        pred_logits[0, local_idx].detach().cpu(),
                        objectness_threshold=objectness_threshold,
                        mask_threshold=args.mask_threshold,
                        max_instances=max_instances,
                        min_area_ratio=min_area_ratio,
                    )
                    gt_masks = sample.target_masks[cursor]
                    if not args.raw_gt:
                        gt_masks = gt_filter(gt_masks, max_masks=cfg["train"].get("max_target_masks"))
                    gt_masks = resize_masks_to(gt_masks.detach().cpu(), rgb.shape[:2])

                    image_id = sample.image_ids[cursor].replace("/", "_")
                    title = f"{image_id}  pred={len(pred_masks)}  gt={int(gt_masks.shape[0])}"
                    out_path = output_dir / f"{saved:03d}_{image_id}.png"
                    save_demo_figure(
                        out_path,
                        rgb,
                        pred_masks,
                        gt_masks,
                        title=title,
                        pred_scores=pred_scores,
                    )
                    if args.save_panels:
                        save_separate_panels(output_dir / "panels", f"{saved:03d}_{image_id}", rgb, pred_masks, gt_masks)

                    saved += 1
                    cursor += 1
                    if saved >= int(args.num_frames):
                        print(f"Saved {saved} demo figures to {output_dir}")
                        return

            progress.update(1)

    print(f"Saved {saved} demo figures to {output_dir}")


def apply_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    if args.data_root is not None:
        cfg["data"]["root"] = args.data_root
    if args.mask_root is not None:
        cfg["data"]["mask_root"] = args.mask_root
    if args.lingbot_checkpoint is not None:
        cfg["lingbot"]["checkpoint"] = args.lingbot_checkpoint


def scenes_for_split(cfg: dict[str, Any], split: str) -> list[str] | None:
    train_scenes = cfg["data"].get("limit_scenes") or []
    val_scenes = cfg["data"].get("val_limit_scenes") or []
    if split == "train":
        return train_scenes
    if split == "val":
        return val_scenes
    if split == "all":
        scenes = []
        seen = set()
        for scene in [*train_scenes, *val_scenes]:
            if scene not in seen:
                scenes.append(scene)
                seen.add(scene)
        return scenes or None
    raise ValueError(f"Unknown split: {split}")


def build_extractor(cfg: dict[str, Any], device: torch.device) -> StreamingLingbotTokenExtractor:
    return StreamingLingbotTokenExtractor.from_checkpoint(
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


def build_seg_head(cfg: dict[str, Any], context_dim: int, device: torch.device) -> Mask2FormerLiteHead:
    return Mask2FormerLiteHead(
        context_dim=context_dim,
        hidden_dim=cfg["model"]["hidden_dim"],
        pixel_dim=cfg["model"].get("pixel_dim", cfg["model"]["hidden_dim"]),
        num_queries=cfg["model"]["num_queries"],
        num_heads=cfg["model"]["num_attention_heads"],
        decoder_layers=cfg["model"].get("decoder_layers", cfg["model"].get("num_predictor_layers", 6)),
        ffn_dim=cfg["model"].get("ffn_dim", 4 * cfg["model"]["hidden_dim"]),
    ).to(device)


def build_refiner_for_demo(
    cfg: dict[str, Any],
    checkpoint: dict[str, Any],
    device: torch.device,
    disabled: bool,
) -> MaskRefinementHead | None:
    if disabled or not cfg["model"].get("use_refinement", False):
        return None
    if "mask_refiner" not in checkpoint:
        print("Checkpoint has no mask_refiner; demo will use coarse mask logits.")
        return None
    refiner = build_mask_refiner(cfg).to(device)
    refiner.load_state_dict(checkpoint["mask_refiner"], strict=True)
    refiner.eval()
    return refiner


def select_pred_instances(
    mask_logits: torch.Tensor,
    class_logits: torch.Tensor,
    objectness_threshold: float,
    mask_threshold: float,
    max_instances: int,
    min_area_ratio: float,
) -> tuple[list[np.ndarray], list[float]]:
    object_scores = class_logits.softmax(dim=-1)[:, 1]
    mask_probs = mask_logits.sigmoid()
    order = torch.argsort(object_scores, descending=True)
    masks: list[np.ndarray] = []
    scores: list[float] = []
    image_area = float(mask_probs.shape[-2] * mask_probs.shape[-1])

    for idx in order.tolist():
        score = float(object_scores[idx])
        if score < objectness_threshold:
            continue
        mask = mask_probs[idx] >= mask_threshold
        area_ratio = float(mask.sum()) / image_area
        if area_ratio < min_area_ratio:
            continue
        masks.append(mask.numpy().astype(bool))
        scores.append(score)
        if len(masks) >= max_instances:
            break
    return masks, scores


def resize_masks_to(masks: torch.Tensor, image_hw: tuple[int, int]) -> torch.Tensor:
    if masks.numel() == 0:
        return torch.zeros((0, image_hw[0], image_hw[1]), dtype=torch.float32)
    if tuple(masks.shape[-2:]) == image_hw:
        return masks.float()
    import torch.nn.functional as F

    return F.interpolate(masks[:, None].float(), size=image_hw, mode="nearest")[:, 0]


def tensor_to_rgb(image: torch.Tensor) -> np.ndarray:
    arr = image.clamp(0, 1).permute(1, 2, 0).numpy()
    return (arr * 255).round().clip(0, 255).astype(np.uint8)


def save_demo_figure(
    path: Path,
    rgb: np.ndarray,
    pred_masks: list[np.ndarray],
    gt_masks: torch.Tensor,
    title: str,
    pred_scores: list[float],
) -> None:
    gt_list = [gt_masks[i].numpy() > 0.5 for i in range(gt_masks.shape[0])]
    pred_overlay = color_overlay(rgb, pred_masks)
    gt_overlay = color_overlay(rgb, gt_list)
    pred_color = color_instances(pred_masks, rgb.shape[:2])
    gt_color = color_instances(gt_list, rgb.shape[:2])

    panels = [
        panel("RGB", rgb),
        panel("Prediction overlay", pred_overlay),
        panel("Prediction instances", pred_color),
        panel("GT overlay", gt_overlay),
        panel("GT instances", gt_color),
    ]
    label_h = 26
    title_h = 28
    width = sum(item.width for item in panels)
    height = title_h + max(item.height for item in panels)
    canvas = Image.new("RGB", (width, height), color=(0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    score_text = ", ".join(f"{score:.2f}" for score in pred_scores[:6])
    draw.text((8, 6), f"{title}  pred scores: [{score_text}]", fill=(255, 255, 255))
    x = 0
    for item in panels:
        canvas.paste(item, (x, title_h))
        x += item.width
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def save_separate_panels(
    output_dir: Path,
    stem: str,
    rgb: np.ndarray,
    pred_masks: list[np.ndarray],
    gt_masks: torch.Tensor,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    gt_list = [gt_masks[i].numpy() > 0.5 for i in range(gt_masks.shape[0])]
    Image.fromarray(rgb).save(output_dir / f"{stem}_rgb.png")
    Image.fromarray(color_overlay(rgb, pred_masks)).save(output_dir / f"{stem}_pred_overlay.png")
    Image.fromarray(color_instances(pred_masks, rgb.shape[:2])).save(output_dir / f"{stem}_pred_instances.png")
    Image.fromarray(color_overlay(rgb, gt_list)).save(output_dir / f"{stem}_gt_overlay.png")
    Image.fromarray(color_instances(gt_list, rgb.shape[:2])).save(output_dir / f"{stem}_gt_instances.png")


def panel(title: str, image: np.ndarray) -> Image.Image:
    label_h = 24
    pil = Image.fromarray(image).convert("RGB")
    out = Image.new("RGB", (pil.width, pil.height + label_h), color=(0, 0, 0))
    out.paste(pil, (0, label_h))
    ImageDraw.Draw(out).text((6, 5), title, fill=(255, 255, 255))
    return out


def color_overlay(rgb: np.ndarray, masks: list[np.ndarray], alpha: float = 0.58) -> np.ndarray:
    out = rgb.astype(np.float32) * 0.72
    for idx, mask in enumerate(masks):
        if mask.shape != rgb.shape[:2]:
            mask = resize_numpy_mask(mask, rgb.shape[:2])
        color = PALETTE[idx % len(PALETTE)].astype(np.float32)
        out[mask] = (1.0 - alpha) * out[mask] + alpha * color
    return out.clip(0, 255).astype(np.uint8)


def color_instances(masks: list[np.ndarray], image_hw: tuple[int, int]) -> np.ndarray:
    out = np.zeros((image_hw[0], image_hw[1], 3), dtype=np.uint8)
    for idx, mask in enumerate(masks):
        if mask.shape != image_hw:
            mask = resize_numpy_mask(mask, image_hw)
        out[mask] = PALETTE[idx % len(PALETTE)]
    return out


def resize_numpy_mask(mask: np.ndarray, image_hw: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    resized = image.resize((image_hw[1], image_hw[0]), Image.Resampling.NEAREST)
    return np.asarray(resized) > 0


if __name__ == "__main__":
    main()
