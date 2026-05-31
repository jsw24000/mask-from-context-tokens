from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from context_seg.dataset import IMAGE_EXTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate offline SAM pseudo masks for scene/images datasets")
    parser.add_argument("--config", type=str, default=None, help="Optional context-seg yaml config")
    parser.add_argument("--image-root", type=str, required=True, help="Dataset root with <scene>/images/*.png")
    parser.add_argument("--output-root", type=str, required=True, help="Output root for <scene>/<frame>.npz masks")
    parser.add_argument("--sam-checkpoint", type=str, default=None)
    parser.add_argument("--model-type", type=str, default=None, choices=["vit_b", "vit_l", "vit_h"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--points-per-side", type=int, default=32)
    parser.add_argument("--pred-iou-thresh", type=float, default=0.88)
    parser.add_argument("--stability-score-thresh", type=float, default=0.95)
    parser.add_argument("--min-mask-region-area", type=int, default=100)
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for smoke tests")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_config_defaults(args)
    if not args.sam_checkpoint:
        raise SystemExit("Missing --sam-checkpoint or sam.checkpoint in config")
    if not args.model_type:
        args.model_type = "vit_b"
    try:
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency 'segment-anything'. Install the original SAM package first, "
            "then rerun this script."
        ) from exc

    image_root = Path(args.image_root)
    output_root = Path(args.output_root)
    images = collect_scene_images(image_root)
    if args.limit is not None:
        images = images[: int(args.limit)]
    if not images:
        raise SystemExit(f"No scene/images files found under {image_root}")

    sam = sam_model_registry[args.model_type](checkpoint=args.sam_checkpoint)
    sam.to(device=args.device)
    generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=args.points_per_side,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        min_mask_region_area=args.min_mask_region_area,
    )

    for scene, image_path in tqdm(images, desc="Generating SAM masks"):
        out_path = output_root / scene / f"{image_path.stem}.npz"
        if out_path.exists() and not args.overwrite:
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)

        image = np.asarray(Image.open(image_path).convert("RGB"))
        annotations = generator.generate(image)
        masks, boxes, scores, areas = pack_annotations(annotations, image.shape[:2])
        np.savez_compressed(
            out_path,
            masks=masks,
            boxes=boxes,
            scores=scores,
            areas=areas,
        )

    print(f"Saved SAM pseudo masks to {output_root}")


def apply_config_defaults(args: argparse.Namespace) -> None:
    if args.config is None:
        return
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    sam_cfg = cfg.get("sam", {}) if isinstance(cfg, dict) else {}
    if args.sam_checkpoint is None:
        args.sam_checkpoint = sam_cfg.get("checkpoint")
    if args.model_type is None:
        args.model_type = sam_cfg.get("model_type")


def collect_scene_images(root: Path) -> list[tuple[str, Path]]:
    results: list[tuple[str, Path]] = []
    for scene_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        image_dir = scene_dir / "images"
        if not image_dir.is_dir():
            continue
        for path in sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix in IMAGE_EXTS):
            results.append((scene_dir.name, path))
    return results


def pack_annotations(
    annotations: list[dict],
    image_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h, w = image_hw
    if not annotations:
        return (
            np.zeros((0, h, w), dtype=np.uint8),
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    annotations = sorted(annotations, key=lambda item: item.get("area", 0), reverse=True)
    masks = np.stack([ann["segmentation"].astype(np.uint8) for ann in annotations])
    boxes = np.asarray([ann["bbox"] for ann in annotations], dtype=np.float32)
    scores = np.asarray([ann.get("predicted_iou", 1.0) for ann in annotations], dtype=np.float32)
    areas = np.asarray([ann.get("area", float(mask.sum())) for ann, mask in zip(annotations, masks)], dtype=np.float32)
    return masks, boxes, scores, areas


if __name__ == "__main__":
    main()
