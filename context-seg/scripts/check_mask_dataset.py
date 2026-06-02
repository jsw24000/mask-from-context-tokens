from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".JPG", ".JPEG")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check image/mask alignment for context-seg scene datasets.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--num-vis", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    scenes = find_scenes(args.data_root, args.scenes)
    if not scenes:
        raise FileNotFoundError(f"No scene/images folders found under {args.data_root}")

    all_pairs = []
    print("scene image_count mask_count missing_masks empty_masks mean_targets max_targets")
    for scene in scenes:
        images = sorted(p for p in (args.data_root / scene / "images").iterdir() if p.suffix in IMAGE_EXTS)
        mask_dir = args.mask_root / scene
        masks = sorted(mask_dir.glob("*.npz")) if mask_dir.is_dir() else []
        missing = []
        empty = 0
        target_counts = []
        for image_path in images:
            mask_path = mask_dir / f"{image_path.stem}.npz"
            if not mask_path.exists():
                missing.append(mask_path)
                continue
            count = load_mask_count(mask_path)
            target_counts.append(count)
            if count == 0:
                empty += 1
            all_pairs.append((scene, image_path, mask_path))
        mean_targets = float(np.mean(target_counts)) if target_counts else 0.0
        max_targets = int(np.max(target_counts)) if target_counts else 0
        print(f"{scene} {len(images)} {len(masks)} {len(missing)} {empty} {mean_targets:.2f} {max_targets}")

    if args.output_dir is not None and all_pairs:
        rng = random.Random(args.seed)
        selected = rng.sample(all_pairs, k=min(int(args.num_vis), len(all_pairs)))
        for idx, (scene, image_path, mask_path) in enumerate(selected):
            save_gt_visualization(args.output_dir / f"{idx:03d}_{scene}_{image_path.stem}.png", image_path, mask_path)
        print(f"Saved {len(selected)} GT visualizations to {args.output_dir}")


def find_scenes(root: Path, selected: list[str] | None) -> list[str]:
    scenes = sorted(p.name for p in root.iterdir() if (p / "images").is_dir())
    if selected:
        wanted = set(selected)
        scenes = [scene for scene in scenes if scene in wanted]
    return scenes


def load_mask_count(path: Path) -> int:
    with np.load(path) as data:
        if "masks" not in data:
            raise KeyError(f"Missing masks in {path}")
        masks = data["masks"]
    if masks.ndim != 3:
        raise ValueError(f"Expected masks [M,H,W] in {path}, got {masks.shape}")
    return int(masks.shape[0])


def save_gt_visualization(path: Path, image_path: Path, mask_path: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    with np.load(mask_path) as data:
        masks = data["masks"]
    if masks.size == 0:
        union = np.zeros((image.height, image.width), dtype=bool)
    else:
        union = masks.max(axis=0) > 0
        if union.shape != (image.height, image.width):
            union_img = Image.fromarray(union.astype(np.uint8) * 255)
            union = np.asarray(union_img.resize((image.width, image.height), Image.Resampling.NEAREST)) > 0

    overlay = np.asarray(image).astype(np.float32) * 0.65
    overlay[union] = 0.35 * overlay[union] + 0.65 * np.array([0, 255, 0], dtype=np.float32)
    overlay = overlay.clip(0, 255).astype(np.uint8)

    label_h = 22
    panel = Image.new("RGB", (image.width * 2, image.height + label_h), color=(0, 0, 0))
    panel.paste(image, (0, label_h))
    panel.paste(Image.fromarray(overlay), (image.width, label_h))
    draw = ImageDraw.Draw(panel)
    draw.text((4, 4), "rgb", fill=255)
    draw.text((image.width + 4, 4), f"gt_union masks={masks.shape[0] if masks.ndim == 3 else 0}", fill=255)
    panel.save(path)


if __name__ == "__main__":
    main()
