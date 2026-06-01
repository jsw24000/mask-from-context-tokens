from __future__ import annotations

import argparse
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
FRAME_RE = re.compile(r"(frame\.(\d+))")


@dataclass(frozen=True)
class FramePair:
    scene: str
    camera: str
    image_path: Path
    instance_path: Path
    frame_index: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert extracted Hypersim scenes into the context-seg training layout: "
            "data_root/<scene_cam>/images/*.jpg and mask_root/<scene_cam>/*.npz."
        )
    )
    parser.add_argument("--hypersim-root", type=Path, required=True, help="Directory containing extracted ai_* scenes.")
    parser.add_argument("--output-root", type=Path, required=True, help="Output image dataset root.")
    parser.add_argument("--mask-output-root", type=Path, required=True, help="Output instance-mask npz root.")
    parser.add_argument("--scenes", nargs="*", default=None, help="Optional scene ids, e.g. ai_006_001.")
    parser.add_argument("--frame-stride", type=int, default=1, help="Keep every Nth frame per camera.")
    parser.add_argument("--limit-frames", type=int, default=None, help="Maximum frames to export across all scenes/cameras.")
    parser.add_argument("--copy-mode", choices=("hardlink", "copy", "symlink"), default="hardlink")
    parser.add_argument("--min-area-pixels", type=int, default=25)
    parser.add_argument("--max-area-ratio", type=float, default=0.95)
    parser.add_argument("--ignore-zero", action="store_true", help="Also ignore instance id 0.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.frame_stride < 1:
        raise ValueError("--frame-stride must be >= 1")

    scene_dirs = find_scene_dirs(args.hypersim_root, args.scenes)
    if not scene_dirs:
        raise FileNotFoundError(f"No Hypersim scenes found under {args.hypersim_root}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    args.mask_output_root.mkdir(parents=True, exist_ok=True)

    exported = 0
    skipped = 0
    for scene_dir in scene_dirs:
        pairs = list(iter_frame_pairs(scene_dir))
        pairs = pairs[:: args.frame_stride]
        print(f"{scene_dir.name}: found {len(pairs)} usable frame/image pairs after stride={args.frame_stride}")

        for pair in pairs:
            if args.limit_frames is not None and exported >= args.limit_frames:
                print(f"Done: exported={exported}, skipped={skipped}")
                return

            out_scene = f"{pair.scene}_{pair.camera}"
            out_stem = f"{pair.frame_index:06d}"
            out_image_dir = args.output_root / out_scene / "images"
            out_mask_dir = args.mask_output_root / out_scene
            out_image_dir.mkdir(parents=True, exist_ok=True)
            out_mask_dir.mkdir(parents=True, exist_ok=True)

            image_suffix = pair.image_path.suffix.lower()
            out_image_path = out_image_dir / f"{out_stem}{image_suffix}"
            out_mask_path = out_mask_dir / f"{out_stem}.npz"

            if out_image_path.exists() and out_mask_path.exists() and not args.overwrite:
                skipped += 1
                continue

            instance_map = load_hdf5_dataset(pair.instance_path)
            masks, boxes, areas = masks_from_instance_map(
                instance_map,
                min_area_pixels=args.min_area_pixels,
                max_area_ratio=args.max_area_ratio,
                ignore_zero=args.ignore_zero,
            )
            scores = np.ones((masks.shape[0],), dtype=np.float32)

            materialize_image(pair.image_path, out_image_path, mode=args.copy_mode, overwrite=args.overwrite)
            np.savez_compressed(
                out_mask_path,
                masks=masks.astype(np.uint8),
                boxes=boxes.astype(np.float32),
                scores=scores,
                areas=areas.astype(np.float32),
            )
            exported += 1

    print(f"Done: exported={exported}, skipped={skipped}")


def find_scene_dirs(root: Path, scene_names: Iterable[str] | None) -> list[Path]:
    root = root.resolve()
    if scene_names:
        dirs: list[Path] = []
        for name in scene_names:
            candidates = []
            direct = root / name
            if direct.is_dir():
                candidates.append(direct)
            candidates.extend(p for p in root.rglob(name) if p.is_dir())
            for candidate in candidates:
                if (candidate / "images").is_dir() and candidate not in dirs:
                    dirs.append(candidate)
        return sorted(dirs)

    if (root / "images").is_dir() and root.name.startswith("ai_"):
        return [root]
    return sorted(p for p in root.rglob("ai_*_*") if p.is_dir() and (p / "images").is_dir())


def iter_frame_pairs(scene_dir: Path) -> Iterable[FramePair]:
    image_root = scene_dir / "images"
    final_dirs = sorted(p for p in image_root.iterdir() if p.is_dir() and p.name.endswith("_final_preview"))
    for final_dir in final_dirs:
        camera = final_dir.name.removesuffix("_final_preview")
        geometry_dir = image_root / f"{camera}_geometry_hdf5"
        image_paths = select_preview_images(final_dir)
        for image_path in image_paths:
            frame_prefix, frame_index = parse_frame_name(image_path)
            instance_path = geometry_dir / f"{frame_prefix}.semantic_instance.hdf5"
            if not instance_path.exists():
                matches = list(scene_dir.rglob(f"{frame_prefix}.semantic_instance.hdf5"))
                if not matches:
                    continue
                instance_path = matches[0]
            yield FramePair(
                scene=scene_dir.name,
                camera=camera,
                image_path=image_path,
                instance_path=instance_path,
                frame_index=frame_index,
            )


def select_preview_images(final_dir: Path) -> list[Path]:
    images = sorted(p for p in final_dir.iterdir() if p.is_file() and p.suffix in IMAGE_EXTS)
    tonemap = [p for p in images if "tonemap" in p.name]
    if tonemap:
        return tonemap
    color = [p for p in images if "color" in p.name]
    return color if color else images


def parse_frame_name(path: Path) -> tuple[str, int]:
    match = FRAME_RE.search(path.name)
    if match is None:
        raise ValueError(f"Cannot parse Hypersim frame name: {path}")
    return match.group(1), int(match.group(2))


def load_hdf5_dataset(path: Path) -> np.ndarray:
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("Missing dependency 'h5py'. Install it with: python -m pip install h5py") from exc

    with h5py.File(path, "r") as f:
        if "dataset" in f:
            data = f["dataset"][()]
        else:
            keys = list(f.keys())
            if not keys:
                raise KeyError(f"No datasets found in {path}")
            data = f[keys[0]][()]
    return np.asarray(data)


def masks_from_instance_map(
    instance_map: np.ndarray,
    min_area_pixels: int,
    max_area_ratio: float,
    ignore_zero: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if instance_map.ndim != 2:
        raise ValueError(f"Expected semantic_instance map [H,W], got {instance_map.shape}")

    h, w = instance_map.shape
    total = float(h * w)
    ids = np.unique(instance_map)
    ids = ids[np.isfinite(ids)]
    masks: list[np.ndarray] = []
    boxes: list[list[float]] = []
    areas: list[float] = []

    for raw_id in ids:
        if raw_id < 0:
            continue
        if ignore_zero and raw_id == 0:
            continue
        mask = instance_map == raw_id
        area = int(mask.sum())
        if area < min_area_pixels:
            continue
        if area / total > max_area_ratio:
            continue
        ys, xs = np.where(mask)
        if ys.size == 0:
            continue
        masks.append(mask.astype(np.uint8))
        boxes.append([float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)])
        areas.append(float(area))

    if not masks:
        return (
            np.zeros((0, h, w), dtype=np.uint8),
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )
    return np.stack(masks, axis=0), np.asarray(boxes, dtype=np.float32), np.asarray(areas, dtype=np.float32)


def materialize_image(src: Path, dst: Path, mode: str, overwrite: bool) -> None:
    if dst.exists():
        if overwrite:
            dst.unlink()
        else:
            return

    if mode == "copy":
        shutil.copy2(src, dst)
        return
    if mode == "symlink":
        os.symlink(src.resolve(), dst)
        return

    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


if __name__ == "__main__":
    main()
