from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare DAVIS 2017 into context-seg scene/images layout")
    parser.add_argument(
        "--davis-root",
        type=str,
        default="data/DAVIS-2017-trainval-480p",
        help="Root of the extracted DAVIS-2017-trainval-480p folder",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="data_davis",
        help="Output root with <scene>/images/<frame>.jpg",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="hardlink",
        choices=["hardlink", "copy"],
        help="hardlink saves disk space; copy duplicates image files",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    davis_root = Path(args.davis_root)
    src_root = davis_root / "DAVIS" / "JPEGImages" / "480p"
    out_root = Path(args.output_root)

    if not src_root.is_dir():
        raise SystemExit(f"DAVIS JPEGImages/480p folder not found: {src_root}")

    scenes = sorted(p for p in src_root.iterdir() if p.is_dir())
    if not scenes:
        raise SystemExit(f"No scene folders found under {src_root}")

    total = 0
    for scene_dir in scenes:
        out_images = out_root / scene_dir.name / "images"
        out_images.mkdir(parents=True, exist_ok=True)
        for src in sorted(p for p in scene_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS):
            dst = out_images / src.name
            if dst.exists():
                if args.overwrite:
                    dst.unlink()
                else:
                    total += 1
                    continue
            if args.mode == "hardlink":
                try:
                    os.link(src, dst)
                except OSError:
                    shutil.copy2(src, dst)
            else:
                shutil.copy2(src, dst)
            total += 1

    print(f"Prepared {len(scenes)} scenes / {total} frames")
    print(f"Output: {out_root.resolve()}")


if __name__ == "__main__":
    main()

