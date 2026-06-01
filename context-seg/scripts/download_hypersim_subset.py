from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path


DEFAULT_BASE_URL = "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1"
DEFAULT_SCENES = ["ai_006_001"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a small subset of Hypersim scene ZIP files.")
    parser.add_argument("--scenes", nargs="+", default=DEFAULT_SCENES, help="Hypersim scene ids, e.g. ai_006_001")
    parser.add_argument("--downloads-dir", type=Path, default=Path("hypersim_downloads"))
    parser.add_argument("--decompress-dir", type=Path, default=None, help="Optional directory to unzip scenes into.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--dry-run", action="store_true", help="Only print URLs and remote sizes.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--delete-zip-after-extract", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.downloads_dir.mkdir(parents=True, exist_ok=True)
    if args.decompress_dir is not None:
        args.decompress_dir.mkdir(parents=True, exist_ok=True)

    for scene in args.scenes:
        url = f"{args.base_url.rstrip('/')}/scenes/{scene}.zip"
        out_path = args.downloads_dir / f"{scene}.zip"
        size = remote_size(url)
        size_text = human_size(size) if size is not None else "unknown size"
        print(f"{scene}: {url} ({size_text})")
        if args.dry_run:
            continue

        if out_path.exists() and not args.overwrite:
            print(f"  found existing {out_path}, skipping download")
        else:
            download(url, out_path)

        if args.decompress_dir is not None:
            extract_scene(out_path, args.decompress_dir, overwrite=args.overwrite)
            if args.delete_zip_after_extract:
                out_path.unlink()
                print(f"  deleted {out_path}")


def remote_size(url: str) -> int | None:
    request = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            value = response.headers.get("Content-Length")
            return int(value) if value is not None else None
    except Exception as exc:
        print(f"  warning: cannot query remote size: {exc}", file=sys.stderr)
        return None


def download(url: str, out_path: Path) -> None:
    print(f"  downloading to {out_path}")
    curl = shutil.which("curl")
    if curl is not None:
        subprocess.run([curl, "-L", "-C", "-", "--fail", "-o", str(out_path), url], check=True)
        return

    wget = shutil.which("wget")
    if wget is not None:
        subprocess.run([wget, "-c", "-O", str(out_path), url], check=True)
        return

    print("  curl/wget not found; falling back to urllib without resume support")
    with urllib.request.urlopen(url) as response, open(out_path, "wb") as f:
        shutil.copyfileobj(response, f)


def extract_scene(zip_path: Path, decompress_dir: Path, overwrite: bool) -> None:
    scene_dir = decompress_dir / zip_path.stem
    if scene_dir.exists() and not overwrite:
        print(f"  found existing {scene_dir}, skipping extraction")
        return
    print(f"  extracting to {decompress_dir}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(decompress_dir)


def human_size(num_bytes: int | None) -> str:
    if num_bytes is None:
        return "unknown"
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TB"


if __name__ == "__main__":
    main()
