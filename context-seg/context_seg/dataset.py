from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".JPG", ".JPEG")


@dataclass(frozen=True)
class VideoClipSample:
    images: torch.Tensor
    target_masks: list[torch.Tensor]
    image_ids: list[str]
    image_paths: list[Path]
    scene: str


class VideoFrameDataset(Dataset):
    """Scan scene/images folders and return fixed-length video clips.

    第一版训练实验使用简单 resize，让图像和 SAM mask 在同一分辨率下对齐。
    如果后续要完全复刻 LingBot 的 crop/pad 预处理，只需要替换这里的 transform。
    """

    def __init__(
        self,
        root: str | Path,
        mask_root: str | Path,
        clip_length: int = 16,
        image_size: int | Sequence[int] = 518,
        clip_stride: int = 1,
        limit_scenes: int | str | Sequence[str] | None = None,
        limit_clips: int | None = None,
    ) -> None:
        self.root = Path(root)
        self.mask_root = Path(mask_root)
        self.clip_length = int(clip_length)
        self.image_size = _to_hw(image_size)
        self.clip_stride = int(clip_stride)

        if self.clip_length < 1:
            raise ValueError("clip_length must be >= 1")
        if self.clip_stride < 1:
            raise ValueError("clip_stride must be >= 1")

        scenes = _find_scenes(self.root)
        if limit_scenes is not None:
            scenes = _limit_scenes(scenes, limit_scenes)

        samples: list[tuple[str, list[Path]]] = []
        for scene, image_paths in scenes:
            max_start = len(image_paths) - self.clip_length
            if max_start < 0:
                continue
            for start in range(0, max_start + 1, self.clip_stride):
                samples.append((scene, image_paths[start : start + self.clip_length]))
                if limit_clips is not None and len(samples) >= int(limit_clips):
                    break
            if limit_clips is not None and len(samples) >= int(limit_clips):
                break

        if not samples:
            raise ValueError(f"No clips found under {self.root} with clip_length={self.clip_length}")
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> VideoClipSample:
        scene, image_paths = self.samples[index]
        image_ids = [f"{scene}/{p.stem}" for p in image_paths]
        images = torch.stack([_load_image(path, self.image_size) for path in image_paths])
        target_masks = [self._load_masks(image_id) for image_id in image_ids]
        return VideoClipSample(
            images=images,
            target_masks=target_masks,
            image_ids=image_ids,
            image_paths=image_paths,
            scene=scene,
        )

    def _load_masks(self, image_id: str) -> torch.Tensor:
        path = self.mask_root / f"{image_id}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Missing pseudo-mask cache: {path}")

        data = np.load(path)
        if "masks" not in data:
            raise KeyError(f"Missing 'masks' in {path}")
        masks = data["masks"]
        if masks.ndim != 3:
            raise ValueError(f"Expected masks [M,H,W] in {path}, got {masks.shape}")
        resized = [_resize_mask(mask, self.image_size) for mask in masks]
        if not resized:
            return torch.zeros((0, self.image_size[0], self.image_size[1]), dtype=torch.float32)
        return torch.stack(resized).float()


def collate_single_clip(batch: list[VideoClipSample]) -> VideoClipSample:
    if len(batch) != 1:
        raise ValueError("The first training scaffold supports batch_size=1")
    return batch[0]


def _find_scenes(root: Path) -> list[tuple[str, list[Path]]]:
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    scenes: list[tuple[str, list[Path]]] = []
    for scene_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        image_dir = scene_dir / "images"
        if not image_dir.is_dir():
            continue
        image_paths = _list_images(image_dir)
        if image_paths:
            scenes.append((scene_dir.name, image_paths))
    return scenes


def _list_images(root: Path) -> list[Path]:
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix in IMAGE_EXTS)


def _limit_scenes(
    scenes: list[tuple[str, list[Path]]],
    limit_scenes: int | str | Sequence[str],
) -> list[tuple[str, list[Path]]]:
    # 支持 limit_scenes: 4 取前 N 个，也支持 ["bear"] 这种指定场景名。
    if isinstance(limit_scenes, int):
        return scenes[:limit_scenes]
    if isinstance(limit_scenes, str):
        names = {limit_scenes}
    else:
        names = {str(name) for name in limit_scenes}
    return [(scene, paths) for scene, paths in scenes if scene in names]


def _load_image(path: Path, image_size: tuple[int, int]) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((image_size[1], image_size[0]), Image.Resampling.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _resize_mask(mask: np.ndarray, image_size: tuple[int, int]) -> torch.Tensor:
    mask_img = Image.fromarray(mask.astype(np.uint8))
    mask_img = mask_img.resize((image_size[1], image_size[0]), Image.Resampling.NEAREST)
    arr = np.asarray(mask_img, dtype=np.float32)
    if arr.max() > 1:
        arr = arr / 255.0
    return torch.from_numpy((arr > 0.5).astype(np.float32))


def _to_hw(image_size: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(image_size, int):
        return (image_size, image_size)
    values = tuple(int(v) for v in image_size)
    if len(values) != 2:
        raise ValueError(f"image_size must be int or (H, W), got {image_size}")
    return values
