from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .types import PseudoMasks


class PseudoMaskProvider:
    """Read offline SAM-style pseudo masks from disk.

    The initial cache format is NPZ:
        masks: [M, H, W] binary or bool array
        boxes or bboxes: optional [M, 4]
        scores: optional [M]
        areas: optional [M]
    """

    def __init__(
        self,
        root: str | Path,
        mask_key: str = "masks",
        suffix: str = ".npz",
        device: Optional[torch.device | str] = None,
    ) -> None:
        # 中文导读：root 下每张图对应一个 `.npz`，文件名由 image_id 决定。
        # 例如 image_id="000001" 时默认读取 root/000001.npz。
        self.root = Path(root)
        self.mask_key = mask_key
        self.suffix = suffix
        self.device = device

    def path_for(self, image_id: str) -> Path:
        return self.root / f"{image_id}{self.suffix}"

    def get(self, image_id: str) -> PseudoMasks:
        path = self.path_for(image_id)
        if not path.exists():
            raise FileNotFoundError(f"Pseudo mask cache not found: {path}")

        # 中文导读：第一版只约定离线 SAM 伪标签的最小字段。
        # 后续如果改成 COCO RLE/json，只需要替换这个 provider，不影响训练模块。
        data = np.load(path)
        if self.mask_key not in data:
            raise KeyError(f"Missing mask key {self.mask_key!r} in {path}")

        masks = torch.as_tensor(data[self.mask_key]).float()
        # 中文导读：boxes/scores/areas 不是当前 loss 必需，但保留下来方便后续做过滤、
        # matching 或按 SAM 置信度加权。
        boxes = _optional_tensor(data, "boxes", "bboxes")
        scores = _optional_tensor(data, "scores")
        areas = _optional_tensor(data, "areas")

        if self.device is not None:
            masks = masks.to(self.device)
            boxes = boxes.to(self.device) if boxes is not None else None
            scores = scores.to(self.device) if scores is not None else None
            areas = areas.to(self.device) if areas is not None else None

        return PseudoMasks(
            image_id=image_id,
            masks=masks,
            boxes=boxes,
            scores=scores,
            areas=areas,
        )


def _optional_tensor(data: np.lib.npyio.NpzFile, *keys: str) -> Optional[torch.Tensor]:
    # 中文导读：兼容不同伪标签脚本的字段命名，比如 boxes 和 bboxes。
    for key in keys:
        if key in data:
            return torch.as_tensor(data[key]).float()
    return None
