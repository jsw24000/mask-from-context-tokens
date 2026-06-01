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
        # root 下每张图对应一个 `.npz`，文件名由 image_id 决定。
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

        # 第一版只约定离线 SAM 伪标签的最小字段。
        # 后续如果改成 COCO RLE/json，只需要替换这个 provider，不影响训练模块。
        data = np.load(path)
        if self.mask_key not in data:
            raise KeyError(f"Missing mask key {self.mask_key!r} in {path}")

        masks = torch.as_tensor(data[self.mask_key]).float()
        # boxes/scores/areas 不是当前 loss 必需，但保留下来方便后续做过滤、
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


class PseudoMaskFilter:
    """Filter noisy SAM automatic masks before set-prediction training."""

    def __init__(
        self,
        min_area_ratio: float = 0.001,
        max_area_ratio: float = 0.6,
        nms_iou: float = 0.8,
        max_masks: int = 20,
    ) -> None:
        self.min_area_ratio = float(min_area_ratio)
        self.max_area_ratio = float(max_area_ratio)
        self.nms_iou = float(nms_iou)
        self.max_masks = int(max_masks)

    def __call__(
        self,
        masks: torch.Tensor,
        scores: Optional[torch.Tensor] = None,
        max_masks: Optional[int] = None,
    ) -> torch.Tensor:
        if masks.ndim != 3:
            raise ValueError(f"Expected masks [M,H,W], got {tuple(masks.shape)}")
        if masks.shape[0] == 0:
            return masks
        max_masks = int(max_masks if max_masks is not None else self.max_masks)
        masks = (masks > 0.5).float()
        areas = masks.flatten(1).sum(dim=1)
        image_area = float(masks.shape[-2] * masks.shape[-1])
        keep = (areas >= self.min_area_ratio * image_area) & (areas <= self.max_area_ratio * image_area)
        indices = torch.nonzero(keep, as_tuple=False).flatten()
        if indices.numel() == 0:
            return masks[:0]

        if scores is None:
            order_scores = areas[indices]
        else:
            order_scores = scores.to(device=masks.device, dtype=masks.dtype)[indices]
        indices = indices[torch.argsort(order_scores, descending=True)]

        selected: list[torch.Tensor] = []
        flat_masks = masks.flatten(1)
        for idx in indices:
            candidate = flat_masks[idx]
            suppress = False
            for kept_idx in selected:
                kept = flat_masks[kept_idx]
                intersection = (candidate * kept).sum()
                union = candidate.sum() + kept.sum() - intersection
                iou = intersection / union.clamp_min(1e-6)
                if float(iou) >= self.nms_iou:
                    suppress = True
                    break
            if not suppress:
                selected.append(idx)
            if len(selected) >= max_masks:
                break
        if not selected:
            return masks[:0]
        return masks[torch.stack(selected).long()]


def _optional_tensor(data: np.lib.npyio.NpzFile, *keys: str) -> Optional[torch.Tensor]:
    # 中文导读：兼容不同伪标签脚本的字段命名，比如 boxes 和 bboxes。
    for key in keys:
        if key in data:
            return torch.as_tensor(data[key]).float()
    return None
