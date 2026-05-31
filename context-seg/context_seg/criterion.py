from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .types import MaskTargets


@dataclass(frozen=True)
class LossWeights:
    bce: float = 1.0
    dice: float = 1.0


class SetCriterion(nn.Module):
    """Loss interface for query masks and SAM-style pseudo masks.

    This scaffold supports aligned query-target pairs. Set matching
    (Hungarian/IoU matching) is intentionally left as the next implementation
    step once target cache format and training data are fixed.
    """

    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0) -> None:
        super().__init__()
        # BCE 约束逐像素分类，Dice 更关注 mask 区域重叠；
        # 第一版先保留两个最常用的 mask loss。
        self.weights = LossWeights(bce=bce_weight, dice=dice_weight)

    def forward(
        self,
        pred_masks: torch.Tensor,
        targets: MaskTargets,
        matched_indices: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute mask losses for matched predictions.

        Args:
            pred_masks: [B, Q, H, W] logits.
            targets: MaskTargets with masks [B, Q, H, W] for this scaffold.
            matched_indices: Reserved for future set matching.
        """
        if matched_indices is not None:
            raise NotImplementedError("matched_indices support will be added with set matching")
        if pred_masks.shape != targets.masks.shape:
            # 这里暂时要求 query 和目标 mask 已经一一对齐。
            # 真正训练时通常需要 Hungarian matching 或基于 IoU 的匹配，把预测 query 对到 SAM masks。
            raise ValueError(
                "SetCriterion scaffold expects aligned pred/target masks with identical shapes: "
                f"pred={tuple(pred_masks.shape)}, target={tuple(targets.masks.shape)}"
            )

        target_masks = targets.masks.to(dtype=pred_masks.dtype, device=pred_masks.device)
        loss_bce = F.binary_cross_entropy_with_logits(pred_masks, target_masks)
        loss_dice = dice_loss(pred_masks, target_masks)
        total = self.weights.bce * loss_bce + self.weights.dice * loss_dice
        return {
            "loss": total,
            "loss_bce": loss_bce,
            "loss_dice": loss_dice,
        }


def dice_loss(pred_masks: torch.Tensor, target_masks: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # 中文导读：Dice loss 在前景像素很少时比单纯 BCE 更稳，适合实例 mask。
    pred = pred_masks.sigmoid().flatten(2)
    target = target_masks.flatten(2)
    numerator = 2 * (pred * target).sum(dim=-1)
    denominator = pred.sum(dim=-1) + target.sum(dim=-1)
    return (1 - (numerator + eps) / (denominator + eps)).mean()
