from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .matcher import HungarianMatcher
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

    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        matcher: Optional[HungarianMatcher] = None,
    ) -> None:
        super().__init__()
        # 中文导读：BCE 约束逐像素分类，Dice 更关注 mask 区域重叠；
        # 第一版先保留两个最常用的 mask loss。
        self.weights = LossWeights(bce=bce_weight, dice=dice_weight)
        self.matcher = matcher if matcher is not None else HungarianMatcher()

    def forward(
        self,
        pred_masks: torch.Tensor,
        targets: MaskTargets,
        matched_indices: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute mask losses for matched predictions.

        Args:
            pred_masks: [Q, H, W] or [B, Q, H, W] logits.
            targets: MaskTargets with masks [M, H, W], [B, M, H, W], or
                aligned [B, Q, H, W].
            matched_indices: Optional precomputed [2, K] indices for single-sample loss.
        """
        if matched_indices is not None:
            return self._loss_single(pred_masks, targets.masks, matched_indices)

        if pred_masks.ndim == 3:
            return self._loss_single(pred_masks, targets.masks)

        if pred_masks.ndim != 4:
            raise ValueError(f"Expected pred_masks [Q,H,W] or [B,Q,H,W], got {tuple(pred_masks.shape)}")

        if pred_masks.shape != targets.masks.shape:
            if targets.masks.ndim != 4 or targets.masks.shape[0] != pred_masks.shape[0]:
                raise ValueError(
                    "Batched SetCriterion expects targets [B,M,H,W] or aligned [B,Q,H,W]: "
                    f"pred={tuple(pred_masks.shape)}, target={tuple(targets.masks.shape)}"
                )

        losses = []
        loss_bce_values = []
        loss_dice_values = []
        for b in range(pred_masks.shape[0]):
            sample_losses = self._loss_single(pred_masks[b], targets.masks[b])
            losses.append(sample_losses["loss"])
            loss_bce_values.append(sample_losses["loss_bce"])
            loss_dice_values.append(sample_losses["loss_dice"])

        loss_bce = torch.stack(loss_bce_values).mean()
        loss_dice = torch.stack(loss_dice_values).mean()
        total = self.weights.bce * loss_bce + self.weights.dice * loss_dice
        return {
            "loss": total,
            "loss_bce": loss_bce,
            "loss_dice": loss_dice,
        }

    def _loss_single(
        self,
        pred_masks: torch.Tensor,
        target_masks: torch.Tensor,
        matched_indices: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if pred_masks.ndim == 4 and pred_masks.shape[0] == 1:
            pred_masks = pred_masks[0]
        if target_masks.ndim == 4 and target_masks.shape[0] == 1:
            target_masks = target_masks[0]
        if pred_masks.ndim != 3 or target_masks.ndim != 3:
            raise ValueError(
                f"Expected single-sample masks [Q,H,W] and [M,H,W], got "
                f"{tuple(pred_masks.shape)} and {tuple(target_masks.shape)}"
            )

        target_masks = target_masks.to(dtype=pred_masks.dtype, device=pred_masks.device)
        if matched_indices is None:
            match = self.matcher(pred_masks, target_masks)
            pred_idx, target_idx = match.pred_indices, match.target_indices
        else:
            if matched_indices.shape[0] != 2:
                raise ValueError("matched_indices must have shape [2, K]")
            pred_idx = matched_indices[0].to(device=pred_masks.device, dtype=torch.long)
            target_idx = matched_indices[1].to(device=pred_masks.device, dtype=torch.long)

        if pred_idx.numel() == 0:
            zero = pred_masks.sum() * 0.0
            return {"loss": zero, "loss_bce": zero, "loss_dice": zero}

        matched_pred = pred_masks[pred_idx]
        matched_target = target_masks[target_idx]
        loss_bce = F.binary_cross_entropy_with_logits(matched_pred, matched_target)
        loss_dice = dice_loss(matched_pred, matched_target)
        total = self.weights.bce * loss_bce + self.weights.dice * loss_dice
        return {"loss": total, "loss_bce": loss_bce, "loss_dice": loss_dice}


def dice_loss(pred_masks: torch.Tensor, target_masks: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # 中文导读：Dice loss 在前景像素很少时比单纯 BCE 更稳，适合实例 mask。
    pred = pred_masks.sigmoid().flatten(2)
    target = target_masks.flatten(2)
    numerator = 2 * (pred * target).sum(dim=-1)
    denominator = pred.sum(dim=-1) + target.sum(dim=-1)
    return (1 - (numerator + eps) / (denominator + eps)).mean()
