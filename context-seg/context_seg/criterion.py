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
    objectness: float = 1.0


class SetCriterion(nn.Module):
    """Loss interface for query masks and SAM-style pseudo masks.

    Supports variable-count SAM masks by matching predicted query masks to
    targets with HungarianMatcher before computing BCE + Dice loss.
    """

    def __init__(
        self,
        bce_weight: float = 0.2,
        dice_weight: float = 2.0,
        objectness_weight: float = 1.0,
        no_object_weight: float = 0.1,
        objectness_threshold: float = 0.5,
        balanced_bce: bool = True,
        matcher: Optional[HungarianMatcher] = None,
    ) -> None:
        super().__init__()
        self.weights = LossWeights(bce=bce_weight, dice=dice_weight, objectness=objectness_weight)
        self.no_object_weight = float(no_object_weight)
        self.objectness_threshold = float(objectness_threshold)
        self.balanced_bce = bool(balanced_bce)
        self.matcher = matcher if matcher is not None else HungarianMatcher()

    def forward(
        self,
        pred_masks: torch.Tensor,
        targets: MaskTargets,
        objectness_logits: Optional[torch.Tensor] = None,
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
            return self._loss_single(pred_masks, targets.masks, objectness_logits, matched_indices)

        if pred_masks.ndim == 3:
            return self._loss_single(pred_masks, targets.masks, objectness_logits)

        if pred_masks.ndim != 4:
            raise ValueError(f"Expected pred_masks [Q,H,W] or [B,Q,H,W], got {tuple(pred_masks.shape)}")
        if objectness_logits is not None and objectness_logits.ndim != 2:
            raise ValueError(f"Expected objectness_logits [B,Q], got {tuple(objectness_logits.shape)}")

        if targets.masks.ndim != 4 or targets.masks.shape[0] != pred_masks.shape[0]:
            raise ValueError(
                "Batched SetCriterion expects targets [B,M,H,W] or aligned [B,Q,H,W]: "
                f"pred={tuple(pred_masks.shape)}, target={tuple(targets.masks.shape)}"
            )

        loss_bce_values = []
        loss_dice_values = []
        loss_objectness_values = []
        num_match_values = []
        pred_prob_mean_values = []
        pred_prob_max_values = []
        objectness_mean_values = []
        objectness_max_values = []
        active_query_values = []
        for b in range(pred_masks.shape[0]):
            sample_objectness = objectness_logits[b] if objectness_logits is not None else None
            sample_losses = self._loss_single(pred_masks[b], targets.masks[b], sample_objectness)
            loss_bce_values.append(sample_losses["loss_mask_bce"])
            loss_dice_values.append(sample_losses["loss_mask_dice"])
            loss_objectness_values.append(sample_losses["loss_objectness"])
            num_match_values.append(sample_losses["num_matches"])
            pred_prob_mean_values.append(sample_losses["pred_prob_mean"])
            pred_prob_max_values.append(sample_losses["pred_prob_max"])
            objectness_mean_values.append(sample_losses["objectness_mean"])
            objectness_max_values.append(sample_losses["objectness_max"])
            active_query_values.append(sample_losses["active_queries"])

        loss_bce = torch.stack(loss_bce_values).mean()
        loss_dice = torch.stack(loss_dice_values).mean()
        loss_objectness = torch.stack(loss_objectness_values).mean()
        num_matches = torch.stack(num_match_values).sum()
        pred_prob_mean = torch.stack(pred_prob_mean_values).mean()
        pred_prob_max = torch.stack(pred_prob_max_values).max()
        objectness_mean = torch.stack(objectness_mean_values).mean()
        objectness_max = torch.stack(objectness_max_values).max()
        active_queries = torch.stack(active_query_values).sum()
        total = self.weights.bce * loss_bce + self.weights.dice * loss_dice + self.weights.objectness * loss_objectness
        return {
            "loss": total,
            "loss_mask_bce": loss_bce,
            "loss_mask_dice": loss_dice,
            "loss_objectness": loss_objectness,
            "loss_bce": loss_bce,
            "loss_dice": loss_dice,
            "num_matches": num_matches,
            "pred_prob_mean": pred_prob_mean,
            "pred_prob_max": pred_prob_max,
            "objectness_mean": objectness_mean,
            "objectness_max": objectness_max,
            "active_queries": active_queries,
        }

    def _loss_single(
        self,
        pred_masks: torch.Tensor,
        target_masks: torch.Tensor,
        objectness_logits: Optional[torch.Tensor] = None,
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
        if objectness_logits is not None and objectness_logits.ndim != 1:
            raise ValueError(f"Expected objectness_logits [Q], got {tuple(objectness_logits.shape)}")

        target_masks = target_masks.to(dtype=pred_masks.dtype, device=pred_masks.device)
        if matched_indices is None:
            match = self.matcher(pred_masks, target_masks, objectness_logits)
            pred_idx, target_idx = match.pred_indices, match.target_indices
        else:
            if matched_indices.shape[0] != 2:
                raise ValueError("matched_indices must have shape [2, K]")
            pred_idx = matched_indices[0].to(device=pred_masks.device, dtype=torch.long)
            target_idx = matched_indices[1].to(device=pred_masks.device, dtype=torch.long)

        zero = pred_masks.sum() * 0.0
        if pred_idx.numel() == 0:
            loss_bce = zero
            loss_dice = zero
            matched_prob = pred_masks.sigmoid()
        else:
            matched_pred = pred_masks[pred_idx]
            matched_target = target_masks[target_idx]
            if self.balanced_bce:
                loss_bce = balanced_bce_with_logits(matched_pred, matched_target)
            else:
                loss_bce = F.binary_cross_entropy_with_logits(matched_pred, matched_target)
            loss_dice = dice_loss(matched_pred, matched_target)
            matched_prob = matched_pred.sigmoid()
        if objectness_logits is not None:
            loss_objectness = objectness_loss_with_logits(
                objectness_logits,
                pred_idx,
                no_object_weight=self.no_object_weight,
            )
            objectness_prob = objectness_logits.sigmoid()
            objectness_mean = objectness_prob.mean()
            objectness_max = objectness_prob.max()
            active_queries = (objectness_prob >= self.objectness_threshold).sum().to(dtype=pred_masks.dtype)
        else:
            loss_objectness = zero
            objectness_mean = zero
            objectness_max = zero
            active_queries = zero
        total = self.weights.bce * loss_bce + self.weights.dice * loss_dice + self.weights.objectness * loss_objectness
        return {
            "loss": total,
            "loss_mask_bce": loss_bce,
            "loss_mask_dice": loss_dice,
            "loss_objectness": loss_objectness,
            "loss_bce": loss_bce,
            "loss_dice": loss_dice,
            "num_matches": torch.as_tensor(float(pred_idx.numel()), device=pred_masks.device),
            "pred_prob_mean": matched_prob.mean(),
            "pred_prob_max": matched_prob.max(),
            "objectness_mean": objectness_mean,
            "objectness_max": objectness_max,
            "active_queries": active_queries,
        }


def balanced_bce_with_logits(
    pred_masks: torch.Tensor,
    target_masks: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Average foreground and background BCE so empty background cannot dominate."""
    bce = F.binary_cross_entropy_with_logits(pred_masks, target_masks, reduction="none")
    target = target_masks.to(dtype=bce.dtype)
    foreground = target
    background = 1.0 - target
    fg_loss = (bce * foreground).flatten(1).sum(dim=1) / foreground.flatten(1).sum(dim=1).clamp_min(eps)
    bg_loss = (bce * background).flatten(1).sum(dim=1) / background.flatten(1).sum(dim=1).clamp_min(eps)
    return (0.5 * (fg_loss + bg_loss)).mean()


def objectness_loss_with_logits(
    objectness_logits: torch.Tensor,
    matched_pred_indices: torch.Tensor,
    no_object_weight: float = 0.1,
) -> torch.Tensor:
    target = torch.zeros_like(objectness_logits)
    weights = torch.full_like(objectness_logits, float(no_object_weight))
    if matched_pred_indices.numel() > 0:
        target[matched_pred_indices] = 1.0
        weights[matched_pred_indices] = 1.0
    loss = F.binary_cross_entropy_with_logits(objectness_logits, target, reduction="none")
    return (loss * weights).sum() / weights.sum().clamp_min(1e-6)


def dice_loss(pred_masks: torch.Tensor, target_masks: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = pred_masks.sigmoid().flatten(2)
    target = target_masks.flatten(2)
    numerator = 2 * (pred * target).sum(dim=-1)
    denominator = pred.sum(dim=-1) + target.sum(dim=-1)
    return (1 - (numerator + eps) / (denominator + eps)).mean()
