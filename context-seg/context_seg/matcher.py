from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class MatchResult:
    pred_indices: torch.Tensor
    target_indices: torch.Tensor


class HungarianMatcher:
    """Mask2Former-style Hungarian matcher for class-agnostic instance masks."""

    def __init__(
        self,
        cost_class: float = 2.0,
        cost_mask: float = 5.0,
        cost_dice: float = 5.0,
        cost_size: int = 128,
        cost_bce: float | None = None,
        cost_objectness: float | None = None,
    ) -> None:
        self.cost_class = float(cost_class)
        self.cost_mask = float(cost_mask if cost_bce is None else cost_bce)
        self.cost_dice = float(cost_dice)
        self.cost_size = int(cost_size)
        self.cost_objectness = float(cost_objectness or 0.0)

    @torch.no_grad()
    def __call__(
        self,
        pred_masks: torch.Tensor,
        target_masks: torch.Tensor,
        pred_logits: torch.Tensor | None = None,
    ) -> MatchResult:
        if pred_masks.ndim != 3:
            raise ValueError(f"Expected pred_masks [Q,H,W], got {tuple(pred_masks.shape)}")
        if target_masks.ndim != 3:
            raise ValueError(f"Expected target_masks [M,H,W], got {tuple(target_masks.shape)}")
        if pred_logits is not None and pred_logits.ndim != 2:
            raise ValueError(f"Expected pred_logits [Q,2], got {tuple(pred_logits.shape)}")

        q = pred_masks.shape[0]
        m = target_masks.shape[0]
        device = pred_masks.device
        if q == 0 or m == 0:
            empty = torch.empty(0, dtype=torch.long, device=device)
            return MatchResult(empty, empty)

        pred_small, target_small = _resize_for_cost(pred_masks, target_masks, self.cost_size)
        pred_flat = pred_small.flatten(1)
        target_flat = target_small.flatten(1).to(pred_flat.dtype)

        mask_cost = _pairwise_bce_with_logits(pred_flat, target_flat)
        dice_cost = _pairwise_dice_cost(pred_flat, target_flat)
        if pred_logits is not None:
            class_cost = -pred_logits.softmax(dim=-1)[:, 1][:, None].expand(q, m)
        else:
            class_cost = torch.zeros_like(dice_cost)
        cost = self.cost_class * class_cost + self.cost_mask * mask_cost + self.cost_dice * dice_cost

        row, col = linear_sum_assignment(cost.detach().cpu().numpy())
        return MatchResult(
            pred_indices=torch.as_tensor(row, dtype=torch.long, device=device),
            target_indices=torch.as_tensor(col, dtype=torch.long, device=device),
        )


def _resize_for_cost(
    pred_masks: torch.Tensor,
    target_masks: torch.Tensor,
    cost_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    h, w = pred_masks.shape[-2:]
    if max(h, w) <= cost_size:
        return pred_masks, target_masks
    pred = F.interpolate(pred_masks[:, None], size=(cost_size, cost_size), mode="bilinear", align_corners=False)[:, 0]
    target = F.interpolate(target_masks[:, None].float(), size=(cost_size, cost_size), mode="nearest")[:, 0]
    return pred, target


def _pairwise_bce_with_logits(pred_flat: torch.Tensor, target_flat: torch.Tensor) -> torch.Tensor:
    q, n = pred_flat.shape
    m = target_flat.shape[0]
    pred = pred_flat[:, None, :].expand(q, m, n)
    target = target_flat[None, :, :].expand(q, m, n)
    return F.binary_cross_entropy_with_logits(pred, target, reduction="none").mean(dim=-1)


def _pairwise_dice_cost(pred_flat: torch.Tensor, target_flat: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = pred_flat.sigmoid()
    target = target_flat.float()
    numerator = 2 * torch.einsum("qn,mn->qm", pred, target)
    denominator = pred.sum(dim=1)[:, None] + target.sum(dim=1)[None, :]
    return 1 - (numerator + eps) / (denominator + eps)
