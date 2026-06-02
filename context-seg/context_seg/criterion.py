from __future__ import annotations

from typing import Dict, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .matcher import HungarianMatcher
from .types import MaskTargets


class SetCriterion(nn.Module):
    """Mask2Former-style set criterion for class-agnostic instance segmentation."""

    def __init__(
        self,
        class_weight: float = 2.0,
        mask_weight: float = 5.0,
        dice_weight: float = 5.0,
        no_object_weight: float = 0.1,
        boundary_weight: float = 0.0,
        boundary_size: int = 128,
        boundary_kernel_size: int = 3,
        num_points: int = 12544,
        oversample_ratio: float = 3.0,
        importance_sample_ratio: float = 0.75,
        matcher: Optional[HungarianMatcher] = None,
        **legacy_kwargs,
    ) -> None:
        super().__init__()
        if "bce_weight" in legacy_kwargs:
            mask_weight = float(legacy_kwargs["bce_weight"])
        if "objectness_weight" in legacy_kwargs:
            class_weight = float(legacy_kwargs["objectness_weight"])
        self.class_weight = float(class_weight)
        self.mask_weight = float(mask_weight)
        self.dice_weight = float(dice_weight)
        self.no_object_weight = float(no_object_weight)
        self.boundary_weight = float(boundary_weight)
        self.boundary_size = int(boundary_size)
        self.boundary_kernel_size = int(boundary_kernel_size)
        self.num_points = int(num_points)
        self.oversample_ratio = float(oversample_ratio)
        self.importance_sample_ratio = float(importance_sample_ratio)
        self.matcher = matcher if matcher is not None else HungarianMatcher()

    def forward(
        self,
        pred_masks: torch.Tensor,
        targets: MaskTargets,
        pred_logits: Optional[torch.Tensor] = None,
        matched_indices: Optional[torch.Tensor] = None,
        aux_outputs: tuple[Mapping[str, torch.Tensor], ...] = (),
        include_boundary: bool = True,
    ) -> Dict[str, torch.Tensor]:
        if pred_masks.ndim == 3:
            losses = self._loss_single(pred_masks, targets.masks, pred_logits, matched_indices, include_boundary)
        elif pred_masks.ndim == 4:
            if targets.masks.ndim != 4 or targets.masks.shape[0] != pred_masks.shape[0]:
                raise ValueError(
                    f"Batched SetCriterion expects targets [B,M,H,W], got pred={tuple(pred_masks.shape)} "
                    f"target={tuple(targets.masks.shape)}"
                )
            values = [
                self._loss_single(
                    pred_masks[b],
                    targets.masks[b],
                    pred_logits[b] if pred_logits is not None else None,
                    include_boundary=include_boundary,
                )
                for b in range(pred_masks.shape[0])
            ]
            losses = average_loss_dicts(values)
        else:
            raise ValueError(f"Expected pred_masks [Q,H,W] or [B,Q,H,W], got {tuple(pred_masks.shape)}")

        if aux_outputs:
            aux_values = []
            for aux in aux_outputs:
                aux_values.append(self.forward(aux["pred_masks"], targets, aux.get("pred_logits"), include_boundary=False))
            aux_losses = average_loss_dicts(aux_values)
            losses["loss_aux"] = aux_losses["loss"]
            losses["loss"] = losses["loss"] + losses["loss_aux"]
        else:
            losses["loss_aux"] = pred_masks.sum() * 0.0
        return losses

    def _loss_single(
        self,
        pred_masks: torch.Tensor,
        target_masks: torch.Tensor,
        pred_logits: Optional[torch.Tensor] = None,
        matched_indices: Optional[torch.Tensor] = None,
        include_boundary: bool = True,
    ) -> Dict[str, torch.Tensor]:
        if pred_masks.ndim != 3 or target_masks.ndim != 3:
            raise ValueError(f"Expected pred [Q,H,W] and target [M,H,W], got {tuple(pred_masks.shape)} and {tuple(target_masks.shape)}")
        if pred_logits is not None and pred_logits.ndim != 2:
            raise ValueError(f"Expected pred_logits [Q,2], got {tuple(pred_logits.shape)}")

        target_masks = target_masks.to(dtype=pred_masks.dtype, device=pred_masks.device)
        if matched_indices is None:
            match = self.matcher(pred_masks, target_masks, pred_logits)
            pred_idx, target_idx = match.pred_indices, match.target_indices
        else:
            pred_idx = matched_indices[0].to(device=pred_masks.device, dtype=torch.long)
            target_idx = matched_indices[1].to(device=pred_masks.device, dtype=torch.long)

        zero = pred_masks.sum() * 0.0
        if pred_logits is not None:
            target_classes = torch.zeros(pred_masks.shape[0], dtype=torch.long, device=pred_masks.device)
            if pred_idx.numel() > 0:
                target_classes[pred_idx] = 1
            class_weights = torch.as_tensor([self.no_object_weight, 1.0], dtype=pred_masks.dtype, device=pred_masks.device)
            loss_cls = F.cross_entropy(pred_logits, target_classes, weight=class_weights)
            object_prob = pred_logits.softmax(dim=-1)[:, 1]
            active_queries = (object_prob >= 0.5).sum().to(dtype=pred_masks.dtype)
            object_prob_mean = object_prob.mean()
            object_prob_max = object_prob.max()
        else:
            loss_cls = zero
            active_queries = zero
            object_prob_mean = zero
            object_prob_max = zero

        if pred_idx.numel() > 0:
            matched_pred = pred_masks[pred_idx]
            matched_target = target_masks[target_idx]
            point_coords = sample_uncertain_points(
                matched_pred,
                num_points=self.num_points,
                oversample_ratio=self.oversample_ratio,
                importance_sample_ratio=self.importance_sample_ratio,
            )
            pred_points = point_sample(matched_pred[:, None], point_coords)[:, 0]
            target_points = point_sample(matched_target[:, None], point_coords)[:, 0]
            loss_mask = F.binary_cross_entropy_with_logits(pred_points, target_points)
            loss_dice = dice_loss(pred_points, target_points)
            if include_boundary and self.boundary_weight > 0:
                loss_boundary = boundary_dice_loss(
                    matched_pred,
                    matched_target,
                    size=self.boundary_size,
                    kernel_size=self.boundary_kernel_size,
                )
            else:
                loss_boundary = zero
            pred_prob = pred_points.sigmoid()
            pred_prob_mean = pred_prob.mean()
            pred_prob_max = pred_prob.max()
        else:
            loss_mask = zero
            loss_dice = zero
            loss_boundary = zero
            pred_prob_mean = zero
            pred_prob_max = zero

        total = (
            self.class_weight * loss_cls
            + self.mask_weight * loss_mask
            + self.dice_weight * loss_dice
            + self.boundary_weight * loss_boundary
        )
        return {
            "loss": total,
            "loss_cls": loss_cls,
            "loss_mask": loss_mask,
            "loss_dice": loss_dice,
            "loss_boundary": loss_boundary,
            "loss_mask_bce": loss_mask,
            "loss_mask_dice": loss_dice,
            "loss_objectness": loss_cls,
            "num_targets": torch.as_tensor(float(target_masks.shape[0]), device=pred_masks.device),
            "num_matches": torch.as_tensor(float(pred_idx.numel()), device=pred_masks.device),
            "active_queries": active_queries,
            "object_prob_mean": object_prob_mean,
            "object_prob_max": object_prob_max,
            "objectness_mean": object_prob_mean,
            "objectness_max": object_prob_max,
            "pred_prob_mean": pred_prob_mean,
            "pred_prob_max": pred_prob_max,
        }


def average_loss_dicts(losses: list[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    keys = losses[0].keys()
    out = {}
    for key in keys:
        values = torch.stack([item[key] for item in losses])
        out[key] = values.sum() if key in {"active_queries", "num_matches", "num_targets"} else values.mean()
    return out


def sample_uncertain_points(
    logits: torch.Tensor,
    num_points: int,
    oversample_ratio: float,
    importance_sample_ratio: float,
) -> torch.Tensor:
    batch = logits.shape[0]
    num_sampled = max(num_points, int(num_points * oversample_ratio))
    coords = torch.rand(batch, num_sampled, 2, device=logits.device, dtype=logits.dtype)
    sampled_logits = point_sample(logits[:, None], coords)[:, 0]
    uncertainty = -sampled_logits.abs()
    num_uncertain = int(num_points * importance_sample_ratio)
    num_uncertain = min(num_uncertain, num_points)
    num_random = num_points - num_uncertain
    if num_uncertain > 0:
        top_idx = torch.topk(uncertainty, k=num_uncertain, dim=1).indices
        batch_idx = torch.arange(batch, device=logits.device)[:, None]
        uncertain_coords = coords[batch_idx, top_idx]
    else:
        uncertain_coords = coords[:, :0]
    if num_random > 0:
        random_coords = torch.rand(batch, num_random, 2, device=logits.device, dtype=logits.dtype)
        return torch.cat([uncertain_coords, random_coords], dim=1)
    return uncertain_coords


def point_sample(input: torch.Tensor, point_coords: torch.Tensor) -> torch.Tensor:
    grid = point_coords.mul(2).sub(1)
    grid = grid[:, :, None, :]
    output = F.grid_sample(input.float(), grid.float(), mode="bilinear", align_corners=False)
    return output[:, :, :, 0].to(dtype=input.dtype)


def dice_loss(pred_logits: torch.Tensor, target_masks: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = pred_logits.sigmoid()
    target = target_masks.float()
    numerator = 2 * (pred * target).sum(dim=1)
    denominator = pred.sum(dim=1) + target.sum(dim=1)
    return (1 - (numerator + eps) / (denominator + eps)).mean()


def boundary_dice_loss(
    pred_logits: torch.Tensor,
    target_masks: torch.Tensor,
    size: int = 128,
    kernel_size: int = 3,
    eps: float = 1e-6,
) -> torch.Tensor:
    if size > 0 and max(pred_logits.shape[-2:]) > size:
        pred_logits = F.interpolate(pred_logits[:, None], size=(size, size), mode="bilinear", align_corners=False)[:, 0]
        target_masks = F.interpolate(target_masks[:, None].float(), size=(size, size), mode="nearest")[:, 0]
    pred_boundary = soft_boundary_map(pred_logits.sigmoid(), kernel_size=kernel_size)
    target_boundary = soft_boundary_map(target_masks.float(), kernel_size=kernel_size)
    numerator = 2 * (pred_boundary * target_boundary).flatten(1).sum(dim=1)
    denominator = pred_boundary.flatten(1).sum(dim=1) + target_boundary.flatten(1).sum(dim=1)
    return (1 - (numerator + eps) / (denominator + eps)).mean()


def soft_boundary_map(masks: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    if masks.ndim != 3:
        raise ValueError(f"Expected masks [M,H,W], got {tuple(masks.shape)}")
    kernel_size = max(3, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
    pad = kernel_size // 2
    x = masks[:, None].float()
    dilated = F.max_pool2d(x, kernel_size=kernel_size, stride=1, padding=pad)
    eroded = -F.max_pool2d(-x, kernel_size=kernel_size, stride=1, padding=pad)
    return (dilated - eroded).clamp(0, 1)[:, 0].to(dtype=masks.dtype)
