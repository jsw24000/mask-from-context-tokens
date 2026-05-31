from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass(frozen=True)
class ContextTokens:
    """Token bundle returned by LingbotTokenExtractor.

    Attributes:
        tokens: Full selected token tensor, shaped [B, S, N_all, C].
        patch_tokens: Patch-only tokens, shaped [B, S, N_patch, C].
        patch_grid: Patch grid as (patch_h, patch_w).
        image_size: Image size as (H, W).
        patch_start_idx: Index where patch tokens begin inside tokens.
    """

    tokens: torch.Tensor
    patch_tokens: torch.Tensor
    patch_grid: Tuple[int, int]
    image_size: Tuple[int, int]
    patch_start_idx: int


@dataclass(frozen=True)
class PseudoMasks:
    """Offline SAM-style pseudo masks for one image."""

    image_id: str
    masks: torch.Tensor
    boxes: Optional[torch.Tensor] = None
    scores: Optional[torch.Tensor] = None
    areas: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class MaskTargets:
    """Training target container consumed by SetCriterion."""

    masks: torch.Tensor
    boxes: Optional[torch.Tensor] = None
    labels: Optional[torch.Tensor] = None

