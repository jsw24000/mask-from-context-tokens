from __future__ import annotations

import torch
import torch.nn as nn


class MaskRefinementHead(nn.Module):
    """Lightweight RGB-guided residual mask refinement.

    The head keeps query assignment unchanged: each query's coarse mask is
    refined independently with shared weights and shallow RGB features.
    """

    def __init__(
        self,
        refine_dim: int = 8,
        hidden_dim: int = 16,
        query_chunk_size: int = 10,
        residual_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.query_chunk_size = int(query_chunk_size)
        self.residual_scale = float(residual_scale)
        if self.query_chunk_size < 1:
            raise ValueError("query_chunk_size must be >= 1")

        self.rgb_encoder = nn.Sequential(
            nn.Conv2d(3, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(4 if hidden_dim % 4 == 0 else 1, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, refine_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(refine_dim + 1, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(4 if hidden_dim % 4 == 0 else 1, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )

    def forward(self, rgb: torch.Tensor, coarse_logits: torch.Tensor) -> torch.Tensor:
        """Return refined logits with the same shape as coarse_logits.

        Args:
            rgb: [1, 3, H, W] or [3, H, W], values normally in [0, 1].
            coarse_logits: [Q, H, W].
        """
        if rgb.ndim == 3:
            rgb = rgb.unsqueeze(0)
        if rgb.ndim != 4 or rgb.shape[0] != 1 or rgb.shape[1] != 3:
            raise ValueError(f"Expected rgb [1,3,H,W] or [3,H,W], got {tuple(rgb.shape)}")
        if coarse_logits.ndim != 3:
            raise ValueError(f"Expected coarse_logits [Q,H,W], got {tuple(coarse_logits.shape)}")
        if tuple(rgb.shape[-2:]) != tuple(coarse_logits.shape[-2:]):
            raise ValueError(
                f"RGB size {tuple(rgb.shape[-2:])} must match coarse mask size {tuple(coarse_logits.shape[-2:])}"
            )

        rgb_features = self.rgb_encoder(rgb.to(dtype=coarse_logits.dtype, device=coarse_logits.device))
        refined_chunks = []
        for start in range(0, coarse_logits.shape[0], self.query_chunk_size):
            chunk = coarse_logits[start : start + self.query_chunk_size]
            features = rgb_features.expand(chunk.shape[0], -1, -1, -1)
            residual_input = torch.cat([features, chunk[:, None]], dim=1)
            residual = self.refine(residual_input)[:, 0]
            refined_chunks.append(chunk + self.residual_scale * residual)
        return torch.cat(refined_chunks, dim=0)
