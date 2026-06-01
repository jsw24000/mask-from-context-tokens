from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .predictor import InstanceQueryPredictor
from .types import QuerySegOutput


class QueryInstanceHead(nn.Module):
    """Class-agnostic instance segmentation head with objectness-gated queries."""

    def __init__(
        self,
        context_dim: int,
        hidden_dim: int = 256,
        num_queries: int = 100,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.0,
        mask_hidden_dim: Optional[int] = None,
        upsample_mode: str = "bilinear",
    ) -> None:
        super().__init__()
        mask_hidden_dim = int(mask_hidden_dim or hidden_dim)
        self.predictor = InstanceQueryPredictor(
            context_dim=context_dim,
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.objectness_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.query_mask_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, mask_hidden_dim),
            nn.GELU(),
            nn.Linear(mask_hidden_dim, mask_hidden_dim),
        )
        self.patch_mask_proj = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, mask_hidden_dim),
            nn.GELU(),
            nn.Linear(mask_hidden_dim, mask_hidden_dim),
        )
        self.scale = mask_hidden_dim**-0.5
        self.upsample_mode = upsample_mode

    def forward(
        self,
        patch_tokens: torch.Tensor,
        patch_grid: Tuple[int, int],
        image_size: Optional[Tuple[int, int]] = None,
    ) -> QuerySegOutput:
        """Return mask logits and objectness logits for each query."""
        query_embeddings = self.predictor(patch_tokens)
        original_ndim = query_embeddings.ndim
        if original_ndim == 4:
            batch, frames, queries, channels = query_embeddings.shape
            flat_queries = query_embeddings.reshape(batch * frames, queries, channels)
            flat_patches = patch_tokens.reshape(batch * frames, patch_tokens.shape[-2], patch_tokens.shape[-1])
        elif original_ndim == 3:
            batch = query_embeddings.shape[0]
            frames = None
            flat_queries = query_embeddings
            flat_patches = patch_tokens
        else:
            raise ValueError(f"Expected query embeddings [B,Q,C] or [B,S,Q,C], got {tuple(query_embeddings.shape)}")

        patch_h, patch_w = patch_grid
        expected_patches = patch_h * patch_w
        if flat_patches.shape[-2] != expected_patches:
            raise ValueError(f"Patch tokens ({flat_patches.shape[-2]}) do not match patch_grid {patch_grid}")

        q = self.query_mask_proj(flat_queries)
        p = self.patch_mask_proj(flat_patches)
        mask_logits = torch.einsum("bqc,bnc->bqn", q, p) * self.scale
        mask_logits = mask_logits.reshape(mask_logits.shape[0], mask_logits.shape[1], patch_h, patch_w)
        if image_size is not None and image_size != patch_grid:
            mask_logits = F.interpolate(mask_logits, size=image_size, mode=self.upsample_mode, align_corners=False)

        objectness_logits = self.objectness_head(flat_queries).squeeze(-1)
        if original_ndim == 4:
            mask_logits = mask_logits.reshape(batch, frames, mask_logits.shape[1], mask_logits.shape[2], mask_logits.shape[3])
            objectness_logits = objectness_logits.reshape(batch, frames, objectness_logits.shape[-1])
        return QuerySegOutput(
            mask_logits=mask_logits,
            objectness_logits=objectness_logits,
            query_embeddings=query_embeddings,
        )

    @torch.no_grad()
    def active_queries(
        self,
        output: QuerySegOutput,
        objectness_threshold: float = 0.5,
        max_active_queries: Optional[int] = None,
    ) -> list[torch.Tensor]:
        """Return active query indices for each flattened sample."""
        logits = output.objectness_logits
        if logits.ndim == 3:
            logits = logits.reshape(-1, logits.shape[-1])
        elif logits.ndim != 2:
            raise ValueError(f"Expected objectness logits [B,Q] or [B,S,Q], got {tuple(logits.shape)}")
        scores = logits.sigmoid()
        active = []
        for sample_scores in scores:
            keep = torch.nonzero(sample_scores >= objectness_threshold, as_tuple=False).flatten()
            if max_active_queries is not None and keep.numel() > int(max_active_queries):
                top = torch.argsort(sample_scores[keep], descending=True)[: int(max_active_queries)]
                keep = keep[top]
            active.append(keep)
        return active
