from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskDecoder(nn.Module):
    """Decode query embeddings into dense mask logits via query-patch similarity."""

    def __init__(
        self,
        query_dim: int,
        patch_dim: int,
        hidden_dim: int = 256,
        upsample_mode: str = "bilinear",
    ) -> None:
        super().__init__()
        # 中文导读：decoder 第一版保持很轻量：把 query 和 patch token 投到同一空间，
        # 用点积得到每个 query 对每个 patch 的 mask logit。
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.patch_proj = nn.Linear(patch_dim, hidden_dim)
        self.scale = hidden_dim ** -0.5
        self.upsample_mode = upsample_mode

    def forward(
        self,
        query_embeddings: torch.Tensor,
        patch_tokens: torch.Tensor,
        patch_grid: Tuple[int, int],
        image_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """Return mask logits shaped [B, Q, H, W] or [B, S, Q, H, W]."""
        original_ndim = query_embeddings.ndim
        if original_ndim == 4:
            # 中文导读：和 predictor 一样，先把每帧视角当作独立样本处理。
            batch, frames, queries, query_dim = query_embeddings.shape
            query_embeddings = query_embeddings.reshape(batch * frames, queries, query_dim)
            patch_tokens = patch_tokens.reshape(batch * frames, patch_tokens.shape[-2], patch_tokens.shape[-1])
        elif original_ndim == 3:
            batch = query_embeddings.shape[0]
            frames = None
        else:
            raise ValueError(
                f"Expected query embeddings [B,Q,C] or [B,S,Q,C], got {tuple(query_embeddings.shape)}"
            )

        patch_h, patch_w = patch_grid
        expected_patches = patch_h * patch_w
        if patch_tokens.shape[-2] != expected_patches:
            raise ValueError(
                f"Patch tokens ({patch_tokens.shape[-2]}) do not match patch_grid {patch_grid}"
            )

        q = self.query_proj(query_embeddings)
        p = self.patch_proj(patch_tokens)
        # 中文导读：logits[b, q, n] 表示第 q 个实例 query 与第 n 个图像 patch 的相似度。
        logits = torch.einsum("bqc,bnc->bqn", q, p) * self.scale
        logits = logits.reshape(logits.shape[0], logits.shape[1], patch_h, patch_w)

        if image_size is not None and image_size != patch_grid:
            # 中文导读：patch 级 mask 上采样回图像分辨率，后续再与 SAM 伪标签计算损失。
            logits = F.interpolate(logits, size=image_size, mode=self.upsample_mode, align_corners=False)

        if original_ndim == 4:
            return logits.reshape(batch, frames, logits.shape[1], logits.shape[2], logits.shape[3])
        return logits
