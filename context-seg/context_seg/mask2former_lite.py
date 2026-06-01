from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .types import QuerySegOutput


class Mask2FormerLiteHead(nn.Module):
    """A small Mask2Former-style decoder for LingBot patch tokens."""

    def __init__(
        self,
        context_dim: int,
        hidden_dim: int = 256,
        pixel_dim: int = 256,
        num_queries: int = 100,
        num_heads: int = 8,
        decoder_layers: int = 6,
        ffn_dim: int = 1024,
        dropout: float = 0.0,
        upsample_mode: str = "bilinear",
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.num_queries = int(num_queries)
        self.hidden_dim = int(hidden_dim)
        self.pixel_dim = int(pixel_dim)
        self.upsample_mode = upsample_mode

        self.pixel_proj = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, pixel_dim),
            nn.GELU(),
            nn.Linear(pixel_dim, pixel_dim),
        )
        self.memory_proj = nn.Linear(pixel_dim, hidden_dim)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        self.layers = nn.ModuleList(
            [_Mask2FormerLiteDecoderLayer(hidden_dim, num_heads, ffn_dim, dropout) for _ in range(decoder_layers)]
        )
        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.class_embed = nn.Linear(hidden_dim, 2)
        self.mask_embed = MLP(hidden_dim, hidden_dim, pixel_dim, num_layers=3)

    def forward(
        self,
        patch_tokens: torch.Tensor,
        patch_grid: Tuple[int, int],
        image_size: Optional[Tuple[int, int]] = None,
    ) -> QuerySegOutput:
        original_ndim = patch_tokens.ndim
        if original_ndim == 4:
            batch, frames, tokens, channels = patch_tokens.shape
            flat_tokens = patch_tokens.reshape(batch * frames, tokens, channels)
        elif original_ndim == 3:
            batch = patch_tokens.shape[0]
            frames = None
            flat_tokens = patch_tokens
        else:
            raise ValueError(f"Expected patch_tokens [B,N,C] or [B,S,N,C], got {tuple(patch_tokens.shape)}")

        patch_h, patch_w = patch_grid
        expected_patches = patch_h * patch_w
        if flat_tokens.shape[1] != expected_patches:
            raise ValueError(f"Patch tokens ({flat_tokens.shape[1]}) do not match patch_grid {patch_grid}")

        pixel_tokens = self.pixel_proj(flat_tokens)
        memory = self.memory_proj(pixel_tokens)
        memory = memory + sine_position_embedding(patch_h, patch_w, self.hidden_dim, flat_tokens.device, flat_tokens.dtype)

        queries = self.query_feat.weight.unsqueeze(0).expand(flat_tokens.shape[0], -1, -1)
        query_pos = self.query_embed.weight.unsqueeze(0).expand_as(queries)

        aux_outputs = []
        pred_logits, pred_masks = self._predict(queries, pixel_tokens, patch_grid, image_size)
        for layer in self.layers:
            queries = layer(queries, memory, query_pos)
            pred_logits, pred_masks = self._predict(queries, pixel_tokens, patch_grid, image_size)
            aux_outputs.append({"pred_logits": pred_logits, "pred_masks": pred_masks})

        final_queries = self.decoder_norm(queries)
        pred_logits, pred_masks = self._predict(final_queries, pixel_tokens, patch_grid, image_size)
        aux_outputs = aux_outputs[:-1]

        objectness_logits = pred_logits[..., 1] - pred_logits[..., 0]
        if original_ndim == 4:
            pred_logits = pred_logits.reshape(batch, frames, pred_logits.shape[-2], pred_logits.shape[-1])
            pred_masks = pred_masks.reshape(batch, frames, pred_masks.shape[-3], pred_masks.shape[-2], pred_masks.shape[-1])
            objectness_logits = objectness_logits.reshape(batch, frames, objectness_logits.shape[-1])
            final_queries = final_queries.reshape(batch, frames, final_queries.shape[-2], final_queries.shape[-1])
            shaped_aux = []
            for aux in aux_outputs:
                shaped_aux.append(
                    {
                        "pred_logits": aux["pred_logits"].reshape(batch, frames, aux["pred_logits"].shape[-2], 2),
                        "pred_masks": aux["pred_masks"].reshape(
                            batch,
                            frames,
                            aux["pred_masks"].shape[-3],
                            aux["pred_masks"].shape[-2],
                            aux["pred_masks"].shape[-1],
                        ),
                    }
                )
            aux_outputs = shaped_aux

        return QuerySegOutput(
            mask_logits=pred_masks,
            objectness_logits=objectness_logits,
            query_embeddings=final_queries,
            pred_logits=pred_logits,
            aux_outputs=tuple(aux_outputs),
        )

    def _predict(
        self,
        queries: torch.Tensor,
        pixel_tokens: torch.Tensor,
        patch_grid: Tuple[int, int],
        image_size: Optional[Tuple[int, int]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        queries = self.decoder_norm(queries)
        pred_logits = self.class_embed(queries)
        mask_embed = self.mask_embed(queries)
        pred_masks = torch.einsum("bqc,bnc->bqn", mask_embed, pixel_tokens)
        pred_masks = pred_masks.reshape(pred_masks.shape[0], pred_masks.shape[1], patch_grid[0], patch_grid[1])
        if image_size is not None and image_size != patch_grid:
            pred_masks = F.interpolate(pred_masks, size=image_size, mode=self.upsample_mode, align_corners=False)
        return pred_logits, pred_masks


class _Mask2FormerLiteDecoderLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries: torch.Tensor, memory: torch.Tensor, query_pos: torch.Tensor) -> torch.Tensor:
        q = self.norm1(queries)
        queries = queries + self.dropout(self.self_attn(q + query_pos, q + query_pos, q, need_weights=False)[0])
        q = self.norm2(queries)
        queries = queries + self.dropout(self.cross_attn(q + query_pos, memory, memory, need_weights=False)[0])
        queries = queries + self.dropout(self.ffn(self.norm3(queries)))
        return queries


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int) -> None:
        super().__init__()
        layers = []
        for layer_idx in range(num_layers):
            in_dim = input_dim if layer_idx == 0 else hidden_dim
            out_dim = output_dim if layer_idx == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if layer_idx < num_layers - 1:
                layers.append(nn.ReLU(inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def sine_position_embedding(
    height: int,
    width: int,
    hidden_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if hidden_dim % 4 != 0:
        raise ValueError("hidden_dim must be divisible by 4 for sine position embedding")
    y, x = torch.meshgrid(
        torch.linspace(0, 1, height, device=device, dtype=dtype),
        torch.linspace(0, 1, width, device=device, dtype=dtype),
        indexing="ij",
    )
    omega = torch.arange(hidden_dim // 4, device=device, dtype=dtype)
    omega = 1.0 / (10000 ** (omega / max(1, hidden_dim // 4 - 1)))
    x = x.flatten()[:, None] * omega[None, :] * (2 * math.pi)
    y = y.flatten()[:, None] * omega[None, :] * (2 * math.pi)
    pos = torch.cat([x.sin(), x.cos(), y.sin(), y.cos()], dim=1)
    return pos.unsqueeze(0)
