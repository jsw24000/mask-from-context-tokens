from __future__ import annotations

import torch
import torch.nn as nn


class InstanceQueryPredictor(nn.Module):
    """Cross-attend learnable object queries to context tokens."""

    def __init__(
        self,
        context_dim: int,
        hidden_dim: int = 256,
        num_queries: int = 100,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        # object queries 是可学习的“槽位”。训练后理想情况下，
        # 不同 query 会各自聚合到不同实例或背景/空目标的信息。
        self.query_embed = nn.Parameter(torch.randn(num_queries, hidden_dim) * 0.02)
        # LingBot token 维度通常较大，先投影到分割头内部 hidden_dim。
        self.context_proj = nn.Linear(context_dim, hidden_dim) if context_dim != hidden_dim else nn.Identity()
        self.layers = nn.ModuleList(
            [_QueryCrossAttentionBlock(hidden_dim, num_heads, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, context_tokens: torch.Tensor) -> torch.Tensor:
        """Predict instance query embeddings.

        Args:
            context_tokens: [B, N, C] or [B, S, N, C].

        Returns:
            [B, Q, hidden_dim] for 3D input or [B, S, Q, hidden_dim] for 4D input.
        """
        original_ndim = context_tokens.ndim
        if original_ndim == 4:
            # 把 [B, S] 合并成 batch 维，表示每个视角独立做 2D 实例分割。
            # 如果后续要做跨视角一致性，可以在这里改成保留 S 维的时序建模。
            batch, frames, tokens, channels = context_tokens.shape
            context_tokens = context_tokens.reshape(batch * frames, tokens, channels)
        elif original_ndim == 3:
            batch = context_tokens.shape[0]
            frames = None
        else:
            raise ValueError(f"Expected [B,N,C] or [B,S,N,C], got {tuple(context_tokens.shape)}")

        memory = self.context_proj(context_tokens)
        # 同一组 query 参数会复制到 batch 中的每个视角，再通过 cross-attention
        # 从当前视角的 context tokens 中读取实例信息。
        queries = self.query_embed.unsqueeze(0).expand(memory.shape[0], -1, -1)
        for layer in self.layers:
            queries = layer(queries, memory)
        queries = self.norm(queries)

        if original_ndim == 4:
            return queries.reshape(batch, frames, self.num_queries, self.hidden_dim)
        return queries


class _QueryCrossAttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )
        self.norm_q1 = nn.LayerNorm(hidden_dim)
        self.norm_q2 = nn.LayerNorm(hidden_dim)
        self.norm_q3 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        # 先让 query 之间交换信息，再让 query attend 到 LingBot context tokens，
        # 最后用 FFN 做非线性更新；这是 DETR/Mask2Former 类方法常见的结构骨架。
        q = self.norm_q1(queries)
        queries = queries + self.dropout(self.self_attn(q, q, q, need_weights=False)[0])
        queries = queries + self.dropout(
            self.cross_attn(self.norm_q2(queries), memory, memory, need_weights=False)[0]
        )
        queries = queries + self.dropout(self.ffn(self.norm_q3(queries)))
        return queries


ObjectQueryPredictor = InstanceQueryPredictor 
