from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .types import ContextTokens


class LingbotTokenExtractor(nn.Module):
    """Extract context tokens from a LingBot-Map model without running heads.

    The wrapper keeps LingBot-Map coupling in one place. It expects a model with
    `_aggregate_features(...)`, as implemented by `lingbot_map.models.GCTStream`.
    """

    def __init__(
        self,
        model: nn.Module,
        token_layer: int = -1, # 默认用最后一层 token，因为 aggregator 返回的是一个多层 token 列表，后续可以改成 concat 多层或加个小 transformer 融合。
        freeze_backbone: bool = True,
        use_no_grad: bool = True,
    ) -> None:
        super().__init__()
        self.model = model
        self.token_layer = token_layer
        self.freeze_backbone = freeze_backbone
        self.use_no_grad = use_no_grad

        if freeze_backbone:
            # 第一版默认只训练实例分割头，不更新 LingBot-Map 主干。
            # 这样显存和训练不稳定性都更可控，也方便先验证 token 是否有用。
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad_(False)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: torch.device | str = "cuda",
        token_layer: int = -1,
        freeze_backbone: bool = True,
        use_no_grad: bool = True,
        **model_kwargs,
    ) -> "LingbotTokenExtractor":
        """Build a LingBot-Map streaming model and load a checkpoint."""
        from lingbot_map.models.gct_stream import GCTStream

        # 这里通过包导入 LingBot-Map，要求先 `pip install -e ../lingbot-map`。
        # 并列项目不要写硬编码源码路径，否则换机器或换工作目录会很脆。
        model = GCTStream(**model_kwargs)
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("model", ckpt)
        model.load_state_dict(state_dict, strict=False)
        model = model.to(device).eval()
        return cls(
            model=model,
            token_layer=token_layer,
            freeze_backbone=freeze_backbone,
            use_no_grad=use_no_grad,
        )

    def forward(
        self,
        images: torch.Tensor,
        num_frame_for_scale: Optional[int] = None,
        num_frame_per_block: int = 1,
    ) -> ContextTokens:
        """Return LingBot context tokens.

        Args:
            images: [S, 3, H, W] or [B, S, 3, H, W] tensor in [0, 1].
            num_frame_for_scale: Passed through to LingBot aggregator.
            num_frame_per_block: Passed through to LingBot aggregator.
        """
        if images.ndim == 4:
            images = images.unsqueeze(0) # 如果输入没有 batch 维，默认加上；后续 decoder 会把 [B,S] 合并成 batch 维处理。
        if images.ndim != 5:
            raise ValueError(f"Expected images [S,3,H,W] or [B,S,3,H,W], got {tuple(images.shape)}")

        if not hasattr(self.model, "_aggregate_features"):
            raise TypeError("LingbotTokenExtractor requires a model with _aggregate_features(...)")

        # LingBot 的 patch token 数量由输入分辨率和 patch_size 决定。
        # 后面的 decoder 会把这些 patch tokens 重新排成 patch_h x patch_w 的 mask 网格。
        _, _, _, height, width = images.shape
        patch_size = int(getattr(self.model, "patch_size", 14))
        patch_grid = (height // patch_size, width // patch_size)

        def _run_aggregate() -> tuple[list[torch.Tensor], int]:
            # 只跑 aggregator，不跑 camera/depth/point heads。
            # 返回的 aggregated_tokens_list 是多层 token，patch_start_idx 用来切掉特殊 token。
            return self.model._aggregate_features(
                images,
                num_frame_for_scale=num_frame_for_scale,
                num_frame_per_block=num_frame_per_block,
            )

        if self.use_no_grad or self.freeze_backbone:
            with torch.no_grad():
                aggregated_tokens_list, patch_start_idx = _run_aggregate()
        else:
            aggregated_tokens_list, patch_start_idx = _run_aggregate()

        selected = aggregated_tokens_list[self.token_layer] # 就是最后一层 token，形状 [B, S, N_all, C] 或 [S, N_all, C]。
        # 第一版先约定 token_layer=-1，即默认用最后一层 token；后续可以改成 concat 多层或加个小 transformer 融合。
        # patch_start_idx 之前通常是 camera/register/scale 等特殊 token。
        patch_tokens = selected[:, :, patch_start_idx:] # 形状 [B, S, N_patch, C]，其中 N_patch = patch_h * patch_w。
        expected_patches = patch_grid[0] * patch_grid[1]
        if patch_tokens.shape[2] != expected_patches:
            raise ValueError(
                "Patch token count does not match image grid: "
                f"tokens={patch_tokens.shape[2]}, grid={patch_grid}"
            )

        return ContextTokens(
            tokens=selected,
            patch_tokens=patch_tokens,
            patch_grid=patch_grid,
            image_size=(height, width),
            patch_start_idx=patch_start_idx,
        )
