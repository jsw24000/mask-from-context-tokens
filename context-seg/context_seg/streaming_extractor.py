from __future__ import annotations

from collections.abc import Iterator
from typing import Optional

import torch
import torch.nn as nn

from .types import ContextTokens


class StreamingLingbotTokenExtractor(nn.Module):
    """LingBot-style online extractor for current-frame context tokens.

    This wrapper mirrors the token-producing part of LingBot-Map streaming:
    initialize a sequence with scale frames, then process one new frame per
    `step(...)` while reusing the model's KV cache. It intentionally does not
    run camera/depth/point heads because the mask task consumes aggregator
    tokens only.
    """

    def __init__(
        self,
        model: nn.Module,
        token_layer: int = -1,
        freeze_backbone: bool = True,
        use_no_grad: bool = True,
        require_camera_only_cache: bool = True,
    ) -> None:
        super().__init__()
        self.model = model
        self.token_layer = token_layer
        self.freeze_backbone = freeze_backbone
        self.use_no_grad = use_no_grad
        self.require_camera_only_cache = require_camera_only_cache
        self.num_scale_frames: Optional[int] = None
        self.frames_seen = 0

        if require_camera_only_cache and getattr(model, "kv_cache_camera_only", None) is not True:
            raise ValueError(
                "StreamingLingbotTokenExtractor expects kv_cache_camera_only=True. "
                "Build it with from_checkpoint(...) or pass a model configured for camera-only cache."
            )

        if freeze_backbone:
            # streaming 版本仍默认冻结 LingBot，只训练后续 mask 模块。
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
        kv_cache_camera_only: bool = True,
        kv_cache_sliding_window: int = 64,
        kv_cache_scale_frames: int = 8,
        kv_cache_cross_frame_special: bool = True,
        kv_cache_include_scale_frames: bool = True,
        **model_kwargs,
    ) -> "StreamingLingbotTokenExtractor":
        """Build a streaming LingBot model with explicit KV cache policy."""
        from lingbot_map.models.gct_stream import GCTStream

        model_kwargs.setdefault("kv_cache_camera_only", kv_cache_camera_only)
        model_kwargs.setdefault("kv_cache_sliding_window", kv_cache_sliding_window)
        model_kwargs.setdefault("kv_cache_scale_frames", kv_cache_scale_frames)
        model_kwargs.setdefault("kv_cache_cross_frame_special", kv_cache_cross_frame_special)
        model_kwargs.setdefault("kv_cache_include_scale_frames", kv_cache_include_scale_frames)

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
            require_camera_only_cache=True,
        )

    def reset(self) -> None:
        """Clear model KV cache and reset streaming counters."""
        if hasattr(self.model, "clean_kv_cache"):
            self.model.clean_kv_cache()
        self.num_scale_frames = None
        self.frames_seen = 0

    def start_sequence(self, scale_images: torch.Tensor) -> ContextTokens:
        """Initialize streaming cache with scale frames and return their tokens."""
        scale_images = _normalize_sequence(scale_images)
        scale_frames = int(scale_images.shape[1])
        if scale_frames < 1:
            raise ValueError("start_sequence requires at least one scale frame")

        self.reset()
        tokens = self._extract_tokens(
            scale_images,
            num_frame_for_scale=scale_frames,
            num_frame_per_block=scale_frames,
        )
        self.num_scale_frames = scale_frames
        self.frames_seen = scale_frames
        return tokens

    def step(self, frame_image: torch.Tensor, is_keyframe: bool = True) -> ContextTokens:
        """Process one new frame and return only that frame's context tokens."""
        if self.num_scale_frames is None:
            raise RuntimeError("Call start_sequence(...) before step(...)")

        frame_image = _normalize_frame(frame_image)
        skip_append = not is_keyframe
        if skip_append:
            self._set_skip_append(True)
        try:
            tokens = self._extract_tokens(
                frame_image,
                num_frame_for_scale=self.num_scale_frames,
                num_frame_per_block=1,
            )
        finally:
            if skip_append:
                self._set_skip_append(False)

        self.frames_seen += 1
        return tokens

    def stream_sequence(
        self,
        images: torch.Tensor,
        num_scale_frames: Optional[int] = None,
        keyframe_interval: int = 1,
    ) -> Iterator[ContextTokens]:
        """Yield scale-frame tokens first, then current-frame tokens per step."""
        images = _normalize_sequence(images)
        total_frames = int(images.shape[1])
        scale_frames = num_scale_frames if num_scale_frames is not None else getattr(
            self.model, "num_frame_for_scale", 1
        )
        scale_frames = max(1, min(int(scale_frames), total_frames))

        yield self.start_sequence(images[:, :scale_frames])

        kf_interval = max(int(keyframe_interval), 1)
        for i in range(scale_frames, total_frames):
            is_keyframe = kf_interval <= 1 or ((i - scale_frames) % kf_interval == 0)
            yield self.step(images[:, i : i + 1], is_keyframe=is_keyframe)

    def _extract_tokens(
        self,
        images: torch.Tensor,
        num_frame_for_scale: int,
        num_frame_per_block: int,
    ) -> ContextTokens:
        if not hasattr(self.model, "_aggregate_features"):
            raise TypeError("StreamingLingbotTokenExtractor requires a model with _aggregate_features(...)")

        _, _, _, height, width = images.shape
        patch_size = int(getattr(self.model, "patch_size", 14))
        patch_grid = (height // patch_size, width // patch_size)

        def _run_aggregate() -> tuple[list[torch.Tensor], int]:
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

        selected = aggregated_tokens_list[self.token_layer]
        patch_tokens = selected[:, :, patch_start_idx:]
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

    def _set_skip_append(self, skip: bool) -> None:
        if hasattr(self.model, "_set_skip_append"):
            self.model._set_skip_append(skip)


def _normalize_sequence(images: torch.Tensor) -> torch.Tensor:
    """Normalize images to [B, S, 3, H, W]."""
    if images.ndim == 4:
        images = images.unsqueeze(0)
    if images.ndim != 5:
        raise ValueError(f"Expected [S,3,H,W] or [B,S,3,H,W], got {tuple(images.shape)}")
    return images


def _normalize_frame(frame: torch.Tensor) -> torch.Tensor:
    """Normalize a single frame to [B, 1, 3, H, W]."""
    if frame.ndim == 3:
        frame = frame.unsqueeze(0).unsqueeze(0)
    elif frame.ndim == 4:
        frame = frame.unsqueeze(1)
    elif frame.ndim == 5 and frame.shape[1] == 1:
        pass
    else:
        raise ValueError(f"Expected [3,H,W], [B,3,H,W], or [B,1,3,H,W], got {tuple(frame.shape)}")
    return frame

