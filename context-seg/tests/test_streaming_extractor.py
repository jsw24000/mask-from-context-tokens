from __future__ import annotations

import torch

from context_seg import StreamingLingbotTokenExtractor


class FakeStreamingLingbot(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_size = 14
        self.context_dim = 32
        self.num_frame_for_scale = 2
        self.kv_cache_camera_only = True
        self.clean_calls = 0
        self.skip_events = []

    def clean_kv_cache(self) -> None:
        self.clean_calls += 1

    def _set_skip_append(self, skip: bool) -> None:
        self.skip_events.append(skip)

    def _aggregate_features(
        self,
        images: torch.Tensor,
        num_frame_for_scale: int | None = None,
        num_frame_per_block: int = 1,
        **kwargs,
    ) -> tuple[list[torch.Tensor], int]:
        batch, frames, _, height, width = images.shape
        patch_h, patch_w = height // self.patch_size, width // self.patch_size
        patch_start_idx = 3
        tokens = patch_start_idx + patch_h * patch_w
        return [torch.randn(batch, frames, tokens, self.context_dim)], patch_start_idx


def test_streaming_shapes() -> None:
    model = FakeStreamingLingbot()
    extractor = StreamingLingbotTokenExtractor(model)

    scale = extractor.start_sequence(torch.randn(2, 3, 56, 56))
    assert scale.patch_tokens.shape == (1, 2, 16, 32)

    step = extractor.step(torch.randn(3, 56, 56))
    assert step.patch_tokens.shape == (1, 1, 16, 32)
    assert model.clean_calls == 1


def test_stream_sequence_and_keyframes() -> None:
    model = FakeStreamingLingbot()
    extractor = StreamingLingbotTokenExtractor(model)
    outputs = list(extractor.stream_sequence(torch.randn(5, 3, 56, 56), num_scale_frames=2, keyframe_interval=2))

    assert len(outputs) == 4
    assert outputs[0].patch_tokens.shape == (1, 2, 16, 32)
    assert outputs[-1].patch_tokens.shape == (1, 1, 16, 32)
    assert model.skip_events == [True, False]

