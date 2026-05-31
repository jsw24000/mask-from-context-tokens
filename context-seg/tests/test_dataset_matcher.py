from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from context_seg import HungarianMatcher, MaskTargets, SetCriterion, VideoFrameDataset


def test_video_frame_dataset_scene_images(tmp_path) -> None:
    image_dir = tmp_path / "data" / "scene_a" / "images"
    mask_dir = tmp_path / "masks" / "scene_a"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    for i in range(3):
        Image.fromarray(np.full((8, 8, 3), i, dtype=np.uint8)).save(image_dir / f"{i:06d}.png")
        np.savez(mask_dir / f"{i:06d}.npz", masks=np.ones((2, 8, 8), dtype=np.uint8))

    dataset = VideoFrameDataset(tmp_path / "data", tmp_path / "masks", clip_length=2, image_size=16)
    sample = dataset[0]
    assert sample.images.shape == (2, 3, 16, 16)
    assert sample.target_masks[0].shape == (2, 16, 16)
    assert sample.image_ids[0] == "scene_a/000000"


def test_hungarian_matcher_variable_counts() -> None:
    pred = torch.randn(5, 16, 16)
    target = torch.randint(0, 2, (3, 16, 16)).float()
    match = HungarianMatcher(cost_size=16)(pred, target)
    assert match.pred_indices.numel() == 3
    assert match.target_indices.numel() == 3


def test_set_criterion_variable_masks() -> None:
    pred = torch.randn(5, 16, 16)
    target = torch.randint(0, 2, (3, 16, 16)).float()
    losses = SetCriterion(matcher=HungarianMatcher(cost_size=16))(pred, MaskTargets(masks=target))
    assert losses["loss"].ndim == 0

