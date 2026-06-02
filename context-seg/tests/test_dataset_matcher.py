from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from context_seg import HungarianMatcher, MaskTargets, PseudoMaskFilter, SetCriterion, VideoFrameDataset


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
    pred_logits = torch.randn(5, 2)
    match = HungarianMatcher(cost_size=16)(pred, target, pred_logits)
    assert match.pred_indices.numel() == 3
    assert match.target_indices.numel() == 3


def test_set_criterion_variable_masks() -> None:
    pred = torch.randn(5, 16, 16)
    pred_logits = torch.randn(5, 2)
    target = torch.randint(0, 2, (3, 16, 16)).float()
    losses = SetCriterion(matcher=HungarianMatcher(cost_size=16), num_points=64)(pred, MaskTargets(masks=target), pred_logits)
    assert losses["loss"].ndim == 0
    assert losses["loss_cls"].ndim == 0
    assert losses["num_matches"].item() == 3


def test_set_criterion_boundary_loss() -> None:
    pred = torch.randn(5, 16, 16)
    pred_logits = torch.randn(5, 2)
    target = torch.zeros(3, 16, 16)
    target[:, 4:12, 4:12] = 1
    losses = SetCriterion(
        matcher=HungarianMatcher(cost_size=16),
        num_points=64,
        boundary_weight=1.0,
        boundary_size=16,
    )(pred, MaskTargets(masks=target), pred_logits)

    assert losses["loss_boundary"].ndim == 0
    assert losses["loss_boundary"].item() >= 0


def test_set_criterion_empty_targets_trains_no_object() -> None:
    pred = torch.randn(5, 16, 16)
    pred_logits = torch.randn(5, 2)
    target = torch.zeros(0, 16, 16)
    losses = SetCriterion(matcher=HungarianMatcher(cost_size=16), num_points=64)(pred, MaskTargets(masks=target), pred_logits)
    assert losses["loss"].ndim == 0
    assert losses["loss_cls"].item() > 0
    assert losses["num_matches"].item() == 0


def test_pseudo_mask_filter_removes_large_and_overlapping_masks() -> None:
    masks = torch.zeros(4, 10, 10)
    masks[0, :9, :9] = 1
    masks[1, 1:4, 1:4] = 1
    masks[2, 1:4, 1:4] = 1
    masks[3, 6:9, 6:9] = 1
    scores = torch.tensor([0.9, 0.8, 0.7, 0.6])

    kept = PseudoMaskFilter(min_area_ratio=0.01, max_area_ratio=0.6, nms_iou=0.8, max_masks=3)(masks, scores)

    assert kept.shape[0] == 2
