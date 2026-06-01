from __future__ import annotations

import numpy as np
import torch

from context_seg import (
    InstanceQueryPredictor,
    MaskDecoder,
    MaskTargets,
    PseudoMaskProvider,
    QueryInstanceHead,
    SetCriterion,
)


def test_predictor_decoder_shapes() -> None:
    batch, tokens, context_dim = 2, 16, 32
    patch_tokens = torch.randn(batch, tokens, context_dim)
    predictor = InstanceQueryPredictor(context_dim=context_dim, hidden_dim=16, num_queries=4, num_heads=4, num_layers=1)
    decoder = MaskDecoder(query_dim=16, patch_dim=context_dim, hidden_dim=16)

    queries = predictor(patch_tokens)
    masks = decoder(queries, patch_tokens, patch_grid=(4, 4), image_size=(32, 32))

    assert queries.shape == (batch, 4, 16)
    assert masks.shape == (batch, 4, 32, 32)


def test_query_instance_head_shapes() -> None:
    batch, tokens, context_dim = 2, 16, 32
    patch_tokens = torch.randn(batch, tokens, context_dim)
    head = QueryInstanceHead(
        context_dim=context_dim,
        hidden_dim=16,
        mask_hidden_dim=16,
        num_queries=4,
        num_heads=4,
        num_layers=1,
    )

    output = head(patch_tokens, patch_grid=(4, 4), image_size=(32, 32))

    assert output.query_embeddings.shape == (batch, 4, 16)
    assert output.objectness_logits.shape == (batch, 4)
    assert output.mask_logits.shape == (batch, 4, 32, 32)


def test_criterion_aligned_masks() -> None:
    pred = torch.randn(2, 4, 16, 16)
    target = MaskTargets(masks=torch.randint(0, 2, pred.shape).float())
    losses = SetCriterion()(pred, target)
    assert "loss" in losses
    assert losses["loss"].ndim == 0


def test_pseudo_mask_provider_npz(tmp_path) -> None:
    np.savez(
        tmp_path / "frame_000.npz",
        masks=np.ones((2, 8, 8), dtype=np.uint8),
        boxes=np.zeros((2, 4), dtype=np.float32),
        scores=np.ones((2,), dtype=np.float32),
        areas=np.full((2,), 64, dtype=np.float32),
    )
    masks = PseudoMaskProvider(tmp_path).get("frame_000")
    assert masks.masks.shape == (2, 8, 8)
    assert masks.boxes is not None and masks.boxes.shape == (2, 4)
