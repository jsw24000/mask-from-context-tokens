# Context Seg

This package is a thin research scaffold for testing whether LingBot-Map context
tokens can support 2D instance segmentation.

Install order:

```bash
cd ../lingbot-map
pip install -e .

cd ../context-seg
pip install -e .
```

The first version intentionally provides interfaces and shape-checked modules,
not a complete training pipeline. SAM-style masks are expected to be generated
offline and read through `PseudoMaskProvider`.

Recommended pseudo-label source: SAM 2.1 Hiera Base+ for a quality/speed
balance, or SAM 2.1 Hiera Large when label quality matters more than cost.
Install the official SAM2 code/checkpoints separately in the environment used
for offline mask generation.

## First Training Experiment

You already have the LingBot checkpoint at:

```text
../paths/lingbot-map.pt
```

For pseudo labels, install the original SAM package and download one checkpoint:

- Fast smoke tests: `sam_vit_b_01ec64.pth`
- Higher-quality labels: `sam_vit_h_4b8939.pth`

Expected local frame dataset:

```text
data/
  scene_001/
    images/
      000000.png
      000001.png
```

Generate offline SAM masks:

```bash
python scripts/generate_sam_masks.py \
  --image-root ../data \
  --output-root ../data_sam_masks \
  --sam-checkpoint /path/to/sam_vit_b_01ec64.pth \
  --model-type vit_b
```

Run a small overfit experiment:

```bash
python scripts/train.py \
  --config configs/default.yaml \
  --data-root ../data \
  --mask-root ../data_sam_masks \
  --lingbot-checkpoint ../paths/lingbot-map.pt \
  --overfit
```

From the repository root, use:

```bash
python context-seg/scripts/train.py \
  --config context-seg/configs/default.yaml \
  --data-root data \
  --mask-root data_sam_masks \
  --lingbot-checkpoint paths/lingbot-map.pt \
  --overfit
```
