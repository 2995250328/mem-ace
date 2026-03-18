# ACE Sampler

Learnable buffer sampling for ACE training. Replaces uniform random sampling with a confidence-guided strategy trained from reprojection error, optionally augmented with MC Dropout uncertainty estimation.

## Overview

Standard ACE fills its training buffer by randomly sampling pixels from each image. This module trains a lightweight `SamplerNet` to predict per-pixel confidence from encoder features, then uses it to bias sampling toward points the current ACE model localizes well.

**Two-phase workflow:**

- **Phase 1** — Train `SamplerNet` end-to-end using a frozen ACE model's reprojection error (and optionally MC Dropout uncertainty) as supervision signal.
- **Phase 2** — Plug the trained `SamplerNet` into `ace_trainer.py` buffer filling as a drop-in replacement for `torch.multinomial`.

## Architecture

### SamplerNet

Lightweight depthwise-separable CNN confidence predictor:

```
Input:  (B, 512, H/8, W/8)  — ACE encoder features
Block1: DW-Conv(512) → PW-Conv(128) → ReLU
Block2: DW-Conv(128) → PW-Conv(64)  → ReLU
Head:   Conv(64 → 1) → Sigmoid
Output: (B, 1, H/8, W/8)  — confidence map in [0, 1]
```

~80K parameters. Operates at feature resolution (8× downsampled).

### UncertaintyHead (optional, MC Dropout)

Lightweight coordinate predictor with dropout for uncertainty estimation:

```
Input:  (B, 512, H/8, W/8)  — frozen encoder features
        Conv1x1(512 → 256) → Dropout2d(p) → ReLU → Conv1x1(256 → 3)
Output: (B, 3, H/8, W/8)  — scene coordinate prediction
```

Run T times in `train()` mode → variance of reprojected 2D positions = per-pixel uncertainty.

**Confidence target with MC Dropout:**
```
conf = exp(-(α × repro_error + β × mc_variance))
```

Low reprojection error AND low prediction variance → high confidence → preferred for sampling.

## Phase 1: Train SamplerNet

### Basic (reprojection error only)

```bash
python train_ace_sampler.py \
    datasets/7scenes_chess \
    output/chess_baseline.pt \
    output/chess_sampler.pt \
    --device cuda:0
```

### With MC Dropout uncertainty

```bash
python train_ace_sampler.py \
    datasets/7scenes_chess \
    output/chess_baseline.pt \
    output/chess_sampler_mc.pt \
    --device cuda:0 \
    --use_mc_dropout True \
    --mc_samples 10 \
    --mc_dropout_p 0.1 \
    --sampler_beta 0.01
```

## Phase 2: Train ACE with Guided Sampling

```bash
python train_ace.py \
    datasets/7scenes_chess \
    output/chess_guided.pt \
    --device cuda:0 \
    --sampler_path output/chess_sampler_mc.pt \
    --sampler_ratio 0.7
```

Without `--sampler_path`, `train_ace.py` behaves exactly as before.

## Full Parameter Reference

### Phase 1 — `train_ace_sampler.py`

| Argument | Default | Description |
|---|---|---|
| `scene` | — | Scene root directory (must contain `train/`) |
| `ace_head` | — | Pre-trained ACE head `.pt` (frozen during Phase 1) |
| `sampler_output` | — | Output path for trained SamplerNet `.pt` |
| `--encoder_path` | `ace_encoder_pretrained.pt` | Pre-trained FCN encoder weights |
| `--num_head_blocks` | 1 | ACE head depth (must match the loaded head) |
| `--use_homogeneous` | `False` | Whether the loaded head uses homogeneous coords |
| `--image_resolution` | 480 | Input image height in pixels |
| `--use_aug` | `True` | Enable data augmentation (rotation + scale) |
| `--aug_rotation` | 15 | Max in-plane rotation angle (degrees) |
| `--aug_scale` | 1.5 | Max scale factor (range: `[1/aug_scale, aug_scale]`) |
| `--sampler_epochs` | 5 | Number of training epochs for SamplerNet |
| `--sampler_lr` | 1e-3 | Learning rate for SamplerNet (AdamW) |
| `--sampler_alpha` | 0.1 | Reprojection error weight: `exp(-α × error_px)`. Higher α = sharper target (penalizes large errors more) |
| `--use_half` | `True` | Use FP16 mixed precision |
| `--device` | `cuda` | GPU device (e.g. `cuda:0`, `cuda:1`) |
| `--use_mc_dropout` | `False` | Enable MC Dropout uncertainty estimation |
| `--mc_samples` | 10 | Number of stochastic forward passes per image (T) |
| `--mc_dropout_p` | 0.1 | Dropout probability for UncertaintyHead |
| `--sampler_beta` | 0.01 | Uncertainty weight: `exp(-β × mc_variance_px²)`. Higher β = more aggressive uncertainty penalty |
| `--uncertainty_head_path` | `None` | Path to pre-trained UncertaintyHead `.pt`. If `None` and `--use_mc_dropout True`, the head is trained from scratch via 2-epoch distillation before Phase 1 begins |

### Phase 2 — `train_ace.py` (additional arguments)

| Argument | Default | Description |
|---|---|---|
| `--sampler_path` | `None` | Trained SamplerNet `.pt`. If `None`, original random sampling is used (no behavior change) |
| `--sampler_ratio` | 0.7 | Fraction of `samples_per_image` drawn from top-k confidence (Part A). Remaining `1 - ratio` are drawn randomly (Part B) |
| `--use_neighbors` | `False` | Expand each top-k selection to its 4-connected neighbors at feature resolution |

## Sampling Strategy

For each image, `fill_buffer_with_sampler` draws `samples_per_image` points:

- **Part A** (`ratio × spi` points) — top-k pixels by SamplerNet confidence, masked by `image_mask`
- **Part B** (`(1-ratio) × spi` points) — uniform random from `image_mask` (same as original ACE)

Part B ensures coverage of regions the current model hasn't learned yet (cold-start robustness).

## MC Dropout Workflow

When `--use_mc_dropout True`:

1. **UncertaintyHead pre-training** (2 epochs, before SamplerNet training): distills the frozen ACE Head's predictions into UncertaintyHead via MSE loss. This ensures MC variance is meaningful rather than random.
2. **Per-image uncertainty**: for each training image, UncertaintyHead runs T stochastic forward passes → T sets of scene coordinates → T reprojected 2D positions → variance across T = uncertainty map.
3. **Joint target**: `conf = exp(-(α × repro_error + β × variance))`. Both terms are in pixel units (error in px, variance in px²), so β should be much smaller than α (default: α=0.1, β=0.01).

## Module Structure

```
ace_sampler/
├── __init__.py          # Exports SamplerNet, SamplerTrainer, fill_buffer_with_sampler
├── model.py             # SamplerNet definition + save/load
├── uncertainty.py       # UncertaintyHead with MC Dropout + save/load
├── options.py           # add_sampler_train_args, add_sampler_buffer_args
├── trainer.py           # Phase 1: SamplerTrainer (with optional MC Dropout)
└── buffer_sampler.py    # Phase 2: fill_buffer_with_sampler
train_ace_sampler.py     # Phase 1 entry script (project root)
```

## Notes

- The ACE head passed to Phase 1 should be reasonably trained. A head trained for even a few epochs gives a meaningful reprojection error signal.
- `SamplerNet` is frozen during Phase 2 — it only influences which pixels are sampled, not the ACE loss.
- Buffer schema produced by `fill_buffer_with_sampler` is identical to `ace_trainer.py`: `features / target_px / gt_poses_inv / intrinsics / intrinsics_inv`.
- `ace_network.py` is not modified. The UncertaintyHead is a separate module that runs alongside the frozen ACE regressor.
