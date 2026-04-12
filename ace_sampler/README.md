# ACE Sampler

Learnable buffer sampling for ACE training. Replaces uniform random sampling with a confidence-guided strategy trained from reprojection error, optionally augmented with MC Dropout uncertainty estimation.

## Overview

Standard ACE fills its training buffer by randomly sampling pixels from each image. This module trains a lightweight `SamplerNet` to predict per-pixel confidence from encoder features, then uses it to bias sampling toward points the current ACE model localizes well.

**Two-phase workflow:**

- **Phase 1** — Train `SamplerNet` end-to-end using a frozen ACE model's reprojection error (and optionally MC Dropout uncertainty) as supervision signal.
- **Phase 2** — Plug the trained `SamplerNet` into `ace_trainer.py` buffer filling as a drop-in replacement for `torch.multinomial`.

## Architecture

### SamplerNet (configurable capacity)

Two capacity modes via `--mid_channels` and `--num_branches`:

```
small (mid_channels=128, num_branches=2) — ~107K params, single-scene training
─────────────────────────────────────────────────────────────────────────────
Stage 1: DW-Conv(512,k=3) → GN(32) → PW-Conv(128) → GN(32) → SiLU
Stage 2: 2-branch Lite-ASPP (dilation 1,3), each → 32ch, concat → 64ch
Stage 3: ECA-lite(64)
Stage 4: Conv1×1(64→1) → Sigmoid

large (mid_channels=256, num_branches=3) — ~366K params, multi-scene training
─────────────────────────────────────────────────────────────────────────────
Stage 1: DW-Conv(512,k=3) → GN(32) → PW-Conv(256) → GN(32) → SiLU
Stage 2: 3-branch Lite-ASPP (dilation 1,3,5), each → 64ch, concat → 192ch
         + residual_proj(256→192)
Stage 3: ECA-lite(192)
Stage 4: Conv1×1(192→64) → GN(32) → SiLU → Conv1×1(64→1) → Sigmoid
```

Both use GroupNorm (not BatchNorm) — safe with batch_size=1.

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

## Quick Test with Pre-trained Weights

Skip training entirely and evaluate a downloaded head directly:

```bash
python test_ace.py \
    /mnt/storage/xwh/7Scenes/pgt_7scenes_chess \
    output/ace_models/7Scenes_pgt/pgt_7scenes_chess.pt \
    --device cuda:0
```

Available pre-trained heads in `output/ace_models/7Scenes_pgt/`:

| Scene | File |
|---|---|
| chess | `pgt_7scenes_chess.pt` |
| fire | `pgt_7scenes_fire.pt` |
| heads | `pgt_7scenes_heads.pt` |
| office | `pgt_7scenes_office.pt` |
| pumpkin | `pgt_7scenes_pumpkin.pt` |
| redkitchen | `pgt_7scenes_redkitchen.pt` |
| stairs | `pgt_7scenes_stairs.pt` |

Results are saved to the same directory as the `.pt` file:
- `test_<scene>_.txt` — median rotation/translation error and avg inference time
- `poses_<scene>_.txt` — per-frame pose estimates and errors

## Phase 1: Train SamplerNet

> **Note on `--encoder_path`**: ACE saves only the scene-specific head weights (~4 MB) to
> the output `.pt` file; the encoder is stored separately in `ace_encoder_pretrained.pt`.
> `train_ace_sampler.py` loads them independently and merges them at runtime.
> The default `--encoder_path ace_encoder_pretrained.pt` resolves to the project root, so
> no explicit flag is needed when running from there.

> **Note on output path**: The `sampler_output` argument is a hint for the filename stem.
> The script automatically redirects the actual save location to
> `ace_sampler/04_evaluation/{dataset}/{scene}/{timestamp}_{stem}.pt`.
> The resolved path is printed at startup.

### Basic (reprojection error only)

```bash
python train_ace_sampler.py \
    /mnt/storage/xwh/7Scenes/pgt_7scenes_chess \
    output/ace_models/7Scenes_pgt/pgt_7scenes_chess.pt \
    chess_sampler.pt \
    --device cuda:0
# Saved to: ace_sampler/04_evaluation/7Scenes_pgt/pgt_7scenes_chess/<timestamp>_chess_sampler.pt
```

### With MC Dropout uncertainty

```bash
python train_ace_sampler.py \
    /mnt/storage/xwh/7Scenes/pgt_7scenes_chess \
    output/ace_models/7Scenes_pgt/pgt_7scenes_chess.pt \
    chess_sampler_mc.pt \
    --device cuda:0 \
    --use_mc_dropout True \
    --mc_samples 10 \
    --mc_dropout_p 0.1 \
    --sampler_beta 0.01
# Saved to: ace_sampler/04_evaluation/7Scenes_pgt/pgt_7scenes_chess/<timestamp>_chess_sampler_mc.pt
```

## Phase 2: Train ACE with Guided Sampling

Training automatically runs evaluation on the test split when it finishes.
Pass `--eval_after_train False` to skip.

```bash
python train_ace.py \
    /mnt/storage/xwh/7Scenes/pgt_7scenes_chess \
    output/chess_guided.pt \
    --sampler_path output/chess_sampler_mc.pt \
    --sampler_ratio 0.7
```

Results are saved alongside the output `.pt`:
- `test_<scene>_.txt` — median rotation/translation error and avg inference time
- `poses_<scene>_.txt` — per-frame pose estimates and errors

Without `--sampler_path`, `train_ace.py` behaves exactly as before (random sampling + auto-eval).

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

## Multi-Scene Universal Sampler (Recommended)

Training on a single scene produces a sampler that overfits to that scene's statistics.
Training across diverse scenes teaches the sampler scene-agnostic localizability cues
(textureless regions, repetitive patterns, specular highlights).

### Cambridge: cluster heads vs. single-scene heads

Cambridge Landmarks uses spatial clustering (ACE Poker). Each scene is split into N
spatial clusters, each with its own specialized head. Using cluster heads for sampler
training gives more accurate reprojection error maps than the single-scene head.

The scene list format supports an optional `num_clusters cluster_idx` suffix:

```
# full scene (7Scenes, 12Scenes)
scene_path  head_path

# spatial cluster (Cambridge ensemble)
scene_path  head_path  num_clusters  cluster_idx
```

`configs/sampler_all.txt` uses 4-cluster heads for Cambridge (35 entries total).
`configs/sampler_cambridge_ensemble.txt` is Cambridge-only (20 cluster entries).

### Step 1: Prepare a scene list

Available configs in `configs/`:

| File | Scenes | Entries |
|---|---|---|
| `sampler_7scenes_pgt.txt` | 7Scenes (3 available) | 3 |
| `sampler_cambridge_ensemble.txt` | Cambridge (5 scenes × 4 clusters) | 20 |
| `sampler_all.txt` | 7Scenes + 12Scenes + Cambridge clusters | 35 |

### Step 2: Train universal SamplerNet

> **Note on output path**: Same as single-scene — the script redirects output to
> `ace_sampler/04_evaluation/universal/{timestamp}_{stem}.pt`.

```bash
# Quick start: 3 scenes
python train_ace_sampler_multi.py \
    ace_sampler/configs/sampler_7scenes_pgt.txt \
    universal_sampler.pt \
    --device cuda:0

# Full: 35 entries (7S + 12S + Cambridge clusters)
python train_ace_sampler_multi.py \
    ace_sampler/configs/sampler_all.txt \
    universal_sampler_all.pt \
    --device cuda:0 \
    --sampler_epochs 10
# Saved to: ace_sampler/04_evaluation/universal/<timestamp>_universal_sampler_all.pt
```

### Step 3: Use in Phase 2 (same as single-scene sampler)

```bash
python train_ace.py \
    /mnt/storage/xwh/7Scenes/pgt_7scenes_chess \
    output/chess_guided.pt \
    --sampler_path ace_sampler/04_evaluation/universal/<timestamp>_universal_sampler_all.pt \
    --sampler_ratio 0.7
```

### Multi-scene parameter reference

| Argument | Default | Description |
|---|---|---|
| `scene_list` | — | Text file: `scene head [num_clusters cluster_idx]` per line |
| `sampler_output` | — | Output `.pt` path (rewritten to eval dir with timestamp) |
| `--encoder_path` | `ace_encoder_pretrained.pt` | Shared FCN encoder weights |
| `--sampler_epochs` | 10 | Training epochs |
| `--sampler_lr` | 5e-4 | Learning rate (lower than single-scene due to larger dataset) |
| `--sampler_alpha` | 0.1 | Confidence target sharpness |
| `--mid_channels` | 256 | SamplerNet width: 128=small(~107K), 256=large(~366K) |
| `--num_branches` | 3 | Lite-ASPP branches: 2 or 3 |
| `--device` | `cuda` | GPU device |

### Phase 2 — `train_ace.py` (additional arguments)

| Argument | Default | Description |
|---|---|---|
| `--sampler_path` | `None` | Trained SamplerNet `.pt`. If `None`, original random sampling is used (no behavior change) |
| `--sampler_ratio` | 0.7 | Fraction of `samples_per_image` drawn from top-k confidence (Part A). Remaining `1 - ratio` are drawn randomly (Part B) |
| `--use_neighbors` | `False` | Expand each top-k selection to its 4-connected neighbors at feature resolution |
| `--eval_after_train` | `True` | Run evaluation on the test split immediately after training |

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

## Visualization

Inspect what the trained SamplerNet has learned by overlaying its confidence map and sampled points on test images.

```bash
python ace_sampler/visualize_sampler.py \
    /mnt/storage/xwh/7Scenes/pgt_7scenes_chess \
    output/ace_models/7Scenes_pgt/pgt_7scenes_chess.pt \
    ace_sampler/04_evaluation/universal/<timestamp>_universal_sampler.pt \
    --out_dir ace_sampler/04_evaluation/vis_chess \
    --n_images 20 \
    --device cuda:0
```

Each image produces a `<stem>_side.png` with three panels side by side:

| Panel | Content |
|---|---|
| RGB | Original image |
| Confidence | SamplerNet heatmap (red=high, blue=low) |
| Sampled | Green dots = top-k high-confidence (Part A), red dots = random supplement (Part B) |

**Parameters:**

| Argument | Default | Description |
|---|---|---|
| `scene` | — | Scene root (must contain `test/` or `train/`) |
| `ace_head` | — | Pre-trained ACE head `.pt` |
| `sampler_path` | — | Trained SamplerNet `.pt` |
| `--encoder_path` | `ace_encoder_pretrained.pt` | FCN encoder weights |
| `--out_dir` | `ace_sampler/04_evaluation/vis` | Output directory |
| `--split` | `test` | Which split to visualize (`train` or `test`) |
| `--n_images` | 20 | Number of images to process |
| `--samples_per_image` | 1024 | Total points to draw per image |
| `--sampler_ratio` | 0.7 | Fraction of green (top-k) vs red (random) points |
| `--image_resolution` | 480 | Input image height |
| `--device` | `cuda:0` | GPU device |

## Module Structure

```
ace_sampler/
├── __init__.py            # Exports SamplerNet, SamplerTrainer, fill_buffer_with_sampler
├── model.py               # SamplerNet definition + save/load
├── uncertainty.py         # MCDropoutRegressor — feature-level MC uncertainty
├── options.py             # add_sampler_train_args, add_sampler_buffer_args, add_multi_sampler_train_args
├── trainer.py             # Phase 1: SamplerTrainer (single-scene, with optional MC Dropout)
├── multi_trainer.py       # Phase 1: MultiSceneSamplerTrainer (universal, multi-scene)
├── multi_dataset.py       # MultiSceneDataset + SceneEntry dataclass
├── buffer_sampler.py      # Phase 2: fill_buffer_with_sampler
└── visualize_sampler.py   # Confidence map + sampled points visualization
train_ace_sampler.py       # Phase 1 entry script — single-scene (project root)
train_ace_sampler_multi.py # Phase 1 entry script — multi-scene universal (project root)
ace_eval.py                # Shared evaluation function used by train_ace.py post-training
```

## Notes

- The ACE head passed to Phase 1 should be reasonably trained. A head trained for even a few epochs gives a meaningful reprojection error signal.
- `SamplerNet` is frozen during Phase 2 — it only influences which pixels are sampled, not the ACE loss.
- Buffer schema produced by `fill_buffer_with_sampler` is identical to `ace_trainer.py`: `features / target_px / gt_poses_inv / intrinsics / intrinsics_inv`.
- `ace_network.py` is not modified. The UncertaintyHead is a separate module that runs alongside the frozen ACE regressor.
