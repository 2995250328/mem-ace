# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ACE-DINOv2-LMC: Accelerated Coordinate Encoding with DINOv2 ViT-L/14 backbone and GeoLMC (Geometric Latent Memory Compression) two-stage iterative training for visual camera relocalization.

**Two training modes:**
- **Vanilla** (`--use_lmc False`): Single-stage buffer training, fast baseline
- **LMC** (`--use_lmc True`): Two-stage iterative training (S1 memory alignment + S2 reprojection refinement), requires pre-built pooled memory file

## Environment Setup

```bash
cd /home/xwh/project/ace_depth  # Always run from parent directory
conda activate ace               # PyTorch 2.0.0, CUDA 11.8, Python 3.8

# Build DSAC* C++ bindings (required for RANSAC pose estimation in testing)
cd dsacstar && python setup.py install && cd ..
```

DINOv2 weights: `/mnt/storage/xwh/checkpoints/dinov2_vitl14_pretrain.pth`

## Commands

All commands run from the project root (`ace_depth/`).

### Training

**ACE-G flow (LMC, main use case):**
```bash
python ace_dinov2_lmc/train_ace_dinov2_lmc.py \
    /mnt/storage/xwh/7Scenes/pgt_7scenes_heads \
    heads_aceg_full_refill.pt \
    --device cuda:3 \
    --run_name heads_aceg_full_refill \
    --use_lmc True \
    --memory_path /path/to/7Scenes_heads_train_pooled_GT_patch.pt \
    --dinov2_path /mnt/storage/xwh/checkpoints/dinov2_vitl14_pretrain.pth \
    --lmc_flow ace_g --lmc_mode global \
    --num_latent_tokens 64 --lmc_iterations 28 \
    --lmc_warmup_steps 2000 --lmc_train_steps 600 \
    --s1_learning_rate_max 1e-4 --s2_learning_rate_max 1e-3 \
    --training_buffer_size 2560000 --buffer_size_final 7680000 \
    --epochs 24 --batch_size 5120 --samples_per_image 384 \
    --buffer_batch_size 1 --buffer_on_cpu True \
    --image_resolution 518 \
    --s1_batch_size 16 --s1_use_buffer True \
    --s1_loss_mode sample_per_image --s1_buffer_refill_mode full \
    --ace_g_fusion_in_s2 True --ace_g_fusion_lr_ratio 0.01 \
    --ace_g_cross_iter_eval True \
    --lmc_memory_preflight True --lmc_memory_preflight_strict True \
    --eval_each_iteration True --eval_after_train True \
    --keep_best_only True --best_metric pct5 --use_half True
```

**Indoor6 (large scene, add `--lmc_scene_center_max_distance 4.0`).**

**Vanilla (no LMC):**
```bash
python ace_dinov2_lmc/train_ace_dinov2_lmc.py \
    /mnt/storage/xwh/7Scenes/pgt_7scenes_chess \
    chess_vanilla.pt \
    --use_lmc False --device cuda:0
```

### Testing

```bash
python ace_dinov2_lmc/test_ace_dinov2_lmc.py \
    /mnt/storage/xwh/7Scenes/pgt_7scenes_heads \
    output/.../best_K64_it28_heads_aceg_full_refill.pt \
    --device cuda:0 --session lmc_test
```

### Memory Extraction

```bash
# 7-Scenes chess, 20 views (default)
bash ace_dinov2_lmc/memory_extraction/extract_memory.sh

# Override via env vars
SCENE_TRAIN=fire_train N_VIEWS=40 bash ace_dinov2_lmc/memory_extraction/extract_memory.sh

# Indoor6
DATASET_TYPE=indoor6 SCENE_TRAIN=scene2a_train N_VIEWS=40 \
    bash ace_dinov2_lmc/memory_extraction/extract_memory.sh

# Direct Python
python -m ace_dinov2_lmc.memory_extraction.run_memory_extraction \
    /mnt/storage/xwh/7Scenes/pgt_7scenes_chess output/memory.pt \
    --n_memory 20 --dataset_type 7scenes --device cuda:0

# Validate extracted memory
python -m ace_dinov2_lmc.memory_extraction.validate_memory \
    /path/to/memory_bse.pt --device cuda:0

# Batch all 7-Scenes
for scene in chess fire heads office pumpkin redkitchen stairs; do
    SCENE_TRAIN=${scene}_train bash ace_dinov2_lmc/memory_extraction/extract_memory.sh
done
```

Shell script env vars: `DATASET_TYPE` (7scenes/indoor6), `SCENE_TRAIN` (chess_train), `N_VIEWS` (20), `VOXEL_SIZE` (0.05), `USE_MODEL` (mapanything/dinov2), `GPU_ID` (0).

## Architecture

### Import Convention

All code inside `ace_dinov2_lmc/` uses `sys.path.insert(0, parent_dir)` at module level to import shared dependencies from the parent `ace_depth/` directory. Root-level stubs (`../train_ace_dinov2_lmc.py`, `../test_ace_dinov2_lmc.py`) forward here for backward compatibility.

### Class Hierarchy

```
TrainerACEDINOv2          (trainer_dinov2.py)       — vanilla DINOv2 ACE trainer
  └─ TrainerACEDINOv2LMC  (trainer_dinov2_lmc.py)   — adds LMC two-stage iterative training
```

When `use_lmc=False`, `TrainerACEDINOv2LMC` degrades to the parent class behavior.

### Training Dispatch

`train()` routes based on `--lmc_flow`:

| Flow | Method | Description |
|------|--------|-------------|
| (vanilla, iterations=1) | `super().train()` | Standard ACE single-stage |
| (vanilla, iterations>1) | `_train_vanilla_iterations()` | Repeated single-stage with eval |
| `iterative` | `_train_iterative()` | Original LMC S1/S2 loop |
| `ace_g` | `_train_ace_g()` | Main flow: S1/S2 with buffer refill and fusion |

### Two-Stage Iterative Training (LMC Mode)

Each iteration cycle:

1. **S1 (Memory Alignment)**: Train GeoLMC compressor + LMCFeatureFusion to align latent memory with backbone features. Has warmup phase then training phase. Supports early stopping.
2. **S2 (Reprojection Refinement)**: Train regression head on scene coordinates using reprojection loss. Optionally continues fusion training (`--ace_g_fusion_in_s2`).

Data flow within one iteration:
```
Pre-built memory (.pt file)
  → load_memory_features() → memory_dict (pooled_points, pooled_features, scene_center)
  → _compress_memory() → GeoLMC cross-attention → compressed latent tokens
  → _fuse_features() → LMCFeatureFusion → fused features (backbone + memory)
  → training buffer (features + target_px + poses + intrinsics)
  → run_epoch() → training_step() → regression head → 3D scene coordinate prediction
  → ReproLoss → backprop
```

Two buffer schemas exist:
- `raw_buffer`: stores raw backbone features (ACE-G and S1)
- `fused_buffer`: stores fused backbone+memory features (iterative flow S2)

### Core Files (in `ace_dinov2_lmc/`)

| File | Role |
|------|------|
| `train_ace_dinov2_lmc.py` | Training entry point — validates args, builds trainer, runs training |
| `test_ace_dinov2_lmc.py` | Evaluation — auto-detects LMC vs vanilla checkpoint |
| `options_dinov2_lmc.py` | Full CLI argument parser (all `--` flags) |
| `trainer_dinov2_lmc.py` | `TrainerACEDINOv2LMC` — 2800+ line trainer with S1/S2 logic |

### Shared Dependencies (Parent Level `../`)

| File | Role |
|------|------|
| `trainer_dinov2.py` | Base `TrainerACEDINOv2` |
| `ace_network_dinov2.py` | DINOv2 `Regressor` network |
| `dataset_dinov2.py` | ACE-format dataset loader (`--data_backend ace`) |
| `dataset_wai_dinov2.py` | WAI-format dataset loader (`--data_backend wai`) |
| `utils_lmc.py` | LR schedulers, `load_memory_features`, `preflight_memory_features`, normalization |
| `ace_compressor.py` | `GeoLMC` cross-attention compressor |
| `ace_fusion.py` | `LMCFeatureFusion` module |
| `ace_loss.py` | `ReproLoss` implementation |
| `ace_util.py` | General utilities |

### Memory Extraction (`memory_extraction/`)

| File | Role |
|------|------|
| `bse_pooling.py` | BSE (Bilateral Supervoxel Extraction) with ray strategies and Otsu thresholding |
| `welford_meter.py` | Streaming normalization (Welford's online algorithm) |
| `run_memory_extraction.py` | Main extraction script |
| `validate_memory.py` | Memory file validation |
| `extract_memory.sh` | Shell launcher (env-var driven) |

### Research Stage Directories

Each idea has `00_ideas/` through `04_evaluation/` subdirs (see `.project_index.md` for status):
- `dino_lmc_base/` — initial LMC implementation (stage-5, complete)
- `memory_extraction/` — BSE memory pipeline (stage-5, complete)
- `integrate_bse_memory/` — adapter layer for BSE→LMC (stage-1, active)

## Key CLI Parameters

### LMC Core
- `--use_lmc` (True): Enable LMC mode
- `--memory_path`: Path to pre-built pooled memory file (required for LMC)
- `--lmc_flow` (`ace_g`): Flow type — `ace_g` (main) or `iterative`
- `--lmc_mode` (`global`): Memory mode — `global`, `local`, `hierarchical`, `learned`
- `--num_latent_tokens` (64): Latent tokens for GeoLMC compressor
- `--num_attn_layers` (4): Cross-attention layers in compressor
- `--geo_sigma` (0.5): Gaussian kernel sigma for position encoding

### S1/S2 Iteration Control
- `--lmc_iterations` (28): Number of S1/S2 cycles
- `--lmc_warmup_steps` (2000): S1 warmup steps
- `--lmc_train_steps` (600): S1 training steps after warmup
- `--head_reset_strategy` (`first_only`): `first_only`, `every`, `output_only`, `none`

### S1 Training
- `--s1_batch_size` (28): Batch size for S1
- `--s1_use_buffer` (False): Use buffer in S1
- `--s1_loss_mode` (`full_map`): `full_map`, `sample_per_image`, `sample_pooled`
- `--s1_buffer_refill_mode` (`full`): `full` or `partial`
- `--s1_learning_rate_max` (1e-4): S1 learning rate

### S2 Training
- `--s2_learning_rate_max` (1e-3): S2 learning rate
- `--ace_g_fusion_in_s2` (False): Continue fusion training in S2
- `--ace_g_fusion_lr_ratio` (0.01): Fusion LR scale in S2

### Training Defaults
- `--num_head_blocks` (4): Regression head depth
- `--epochs` (24): Training epochs
- `--training_buffer_size` (2560000): Buffer size
- `--buffer_size_final` (3x training_buffer_size): Final iteration buffer size
- `--batch_size` (5120): Training batch size
- `--samples_per_image` (512): Samples per image
- `--use_half` (True): Half-precision training
- `--buffer_on_cpu` (True): Keep buffer on CPU
- `--repro_loss_type` (`dyntanh`): Reprojection loss type
- `--data_backend` (`ace`): `ace` or `wai` format
- `--image_resolution` (518): Must be multiple of 14

### Evaluation
- `--eval_each_iteration` (True): Run eval after each S1/S2 cycle
- `--keep_best_only` (True): Only keep best checkpoint
- `--best_metric` (`pct5`): Metric for checkpoint selection — `pct5`, `pct10_5`, `composite`, `rt_error`

### Early Stopping (S1)
- `--s1_early_stop` (True): Enable S1 early stopping
- `--s1_early_stop_patience` (180): Patience in steps
- `--s1_early_stop_min_updates` (400): Min updates before checking

## Output Directory Structure

Training outputs go to `ace_dinov2_lmc/04_evaluation/`:

```
04_evaluation/
├── <run_name>/
│   └── <timestamp>/
│       ├── best_K64_it28_<run_name>.pt    # Best checkpoint
│       ├── training_log.txt               # Training log
│       ├── config.json                    # Config snapshot
│       └── eval_results/                  # Per-iteration eval
└── memory_extract/
    └── <scene>/<config_str>/<timestamp>/
        ├── memory_bse.pt                  # Extracted memory
        ├── extraction_config.json
        └── extraction_log.txt
```

## Key Constraints

1. **Working Directory**: Always run commands from `/home/xwh/project/ace_depth` (parent directory)
2. **Image Resolution**: Must be multiple of 14 (DINOv2 patch size). Default: 518 (37x14)
3. **Memory File Required**: LMC mode requires pre-built memory file (pooled format with `pooled_points`/`pooled_features` keys)
4. **GPU Memory**: ~12-16GB VRAM for training with default settings

## Known Issues

- **Obsolete ablation scripts**: `scripts/lmc/ablation_*.sh` use parameters that do NOT exist (`--enable_compression`, `--compression_ratio`, `--use_depth`, `--use_superpoint`, `--use_intrinsics`). See `scripts/lmc/README.md` for how to adapt them.
