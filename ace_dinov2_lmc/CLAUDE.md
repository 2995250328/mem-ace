# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ACE-DINOv2-LMC: Accelerated Coordinate Encoding with DINOv2 ViT-L/14 backbone and GeoLMC (Geometric Latent Memory Compression) two-stage iterative training for visual camera relocalization.

**Two training modes:**
- **Vanilla** (`--use_lmc False`): Single-stage buffer training, fast baseline
- **LMC** (`--use_lmc True`): Two-stage iterative training (S1 memory alignment + S2 reprojection refinement), requires pre-built pooled memory file

## Environment Setup

```bash
# Working directory: always run from parent directory
cd /home/xwh/project/ace_depth

# Conda environment
conda activate ace  # PyTorch 2.0.0, CUDA 11.8, Python 3.8

# DINOv2 weights location
/data/xwh/checkpoints/dinov2_vitl14_pretrain.pth
```

## Training Commands

### ACE-G Flow (Main Use Case)

Standard configuration for experiments. Requires pre-built pooled memory file (GT patch format).

**7-Scenes example:**
```bash
python ace_dinov2_lmc/train_ace_dinov2_lmc.py \
    /data/xwh/7Scenes/pgt_7scenes_heads \
    heads_aceg_full_refill.pt \
    --device cuda:3 \
    --run_name heads_aceg_full_refill \
    --use_lmc True \
    --memory_path /path/to/7Scenes_heads_train_pooled_GT_patch.pt \
    --dinov2_path /data/xwh/checkpoints/dinov2_vitl14_pretrain.pth \
    --lmc_flow ace_g \
    --lmc_mode global \
    --num_latent_tokens 64 \
    --lmc_iterations 28 \
    --lmc_warmup_steps 2000 \
    --lmc_train_steps 600 \
    --s1_learning_rate_max 1e-4 \
    --s2_learning_rate_max 1e-3 \
    --training_buffer_size 2560000 \
    --buffer_size_final 7680000 \
    --epochs 24 \
    --batch_size 5120 \
    --samples_per_image 384 \
    --buffer_batch_size 1 \
    --buffer_on_cpu True \
    --image_resolution 518 \
    --s1_batch_size 16 \
    --s1_use_buffer True \
    --s1_loss_mode sample_per_image \
    --s1_buffer_refill_mode full \
    --ace_g_fusion_in_s2 True \
    --ace_g_fusion_lr_ratio 0.01 \
    --ace_g_cross_iter_eval True \
    --lmc_memory_preflight True \
    --lmc_memory_preflight_strict True \
    --eval_each_iteration True \
    --eval_after_train True \
    --keep_best_only True \
    --best_metric pct5 \
    --use_half True
```

**Indoor6 example** (large scene, add `--lmc_scene_center_max_distance 4.0`):
```bash
python ace_dinov2_lmc/train_ace_dinov2_lmc.py \
    /data/xwh/indoor6_ace/scene3 \
    scene3_aceg_full_refill.pt \
    --device cuda:1 \
    --run_name scene3_aceg_full_refill \
    --use_lmc True \
    --memory_path /path/to/Indoor6_scene3_train_pooled_GT.pt \
    --lmc_scene_center_max_distance 4.0 \
    [... same flags as above]
```

### Vanilla Training (No LMC)

```bash
python ace_dinov2_lmc/train_ace_dinov2_lmc.py \
    /data/xwh/7Scenes/pgt_7scenes_chess \
    chess_vanilla.pt \
    --use_lmc False \
    --device cuda:0
```

## Testing

```bash
python ace_dinov2_lmc/test_ace_dinov2_lmc.py \
    /data/xwh/7Scenes/pgt_7scenes_heads \
    output/.../best_K64_it28_heads_aceg_full_refill.pt \
    --device cuda:0 \
    --session lmc_test
```

## Memory Extraction Pipeline

Extract compressed scene memory from multi-view RGB-D data using BSE (Bilateral Supervoxel Extraction) pooling.

### Quick Start (Shell Script)

```bash
cd /home/xwh/project/ace_depth

# 7-Scenes chess, 20 views (default)
bash ace_dinov2_lmc/memory_extraction/extract_memory.sh

# Override via environment variables
SCENE_TRAIN=fire_train N_VIEWS=40 bash ace_dinov2_lmc/memory_extraction/extract_memory.sh

# Indoor6 scene
DATASET_TYPE=indoor6 SCENE_TRAIN=scene2a_train N_VIEWS=40 \
    bash ace_dinov2_lmc/memory_extraction/extract_memory.sh

# Ablation: different voxel size
VOXEL_SIZE=0.10 N_VIEWS=40 bash ace_dinov2_lmc/memory_extraction/extract_memory.sh

# Use DINOv2 fallback (instead of default MapAnything)
USE_MODEL=dinov2 bash ace_dinov2_lmc/memory_extraction/extract_memory.sh
```

### Direct Python Call

```bash
python -m ace_dinov2_lmc.memory_extraction.run_memory_extraction \
    /data/xwh/7Scenes/pgt_7scenes_chess \
    output/memory.pt \
    --n_memory 20 \
    --dataset_type 7scenes \
    --device cuda:0
```

### Key Shell Script Parameters

Set via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `DATASET_TYPE` | `7scenes` | `7scenes`, `indoor6`, `custom` |
| `SCENE_TRAIN` | `chess_train` | Scene name for extraction |
| `N_VIEWS` | `20` | Number of memory views |
| `VOXEL_SIZE` | `0.05` | BSE voxel grid size (meters) |
| `USE_MODEL` | `mapanything` | `mapanything` or `dinov2` |
| `GPU_ID` | `0` | GPU device index |

### Batch Extraction Examples

**All 7-Scenes:**
```bash
for scene in chess fire heads office pumpkin redkitchen stairs; do
    SCENE_TRAIN=${scene}_train bash ace_dinov2_lmc/memory_extraction/extract_memory.sh
done
```

**Voxel size ablation:**
```bash
for vs in 0.03 0.05 0.08 0.10; do
    VOXEL_SIZE=$vs N_VIEWS=40 bash ace_dinov2_lmc/memory_extraction/extract_memory.sh
done
```

## Architecture

### Core Implementation Files

All implementation files live in `ace_dinov2_lmc/`:

| File | Role |
|------|------|
| `train_ace_dinov2_lmc.py` | Training entry point |
| `test_ace_dinov2_lmc.py` | Evaluation script |
| `options_dinov2_lmc.py` | Full CLI argument parser |
| `trainer_dinov2_lmc.py` | `TrainerACEDINOv2LMC` two-stage trainer |

### Shared Dependencies (Root Level)

Accessed via `sys.path` from parent directory:

| File | Role |
|------|------|
| `trainer_dinov2.py` | Base `TrainerACEDINOv2` |
| `ace_network_dinov2.py` | DINOv2 `Regressor` |
| `dataset_dinov2.py` | ACE-format dataset loader |
| `dataset_wai_dinov2.py` | WAI-format dataset loader |
| `utils_lmc.py` | Shared LMC utilities |
| `ace_compressor.py` | `GeoLMC` compressor |
| `ace_fusion.py` | `LMCFeatureFusion` |
| `ace_loss.py` | `ReproLoss` |
| `ace_util.py` | General utilities |

### Memory Extraction Module

```
memory_extraction/
├── bse_pooling.py            # BSE algorithm with ray strategies
├── welford_meter.py          # Streaming normalization
├── run_memory_extraction.py  # Main extraction script
└── extract_memory.sh         # Shell launcher (env-var driven)
```

## Two-Stage Training Process

**Stage 1 (S1): Memory Alignment**
- Align query features with pre-built memory
- Train compressor and fusion modules
- Uses memory alignment loss

**Stage 2 (S2): Reprojection Refinement**
- Train regression head on scene coordinates
- Uses reprojection loss
- Optionally continues training fusion modules (`--ace_g_fusion_in_s2 True`)

**Iterative Training:**
- Alternates between S1 and S2 for `--lmc_iterations` cycles
- Each iteration: S1 warmup → S1 train → S2 train → evaluation
- Keeps best checkpoint based on `--best_metric` (default: `pct5`)

## Output Directory Structure

Training outputs go to `ace_dinov2_lmc/04_evaluation/`:

```
04_evaluation/
├── <run_name>/
│   ├── <timestamp>/
│   │   ├── best_K64_it28_<run_name>.pt    # Best checkpoint
│   │   ├── training_log.txt               # Training log
│   │   ├── config.json                    # Config snapshot
│   │   └── eval_results/                  # Per-iteration eval
└── memory_extract/
    └── <scene>/
        └── <config_str>/
            └── <timestamp>/
                ├── memory_bse.pt          # Extracted memory
                ├── extraction_config.json
                └── extraction_log.txt
```

## Key Constraints

1. **Working Directory**: Always run commands from `/home/xwh/project/ace_depth` (parent directory), not from `ace_dinov2_lmc/`
2. **Image Resolution**: Must be multiple of 14 (DINOv2 patch size). Default: 518 (37×14)
3. **Memory File Required**: LMC mode requires pre-built memory file from extraction pipeline
4. **GPU Memory**: ~12-16GB VRAM for training with default settings

## Important Notes

- Root-level stubs (`../train_ace_dinov2_lmc.py`, etc.) forward to `ace_dinov2_lmc/` files for backward compatibility
- Memory extraction uses MapAnything by default (better multi-scale features), DINOv2 as fallback
- BSE pooling preserves geometric boundaries via bilateral clustering with Otsu thresholding
- Output memory files use schema v1.2 with multi-strategy ray encoding (mean, dominant, first, all)
