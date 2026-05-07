# ACE-DINOv2-LMC

ACE (Accelerated Coordinate Encoding) with DINOv2 ViT-L/14 backbone and GeoLMC (Geometric Latent Memory Compression) two-stage iterative training.

## Overview

Two training modes:

- **Vanilla** (`--use_lmc False`): single-stage buffer training, fast baseline
- **LMC** (`--use_lmc True`): two-stage iterative training (S1 memory alignment + S2 reprojection refinement), requires a pre-built pooled memory file

## File Structure

All implementation files live in this directory:

| File | Role |
|---|---|
| `train_ace_dinov2_lmc.py` | Training entry point |
| `test_ace_dinov2_lmc.py` | Evaluation script |
| `options_dinov2_lmc.py` | Full CLI argument parser |
| `trainer_dinov2_lmc.py` | `TrainerACEDINOv2LMC` two-stage trainer |

Shared root-level dependencies (accessed via `sys.path`):

| File | Role |
|---|---|
| `../trainer_dinov2.py` | Base `TrainerACEDINOv2` |
| `../ace_network_dinov2.py` | DINOv2 `Regressor` |
| `../dataset_dinov2.py` | ACE-format dataset loader |
| `../dataset_wai_dinov2.py` | WAI-format dataset loader |
| `../utils_lmc.py` | Shared LMC utilities |
| `../ace_compressor.py` | `GeoLMC` compressor |
| `../ace_fusion.py` | `LMCFeatureFusion` |
| `../ace_loss.py` | `ReproLoss` |
| `../ace_util.py` | General utilities |

Root-level stubs (`../train_ace_dinov2_lmc.py`, `../test_ace_dinov2_lmc.py`, etc.) forward to these files for backward compatibility.

## Training Commands

All commands run from the **project root** (`ace_depth/`).

### ACE-G flow (main use case)

<!-- Updated 2026-03-30: clarified memory file naming and training stages -->

The standard configuration used in experiments. **Requires a pre-built pooled memory file** (output of `memory_extraction/` pipeline, typically named `memory_bse.pt`). The `*_GT_patch.pt` naming in some older scripts refers to the same pooled format.

<!-- Updated 2026-03-30: added S1/S2 explanation and parameter context -->

**S1/S2 iterative training:**
- **S1 (memory alignment)**: Trains GeoLMC compressor + LMCFeatureFusion to align latent memory with backbone features
- **S2 (reprojection refinement)**: Trains regression head on fused features; optionally continues fusion training (`--ace_g_fusion_in_s2 True`)
- Cycles alternate for `--lmc_iterations` (default 28) rounds, with evaluation after each cycle

**Key parameter groups not shown in examples:**
- `--head_reset_strategy {first_only,every,output_only,none}`: How regression head resets between S1/S2 cycles (default: `first_only`)
- `--best_metric {pct5,pct10_5,composite,rt_error}`: Metric for best checkpoint selection (default: `pct5`)
- `--use_half`: Half-precision training (default: True)
- `--buffer_on_cpu`: Keep training buffer on CPU to save GPU memory (default: True)
- `--data_backend {ace,wai}`: Dataset format (default: `ace`)

**Memory file note:** The `--memory_path` must point to a pooled memory `.pt` file produced by the extraction pipeline (`memory_extraction/extract_memory.sh`). This file contains pooled 3D points, multi-scale features, ray directions, and camera info.

**7-Scenes example:**
```bash
python train_ace_dinov2_lmc.py \
    /home/xwh/data/7Scenes/pgt_7scenes_heads \
    heads_aceg_full_refill.pt \
    --device cuda:3 \
    --run_name heads_aceg_full_refill \
    --use_lmc True \
    --memory_path /path/to/7Scenes_heads_train_pooled_GT_patch.pt \
    --dinov2_path /home/xwh/data/checkpoints/dinov2_vitl14_pretrain.pth \
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
python train_ace_dinov2_lmc.py \
    /home/xwh/data/indoor6_ace/scene3 \
    scene3_aceg_full_refill.pt \
    --device cuda:1 \
    --run_name scene3_aceg_full_refill \
    --use_lmc True \
    --memory_path /path/to/Indoor6_scene3_train_pooled_GT.pt \
    --lmc_scene_center_max_distance 4.0 \
    ... (same flags as above)
```

### Vanilla (no LMC, fast baseline)

```bash
python train_ace_dinov2_lmc.py \
    /home/xwh/data/7Scenes/pgt_7scenes_chess \
    chess_vanilla.pt \
    --use_lmc False \
    --device cuda:0
```

## Testing

```bash
python test_ace_dinov2_lmc.py \
    /home/xwh/data/7Scenes/pgt_7scenes_heads \
    output/.../best_K64_it28_heads_aceg_full_refill.pt \
    --device cuda:0 \
    --session lmc_test
```

## Research Stages

| Folder | Content |
|---|---|
| `00_ideas/` | Raw idea notes and reuse map |
| `01_design/` | Proposal and design documents |
| `02_architecture/` | System architecture |
| `03_implementation/` | Implementation notes |
| `04_evaluation/` | Evaluation plans and logs |

Current ACE-G global audit and refactor TODO:

- `ACE_G_GLOBAL_CODE_AUDIT_TODO.md`
