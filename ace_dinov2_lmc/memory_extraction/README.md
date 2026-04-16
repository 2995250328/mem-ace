# BSE Memory Extraction

Bilateral Supervoxel Extraction (BSE) enhanced memory extraction pipeline for ACE DINOv2 LMC.

## Overview

This module extracts compressed scene memory from multi-view RGB-D data using:
- **BSE Pooling**: Boundary-preserving bilateral clustering with Otsu adaptive thresholding
- **Welford Normalization**: Numerically stable global metric normalization
- **Multi-Strategy Ray Encoding**: Preserves multi-view geometric information

## Quick Start

<!-- Updated 2026-03-31: added ACE dataset loader examples -->

### Using the Shell Script (Recommended)

```bash
cd /home/xwh/project/ace_depth

# === WAI Dataset (default) ===

# 7-Scenes chess, 20 views (default config, WAI loader)
bash ace_dinov2_lmc/memory_extraction/extract_memory.sh

# Override via environment variables
SCENE_TRAIN=fire_train N_VIEWS=40 bash ace_dinov2_lmc/memory_extraction/extract_memory.sh

# Indoor6 scene
DATASET_TYPE=indoor6 SCENE_TRAIN=scene2a_train N_VIEWS=40 bash ace_dinov2_lmc/memory_extraction/extract_memory.sh

# Strict FPS protocol (default): actual MapAnything inputs match the FPS list one-to-one
WAI_VIEW_MODE=fps_flat DATASET_TYPE=indoor6 SCENE_TRAIN=scene2a_train N_VIEWS=40 \
  bash ace_dinov2_lmc/memory_extraction/extract_memory.sh

# Reproduce original map-anything fps_memory.sh: first FPS anchor expands to 40 covisibility views
WAI_VIEW_MODE=original_multiview DATASET_TYPE=indoor6 SCENE_TRAIN=scene2a_train N_VIEWS=40 \
  bash ace_dinov2_lmc/memory_extraction/extract_memory.sh

# Covisibility-weighted FPS: preserve coverage while avoiding low-overlap large-baseline inputs
WAI_VIEW_MODE=covis_fps COVIS_FPS_ALPHA=1.0 COVIS_FPS_TAU=-1.0 \
  DATASET_TYPE=indoor6 SCENE_TRAIN=scene2a_train N_VIEWS=40 \
  bash ace_dinov2_lmc/memory_extraction/extract_memory.sh

# === ACE Dataset Loader (CamLocDatasetDINOv2) ===
# Works with any scene containing rgb/, poses/, calibration/ directories (7-Scenes / Indoor6 / custom)
#
# Path explanation:
#   DATASET_ROOT points to the scene root (e.g. pgt_7scenes_chess/),
#   the script auto-appends SCENE_TRAIN (e.g. train) to form the final dataset_path
#   (if DATASET_ROOT has no rgb/ subdirectory).
#   If DATASET_ROOT already has rgb/, no appending is done.

# 7-Scenes chess — script auto-appends /train (pgt_7scenes_chess/ has no rgb/, but pgt_7scenes_chess/train/ does)
DATASET_LOADER=ace DATASET_ROOT=/mnt/storage/xwh/7Scenes/pgt_7scenes_chess \
  SCENE_TRAIN=train N_VIEWS=20 GPU_ID=0 \
  bash ace_dinov2_lmc/memory_extraction/extract_memory.sh

# 7-Scenes heads — same auto-append logic
DATASET_LOADER=ace DATASET_ROOT=/mnt/storage/xwh/7Scenes/pgt_7scenes_heads \
  SCENE_TRAIN=train N_VIEWS=20 GPU_ID=0 \
  bash ace_dinov2_lmc/memory_extraction/extract_memory.sh

# Indoor6 scene2a — same format (scene2a/ has train/rgb/)
DATASET_LOADER=ace DATASET_ROOT=/mnt/storage/xwh/indoor6_ace/scene2a \
  SCENE_TRAIN=train N_VIEWS=40 GPU_ID=0 \
  bash ace_dinov2_lmc/memory_extraction/extract_memory.sh

# === Other Options ===

# Ablation: different voxel size
VOXEL_SIZE=0.10 N_VIEWS=40 bash ace_dinov2_lmc/memory_extraction/extract_memory.sh

# Use DINOv2 fallback (instead of default MapAnything)
USE_MODEL=dinov2 bash ace_dinov2_lmc/memory_extraction/extract_memory.sh

# Disable SOR (enabled by default)
ENABLE_SOR=false bash ace_dinov2_lmc/memory_extraction/extract_memory.sh
```

### Direct Python Call

**WAI dataset:**

```bash
python -m ace_dinov2_lmc.memory_extraction.run_memory_extraction \
    /mnt/storage/xwh/mapanything-dataset/wai_data/7scenes \
    /path/to/out/memory.pt \
    --n_memory 20 \
    --dataset_type 7scenes \
    --dataset_loader wai \
    --patch_depth_sampling nearest \
    --enable_sor \
    --device cuda:0
```

**ACE dataset (CamLocDatasetDINOv2):**

`dataset_path` should point directly to the directory containing `rgb/`, `poses/`, `depth/`, `calibration/`. Unlike WAI, ACE does not require `--scene_name`.

```bash
# dataset_path points to dir with rgb/ (7-Scenes: pgt_7scenes_chess/train/)
python -m ace_dinov2_lmc.memory_extraction.run_memory_extraction \
    /mnt/storage/xwh/7Scenes/pgt_7scenes_chess/train \
    /path/to/out/memory.pt \
    --n_memory 20 \
    --dataset_loader ace \
    --enable_sor \
    --device cuda:0

# Indoor6: same, point directly to train/ directory
python -m ace_dinov2_lmc.memory_extraction.run_memory_extraction \
    /mnt/storage/xwh/indoor6_ace/scene2a/train \
    /path/to/out/memory.pt \
    --n_memory 40 \
    --dataset_loader ace \
    --enable_sor \
    --depth_valid_range 0.02 100.0 \
    --device cuda:0
```

### Differences Between Dataset Loaders

| Feature | WAI (`--dataset_loader wai`) | ACE (`--dataset_loader ace`) |
|---------|------------------------------|------------------------------|
| Data format | MapAnything WAI directory structure | ACE CamLocDataset format |
| `dataset_path` | Dataset root (contains scene subdirectories) | **Points directly to dir with `rgb/`, `poses/`** |
| Scene selection | `--scene_name chess_train` | No extra parameter needed |
| Depth source | WAI format depth map (COLMAP sparse / GT) | ACE format depth map |
| Normalization | MapAnything style | ImageNet (DINOv2) |
| Shell auto-append | No appending (WAI DATASET_ROOT is the dataset root) | If `DATASET_ROOT/` has no `rgb/`, auto-appends `/${SCENE_TRAIN}` |
| Compatible data | `/mnt/storage/xwh/mapanything-dataset/wai_data/` | `/mnt/storage/xwh/7Scenes/pgt_7scenes_*/`, `/mnt/storage/xwh/indoor6_ace/*/` |

**ACE path concatenation rule**: When `DATASET_LOADER=ace`, the shell script checks if `DATASET_ROOT/rgb/` exists:
- If yes → uses `DATASET_ROOT` directly as `dataset_path`
- If no → appends `DATASET_ROOT/${SCENE_TRAIN}` (e.g. `pgt_7scenes_chess/train`) as `dataset_path`

## Complete CLI Reference

### Required Arguments

| Parameter | Description |
|-----------|-------------|
| `dataset_path` | Path to scene dataset (positional) |
| `output_path` | Output .pt file path (positional) |

### Memory Selection

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--n_memory` | 100 | Number of memory views to extract |

### Dataset Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--dataset_type` | `auto` | Dataset type: `auto`, `7scenes`, `indoor6`, `custom` |
| `--dataset_loader` | `wai` | **Dataset loader**: `wai` (MapAnything WAI) or `ace` (ACE CamLocDataset) |
| `--wai_view_mode` | `fps_flat` | WAI view loading protocol: `fps_flat` makes actual model inputs match the FPS list; `original_multiview` reproduces map-anything `fps_memory.sh` covisibility multi-view samples; `covis_fps` uses covisibility-weighted FPS |
| `--covis_fps_alpha` | `1.0` | Covisibility exponent for `covis_fps`; `0` degenerates to pure FPS, larger values prefer overlap |
| `--covis_fps_eps` | `1e-6` | Epsilon used by the `covis_fps` soft score |
| `--covis_fps_tau` | `-1.0` | Hard max-covisibility threshold for `covis_fps`; `<0` disables hard thresholding |
| `--covis_max_dist_to_ref` | `-1.0` | Locality constraint for `covis_fps`: max camera-center distance from the reference view in meters; `<0` disables |
| `--scene_name` | None | Scene name for WAI datasets (e.g. `chess_train`, `scene2a_train`) |

### BSE Pooling Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--voxel_size` | 0.05 | BSE voxel grid size (meters) |
| `--use_bse` | True | Enable BSE pooling |
| `--use_otsu` | True | Use Otsu adaptive thresholding |

### Feature Extraction

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--use_model` | `mapanything` | Feature extractor: `mapanything` or `dinov2` |
| `--model_str` | None | MapAnything model config string (default: `mapanything_store_intermediates_ace`) |
| `--model_config` | None | MapAnything config path |
| `--model_checkpoint` | None | MapAnything checkpoint path |
| `--dinov2_checkpoint` | `/mnt/storage/xwh/checkpoints/dinov2_vitl14_pretrain.pth` | DINOv2 weights |
| `--dinov2_intermediate_layers` | None | DINOv2 block indices for multi-scale (default: 8 DPT layers) |
| `--use_patch_based` | False | Patch-based vs bilinear upsampling |

### Depth Processing

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--depth_valid_range` | 0.1 6.0 | Valid depth range in meters (min max); Indoor6 recommended `0.1 100` |
| `--patch_depth_sampling` | `nearest_valid` | Depth sampling: `nearest_valid`, `bilinear`, `nearest` |

### SOR Filtering

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--enable_sor` | False | Enable Statistical Outlier Removal |
| `--sor_k` | 20 | SOR k neighbors |
| `--sor_std_ratio` | 2.0 | SOR standard deviation ratio |

### Ray Pooling

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--ray_pool_strategy` | `mean` | Ray pooling: `mean`, `dominant`, `first`, `all` |
| `--save_all_ray_strategies` | True | Save all ray representations |

### Advanced

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--device` | `cuda:0` | GPU device |
| `--temp_dir` | `/dev/shm` | Temporary directory for chunks |
| `--pose_eval_translation_ok_m` | 0.1 | Translation error threshold (meters) |
| `--pose_eval_strict` | False | Exit on any non-ref view exceeding threshold |

## Dataset Loaders

### WAI Loader (Default)

MapAnything WAI format dataset. Requires `--scene_name` parameter. The `dataset_path` should point to the dataset root containing scene subdirectories.

```bash
# 7-Scenes
python -m ace_dinov2_lmc.memory_extraction.run_memory_extraction \
    /mnt/storage/xwh/mapanything-dataset/wai_data/7scenes \
    output/memory.pt \
    --dataset_loader wai \
    --dataset_type 7scenes \
    --scene_name chess_train \
    --n_memory 20

# Indoor6
python -m ace_dinov2_lmc.memory_extraction.run_memory_extraction \
    /mnt/storage/xwh/mapanything-dataset/wai_data/indoor6 \
    output/memory.pt \
    --dataset_loader wai \
    --dataset_type indoor6 \
    --scene_name scene2a_train \
    --n_memory 40 \
    --depth_valid_range 0.02 100.0
```

### ACE Loader (CamLocDatasetDINOv2)

ACE-format dataset loader. `dataset_path` points directly to the directory containing `rgb/`, `poses/`, `calibration/`. Compatible with the ACE training pipeline — any scene used for ACE training can be used here.

```bash
# 7-Scenes: dataset_path points to the train/ directory
python -m ace_dinov2_lmc.memory_extraction.run_memory_extraction \
    /mnt/storage/xwh/7Scenes/pgt_7scenes_chess/train \
    output/memory.pt \
    --dataset_loader ace \
    --n_memory 20 \
    --enable_sor

# Indoor6: same, point to train/ directory
python -m ace_dinov2_lmc.memory_extraction.run_memory_extraction \
    /mnt/storage/xwh/indoor6_ace/scene2a/train \
    output/memory.pt \
    --dataset_loader ace \
    --n_memory 40 \
    --enable_sor \
    --depth_valid_range 0.02 100.0
```

**Key differences:**
- WAI: Requires `--scene_name`, `dataset_path` is the dataset root
- ACE: No `--scene_name` needed, `dataset_path` is the scene directory itself

## Output Directory Structure

All outputs are organized under `ace_dinov2_lmc/04_evaluation/memory_extract/`:

```
memory_extract/
├── chess/
│   ├── 20v_v0.05_bilinear_l2/
│   │   └── 20260328_143000/
│   │       ├── memory_bse.pt              # Main output
│   │       ├── extraction_config.json     # Config snapshot
│   │       ├── extraction_log.txt         # Run log
│   │       ├── loaded_model_input_views.json  # Actual model input filenames / FPS index check
│   │       ├── depth_validation/          # Depth and RGB alignment check (aligned with fps_memory)
│   │       │   ├── view_00_rgb_raw.png
│   │       │   ├── view_XX_depth_vis.png
│   │       │   ├── view_XX_depth_on_rgb.png
│   │       │   └── view_00_raw_depth_pointcloud.ply  # If Open3D available and view0 has valid points
│   │       └── model_input_vis/           # Normalized model inputs, annotated with filename/FPS index
│   │           └── view_XX_model_input.png
│   └── 20v_v0.10_bilinear_l2/            # Ablation: different voxel size
│       └── 20260328_150000/
│           └── ...
├── fire/
│   └── 40v_v0.05_bilinear_l2/
│       └── ...
└── scene2a/
    └── 40v_v0.05_bilinear_l2/
        └── ...
```

**Naming convention**: `<N>v_<voxel_size>_<extraction_mode>[_sor][_l2]`

| Component | Meaning |
|-----------|---------|
| `20v` | Number of memory views |
| `v0.05` | Voxel size |
| `bilinear` / `patch` | Feature extraction mode |
| `sor` | SOR filtering enabled |
| `l2` | L2 normalization on features |

## Shell Script Parameters

All parameters are set via environment variables:

### Dataset & Scene

| Variable | Default | Description |
|----------|---------|-------------|
| `DATASET_TYPE` | `7scenes` | Dataset type: `7scenes`, `indoor6`, `custom` |
| `SCENE_TRAIN` | `chess_train` | Scene name for extraction |
| `SCENE_TEST` | auto-derived | Test scene name |
| `DATASET_ROOT` | auto | Root directory (auto-set by `DATASET_TYPE`) |
| `N_VIEWS` | `20` | Number of memory views |
| `GPU_ID` | `0` | GPU device index |
| `DATASET_LOADER` | `wai` | Dataset loader: `wai` or `ace` |
| `WAI_VIEW_MODE` | `fps_flat` | WAI view protocol: `fps_flat` uses one actual input per FPS index; `original_multiview` reproduces map-anything `fps_memory.sh` covisibility multi-view behavior; `covis_fps` uses covisibility-weighted FPS |
| `COVIS_FPS_ALPHA` | `1.0` | Covisibility exponent for `covis_fps` |
| `COVIS_FPS_EPS` | `1e-6` | Epsilon used by the `covis_fps` soft score |
| `COVIS_FPS_TAU` | `-1.0` | Hard max-covisibility threshold for `covis_fps`; `<0` disables hard thresholding |
| `COVIS_MAX_DIST_TO_REF` | `-1.0` | Locality constraint for `covis_fps`: max camera-center distance from the reference view in meters; `<0` disables |

**Dataset path auto-derivation:**

| DATASET_TYPE | Resolved Path | Example |
|--------------|---------------|---------|
| `7scenes` | `/mnt/storage/xwh/7Scenes/pgt_7scenes_<scene>` | `chess_train` → `/mnt/storage/xwh/7Scenes/pgt_7scenes_chess` |
| `indoor6` | `/mnt/storage/xwh/mapanything-dataset/wai_data/indoor6` | `scene2a_train` → `.../scene2a_train` |
| `custom` | `$DATASET_ROOT/$SCENE_TRAIN` | user-specified |

**Depth range auto-derivation:**

| DATASET_TYPE | DEPTH_MIN | DEPTH_MAX |
|--------------|-----------|-----------|
| `7scenes` | 0.02 | 6.0 |
| `indoor6` | 0.02 | 100.0 (COLMAP / GT sparse depth, avoid clipping far points) |

**Grid depth sampling (aligned with fps_memory.sh `PATCH_DEPTH_SAMPLING`):**

| Variable | Default (by `DATASET_TYPE`) | Description |
|----------|------------------------------|-------------|
| `PATCH_DEPTH_SAMPLING` | indoor6 → `nearest_valid`; otherwise → `nearest` | `nearest`: single pixel; `median`: 3x3 median; `nearest_valid`: median of valid depths within radius 10 (**recommended for sparse depth**) |

### BSE Pooling

| Variable | Default | Description |
|----------|---------|-------------|
| `VOXEL_SIZE` | `0.05` | Coarse voxel grid size (meters) |
| `USE_OTSU` | `true` | Use Otsu adaptive thresholding |
| `ENABLE_SOR` | `true` | Enable Statistical Outlier Removal |

## Alignment with map-anything `fps_memory.sh`

- **WAI view loading protocol**: `WAI_VIEW_MODE=fps_flat` (default) makes each FPS index load exactly one image, so the actual `input_views` sent to MapAnything match `select_optimal_memory_indices()` one-to-one. `WAI_VIEW_MODE=original_multiview` reproduces the original `fps_memory.sh` behavior: WAI `dataset[idx]` uses `num_views=N_VIEWS`, so the first FPS anchor can expand to `N_VIEWS` covisibility views and later FPS indices may not be consumed. `WAI_VIEW_MODE=covis_fps` reads the real WAI `scene_root/covisibility/v0/*.npy` matrix and uses `score = normalized_min_dist_to_selected * (eps + normalized_max_covis_to_selected)^alpha` to trade off coverage and overlap. These are different experiment protocols; do not compare PoseEval / pooled-memory point counts as if they used the same selected frames.
- **Input audit artifacts**: each run logs `[Data] Actual model input views:`, saves `loaded_model_input_views.json`, and annotates `model_input_vis/view_XX_model_input.png` with `src`, `actual_idx`, `fps_idx`, and `outer/inner` so the loaded filenames can be checked against the FPS list.
- **Grid depth sampling**: Aligned with `mapanything/tasks/run_memory_extraction.py` `generate_patch_point_cloud`. Supports `nearest` / `median` / `nearest_valid`. Indoor6 and other **sparse depth** must use **`nearest_valid`** (median of valid depths within grid center neighborhood), otherwise you may get "whole image has depth but almost no valid points on the grid".
- **Shell defaults**: `DATASET_TYPE=indoor6` defaults to `PATCH_DEPTH_SAMPLING=nearest_valid`; `7scenes` defaults to `nearest`.
- **Depth validation images**: After selecting frames and loading views, outputs `depth_validation/` and `model_input_vis/` in the run directory (behavior aligned with map-anything; reuses `_save_raw_rgb` / `_save_depth_on_rgb_vis` from `mapanything.tasks.run_memory_extraction` if importable; sparse depth uses scatter plot, very dense uses vectorized turbo coloring).
- **Depth tensor layout**: WAI commonly uses `[H,W,1]` (HWC), which is normalized to `[H,W]` before unprojection to avoid misinterpreting as `[B,H,W]` and only taking one row.
| `RAY_POOL_STRATEGY` | `mean` | Strategy: `mean`, `dominant`, `first`, `all` |

### Feature Extraction

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_PATCH_BASED` | `false` | Patch-based (grid) vs bilinear upsampling |
| `USE_L2_NORMALIZATION` | `true` | L2 normalize features |
| `DINOV2_CHECKPOINT` | `/mnt/storage/xwh/checkpoints/dinov2_vitl14_pretrain.pth` | DINOv2 weights |
| `USE_MODEL` | `mapanything` | Feature extractor: `mapanything` or `dinov2` |
| `MODEL_STR` | `mapanything_store_intermediates_ace` | MapAnything config name |
| `MODEL_CHECKPOINT` | `/mnt/storage/xwh/checkpoints/facebook_map-anything.pth` | MapAnything checkpoint |

### Output

| Variable | Default | Description |
|----------|---------|-------------|
| `OUTPUT_ROOT` | `ace_dinov2_lmc/04_evaluation/memory_extract` | Root output directory |

## Output Format

The output `.pt` file (schema v1.2) contains:

```python
{
    'schema_version': '1.2',

    # Point-level data
    'points': P_norm,                  # [N, 3] float32, normalized
    'features': F_bse,                 # [N, C] float16, multi-scale DINOv2 features
    'colors': C_bse,                   # [N, 3] float32, RGB [0,1]
    'cluster_sizes': sizes,            # [N] int64, points per cluster

    # Ray direction representations
    'ray_dirs': D_mean,                # [N, 3] float32, pooled by strategy
    'ray_dirs_mean': D_mean,           # [N, 3] float32, mean + normalize
    'ray_dirs_dominant': D_dominant,   # [N, 3] float32, highest similarity
    'ray_dirs_first': D_first,         # [N, 3] float32, first in cluster
    'plucker_rays': plucker,           # [N, 6] float32, (direction, moment)

    # View-level camera information (M = memory views)
    'view_camera_centers': centers,    # [M, 3] float32
    'view_camera_rotations': rots,     # [M, 3, 3] float32
    'view_camera_intrinsics': Ks,      # [M, 3, 3] float32
    'view_plucker_main_rays': plucker, # [M, 6] float32

    # Normalization statistics
    'mu': mu_scene,                    # [3] float32, scene mean
    'sigma': sigma_scene,              # scalar float32, scene std
}
```

## Ray Direction Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `mean` | Average + L2 normalize | Default baseline |
| `dominant` | Select ray with highest feature similarity | Preserve strongest viewpoint |
| `first` | Use first ray in cluster | Preserve original direction |
| `all` | Keep all rays per cluster | Full information for learned compression |

### Plucker Encoding

6D ray representation: `plucker = (direction, moment)`, where `moment = camera_center x direction`.

## Common Workflows

### Batch extraction across all 7-Scenes

```bash
for scene in chess fire heads office pumpkin redkitchen stairs; do
    SCENE_TRAIN=${scene}_train bash ace_dinov2_lmc/memory_extraction/extract_memory.sh
done
```

### Voxel size ablation

```bash
for vs in 0.03 0.05 0.08 0.10; do
    VOXEL_SIZE=$vs N_VIEWS=40 bash ace_dinov2_lmc/memory_extraction/extract_memory.sh
done
```

### Indoor6 extraction

```bash
for scene in scene1 scene2a scene3 scene4a scene5 scene6; do
    DATASET_TYPE=indoor6 SCENE_TRAIN=${scene}_train N_VIEWS=40 \
        bash ace_dinov2_lmc/memory_extraction/extract_memory.sh
done
```

### Using ACE dataset loader

```bash
# 7-Scenes chess
DATASET_LOADER=ace DATASET_ROOT=/mnt/storage/xwh/7Scenes/pgt_7scenes_chess \
    SCENE_TRAIN=train N_VIEWS=20 GPU_ID=0 \
    bash ace_dinov2_lmc/memory_extraction/extract_memory.sh

# 7-Scenes batch
for scene in chess fire heads office pumpkin redkitchen stairs; do
    DATASET_LOADER=ace DATASET_ROOT=/mnt/storage/xwh/7Scenes/pgt_7scenes_${scene} \
        SCENE_TRAIN=train N_VIEWS=20 GPU_ID=0 \
        bash ace_dinov2_lmc/memory_extraction/extract_memory.sh
done

# Indoor6 scene2a
DATASET_LOADER=ace DATASET_ROOT=/mnt/storage/xwh/indoor6_ace/scene2a \
    SCENE_TRAIN=train N_VIEWS=40 GPU_ID=0 \
    bash ace_dinov2_lmc/memory_extraction/extract_memory.sh

# Indoor6 batch
for scene in scene1 scene2a scene3 scene4a scene5 scene6; do
    DATASET_LOADER=ace DATASET_ROOT=/mnt/storage/xwh/indoor6_ace/${scene} \
        SCENE_TRAIN=train N_VIEWS=40 GPU_ID=0 \
        bash ace_dinov2_lmc/memory_extraction/extract_memory.sh
done
```

## Validation

After extraction, validate the memory file:

```bash
python ace_dinov2_lmc/memory_extraction/validate_memory.py \
    ace_dinov2_lmc/04_evaluation/memory_extract/chess/20v_v0.05_bilinear_l2/.../memory_bse.pt \
    --save_ply
```

This generates a PLY file for visualization in MeshLab, CloudCompare, or Open3D.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ValueError: No points accumulated` / very few valid grid points | For Indoor6, ensure `PATCH_DEPTH_SAMPLING=nearest_valid` (default for indoor6); check `--depth_valid_range` matches data (e.g. `0.02 100`) |
| Depth shape anomaly in logs (e.g. `[518,1]`) | HWC/`[H,W,1]` parsing has been fixed; use current `run_memory_extraction.py` |
| `features_flat` / `valid_mask` device mismatch | Fixed: feature indexing aligns with CPU/GPU |
| ACE loader FileNotFoundError: `.../rgb` not found | Ensure `DATASET_ROOT` points to scene root (e.g. `pgt_7scenes_chess/`), script auto-appends `/${SCENE_TRAIN}` if no `rgb/` found |
| Depth validation phase stalls | Vectorized coloring used for very dense depth; check disk I/O and Matplotlib backend |
| OOM | Reduce `N_VIEWS`, increase `VOXEL_SIZE`, use smaller GPU |
| Dataset not found | Check `DATASET_ROOT` and `SCENE_TRAIN`; script prints resolved path |
| `mapanything` import error | Script auto-adds `/home/xwh/project/map-anything` to PYTHONPATH; visualization falls back to built-in implementation |
| Slow processing | Increase `VOXEL_SIZE`, disable `ENABLE_SOR` |

## Dependencies

- PyTorch >= 2.0
- MapAnything — default multi-scale feature extractor (DINOv2 + AAT + DPT)
- DINOv2 (facebookresearch/dinov2) — fallback feature extractor
- Matplotlib, PIL — depth/RGB validation image export
- Open3D (optional — for `view_00_raw_depth_pointcloud.ply`)
