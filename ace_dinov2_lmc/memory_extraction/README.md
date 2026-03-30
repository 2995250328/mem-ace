# BSE Memory Extraction

Bilateral Supervoxel Extraction (BSE) enhanced memory extraction pipeline for ACE DINOv2 LMC.

## Overview

This module extracts compressed scene memory from multi-view RGB-D data using:
- **BSE Pooling**: Boundary-preserving bilateral clustering with Otsu adaptive thresholding
- **Welford Normalization**: Numerically stable global metric normalization
- **Multi-Strategy Ray Encoding**: Preserves multi-view geometric information

## Quick Start

### Using the Shell Script (Recommended)

```bash
cd /home/xwh/project/ace_depth

# 7-Scenes chess, 20 views (default)
bash ace_dinov2_lmc/memory_extraction/extract_memory.sh

# Override via environment variables
SCENE_TRAIN=fire_train N_VIEWS=40 bash ace_dinov2_lmc/memory_extraction/extract_memory.sh

# Indoor6 scene
DATASET_TYPE=indoor6 SCENE_TRAIN=scene2a_train N_VIEWS=40 bash ace_dinov2_lmc/memory_extraction/extract_memory.sh

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
| `--dinov2_checkpoint` | `/data/xwh/checkpoints/dinov2_vitl14_pretrain.pth` | DINOv2 weights |
| `--dinov2_intermediate_layers` | None | DINOv2 block indices for multi-scale (default: 8 DPT layers) |
| `--use_patch_based` | False | Patch-based vs bilinear upsampling |

### Depth Processing

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--depth_valid_range` | 0.1 6.0 | Valid depth range in meters (min max) |
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

MapAnything WAI format dataset. Requires `scene_name` parameter.

```bash
# 7-Scenes
python -m ace_dinov2_lmc.memory_extraction.run_memory_extraction \
    /data/xwh/7Scenes \
    output/memory.pt \
    --dataset_loader wai \
    --dataset_type 7scenes \
    --scene_name chess_train \
    --n_memory 20

# Indoor6
python -m ace_dinov2_lmc.memory_extraction.run_memory_extraction \
    /data/xwh/mapanything-dataset/wai_data/indoor6 \
    output/memory.pt \
    --dataset_loader wai \
    --dataset_type indoor6 \
    --scene_name scene2a_train \
    --n_memory 40 \
    --depth_valid_range 0.1 100.0
```

### ACE Loader (New)

ACE CamLocDatasetDINOv2 format. Compatible with ACE training pipeline.

```bash
# ACE format dataset (same as ACE training)
python -m ace_dinov2_lmc.memory_extraction.run_memory_extraction \
    /path/to/ace/dataset/chess \
    output/memory.pt \
    --dataset_loader ace \
    --n_memory 20
```

**Key differences:**
- WAI: Requires `--scene_name`, auto ImageNet normalization
- ACE: Direct scene path, uses dataset's normalization

## Output Directory Structure

All outputs are organized under `ace_dinov2_lmc/04_evaluation/memory_extract/`:

```
memory_extract/
├── chess/
│   ├── 20v_v0.05_bilinear_l2/
│   │   └── 20260328_143000/
│   │       ├── memory_bse.pt              # Main output
│   │       ├── memory_bse.ply             # Point cloud visualization
│   │       ├── extraction_config.json     # Config snapshot
│   │       └── extraction_log.txt         # Run log
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

**Dataset path auto-derivation:**

| DATASET_TYPE | Resolved Path | Example |
|--------------|---------------|---------|
| `7scenes` | `/data/xwh/7Scenes/pgt_7scenes_<scene>` | `chess_train` → `/data/xwh/7Scenes/pgt_7scenes_chess` |
| `indoor6` | `/data/xwh/mapanything-dataset/wai_data/indoor6` | `scene2a_train` → `.../scene2a_train` |
| `custom` | `$DATASET_ROOT/$SCENE_TRAIN` | user-specified |

**Depth range auto-derivation:**

| DATASET_TYPE | DEPTH_MIN | DEPTH_MAX |
|--------------|-----------|-----------|
| `7scenes` | 0.1 | 6.0 |
| `indoor6` | 0.1 | 100.0 |

### BSE Pooling

| Variable | Default | Description |
|----------|---------|-------------|
| `VOXEL_SIZE` | `0.05` | Coarse voxel grid size (meters) |
| `USE_OTSU` | `true` | Use Otsu adaptive thresholding |
| `ENABLE_SOR` | `false` | Enable Statistical Outlier Removal |
| `RAY_POOL_STRATEGY` | `mean` | Strategy: `mean`, `dominant`, `first`, `all` |

### Feature Extraction

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_PATCH_BASED` | `false` | Patch-based (grid) vs bilinear upsampling |
| `USE_L2_NORMALIZATION` | `true` | L2 normalize features |
| `DINOV2_CHECKPOINT` | `/data/xwh/checkpoints/dinov2_vitl14_pretrain.pth` | DINOv2 weights |
| `USE_MODEL` | `mapanything` | Feature extractor: `mapanything` or `dinov2` |
| `MODEL_STR` | `mapanything_store_intermediates_ace` | MapAnything config name |
| `MODEL_CHECKPOINT` | `/data/xwh/checkpoints/facebook_map-anything.pth` | MapAnything checkpoint |

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
# Single scene
DATASET_LOADER=ace DATASET_ROOT=/path/to/ace/scenes SCENE_TRAIN=chess \
    bash ace_dinov2_lmc/memory_extraction/extract_memory.sh

# Batch
for scene in chess fire heads; do
    DATASET_LOADER=ace SCENE_TRAIN=$scene \
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
| OOM | Reduce `N_VIEWS`, increase `VOXEL_SIZE`, use smaller GPU |
| Dataset not found | Check `DATASET_ROOT` and `SCENE_TRAIN`; script prints resolved path |
| `mapanything` import error | Script auto-adds `/home/xwh/project/map-anything` to PYTHONPATH |
| Slow processing | Increase `VOXEL_SIZE`, disable `ENABLE_SOR` |
| Depth all zeros | Check `--depth_valid_range`; Indoor6 uses COLMAP sparse depth |
| ACE loader fails | Ensure dataset follows ACE directory structure |

## Dependencies

- PyTorch >= 2.0
- MapAnything — default feature extractor (multi-scale DINOv2 + AAT)
- DINOv2 (facebookresearch/dinov2) — optional fallback
- Open3D (optional — for visualization)
