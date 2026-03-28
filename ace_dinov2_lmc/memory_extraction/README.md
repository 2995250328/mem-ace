# BSE Memory Extraction

Bilateral Supervoxel Extraction (BSE) enhanced memory extraction pipeline for ACE DINOv2 LMC.

## Overview

This module extracts compressed scene memory from multi-view RGB-D data using:
- **BSE Pooling**: Boundary-preserving bilateral clustering with Otsu adaptive thresholding
- **Welford Normalization**: Numerically stable global metric normalization
- **Multi-Strategy Ray Encoding**: Preserves multi-view geometric information

## Quick Start

### Basic Usage

```bash
# Activate environment
conda activate mapanything

# Run extraction
python -m ace_dinov2_lmc.memory_extraction.run_memory_extraction \
    /path/to/dataset \
    output/memory.pt \
    --n_memory 100 \
    --device cuda:0
```

### Using the Shell Script

```bash
cd ace_dinov2_lmc/memory_extraction

# Basic extraction
./extract_memory.sh /path/to/dataset output/memory.pt 100 0

# With custom BSE parameters
VOXEL_SIZE=0.03 USE_OTSU=true ./extract_memory.sh /path/to/dataset output/memory.pt 100 0
```

## Parameters

### Core Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dataset_path` | Required | Path to scene dataset |
| `output_path` | Required | Output .pt file path |
| `--n_memory` | 100 | Number of memory views to select |
| `--device` | cuda:0 | GPU device |

### BSE Pooling Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--voxel_size` | 0.05 | Coarse voxel grid size |
| `--use_otsu` | True | Use Otsu adaptive thresholding |
| `--otsu_bins` | 256 | Histogram bins for Otsu |
| `--unimodal_threshold` | 0.02 | Skip split if similarity std < threshold |

### Ray Direction Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--ray_pool_strategy` | mean | Strategy for pooling ray directions |
| `--save_all_ray_strategies` | True | Save all strategy results for comparison |

#### Ray Pooling Strategies

| Strategy | Description |
|----------|-------------|
| `mean` | Average + L2 normalize (default baseline) |
| `dominant` | Select ray with highest feature similarity |
| `first` | Use first ray in cluster (preserves original direction) |
| `all` | Keep all rays per cluster |

### Depth Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--depth_valid_range` | 0.1 6.0 | Min/max valid depth range |
| `--enable_sor` | False | Enable Statistical Outlier Removal |

### Model Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model_str` | None | MapAnything model architecture string |
| `--model_config` | None | Path to model config |
| `--model_checkpoint` | None | Optional model checkpoint path |
| `--dinov2_checkpoint` | /data/xwh/checkpoints/dinov2_vitl14_pretrain.pth | DINOv2 weights (used if no model specified) |

## Output Format

The output `.pt` file contains:

```python
{
    # Schema
    'schema_version': '1.2',

    # Point-level data
    'points': P_norm,                  # [N, 3] float32, normalized
    'features': F_bse,                 # [N, C] float16, multi-scale DINOv2 features (~5000 dim)
    'colors': C_bse,                   # [N, 3] float32, RGB [0,1]
    'cluster_sizes': sizes,            # [N] int64, points per cluster

    # Ray direction representations (point-level, pooled)
    'ray_dirs': D_mean,                # [N, 3] float32, pooled by strategy
    'ray_dirs_mean': D_mean,           # [N, 3] float32, mean + normalize
    'ray_dirs_dominant': D_dominant,   # [N, 3] float32, highest similarity
    'ray_dirs_first': D_first,         # [N, 3] float32, first in cluster
    'plucker_rays': plucker,           # [N, 6] float32, (direction, moment)

    # View-level camera information (M = number of memory views)
    'view_camera_centers': centers,    # [M, 3] float32, camera positions
    'view_camera_rotations': rots,     # [M, 3, 3] float32, rotation matrices
    'view_camera_intrinsics': Ks,      # [M, 3, 3] float32, intrinsic matrices
    'view_plucker_main_rays': plucker, # [M, 6] float32, (direction, moment)

    # Normalization statistics
    'mu': mu_scene,                    # [3] float32, scene mean
    'sigma': sigma_scene,              # scalar float32, scene std
}
```

## Ray Direction Handling

### Why Multiple Strategies?

When observing an object from multiple viewpoints, ray directions can vary significantly. Simple averaging may lose important multi-view information. This module provides multiple strategies:

1. **Mean + Normalize**: Baseline approach, may blur multi-view information
2. **Dominant Ray**: Preserves the most representative ray per cluster
3. **First Ray**: Simple, preserves original direction without processing
4. **Plücker Encoding**: 6D ray representation for downstream processing

### Plücker Ray Encoding

Plücker coordinates provide a unique 6D representation for 3D lines:

```
plucker = (direction, moment)
moment = camera_center × direction
```

This encoding is useful for:
- View-dependent reasoning in downstream Transformer
- Multi-view consistency checks
- Learned compression in subsequent stages

## Architecture

```
memory_extraction/
├── __init__.py           # Module exports
├── bse_pooling.py        # BSE algorithm with ray strategies
├── welford_meter.py      # Streaming normalization
├── run_memory_extraction.py  # Main extraction script
└── extract_memory.sh     # Shell launcher
```

### BSEPooler

Core pooling algorithm:

1. **Coarse Voxel Hash**: Quantize points to voxel grid
2. **Cluster Mean Features**: Compute mean features per voxel
3. **Otsu Thresholding**: Adaptive similarity threshold
4. **Binary Split**: Separate outliers from main cluster
5. **Fine-grained Pooling**: Pool all attributes with multiple ray strategies

### WelfordNormalizer

Streaming statistics computation:

- Numerically stable mean/variance in FP64
- Incremental updates for memory efficiency
- Two-pass normalization: collect stats → normalize

## Examples

### Extract from 7-Scenes

```bash
python -m ace_dinov2_lmc.memory_extraction.run_memory_extraction \
    /data/xwh/7Scenes/pgt_7scenes_chess \
    output/chess_memory.pt \
    --n_memory 100 \
    --device cuda:0
```

### Extract with SOR Filtering

```bash
python -m ace_dinov2_lmc.memory_extraction.run_memory_extraction \
    /path/to/dataset \
    output/memory.pt \
    --n_memory 100 \
    --enable_sor \
    --device cuda:0
```

### Extract with Custom Voxel Size

```bash
python -m ace_dinov2_lmc.memory_extraction.run_memory_extraction \
    /path/to/dataset \
    output/memory.pt \
    --n_memory 100 \
    --voxel_size 0.03 \
    --device cuda:0
```

## Troubleshooting

### Out of Memory

```bash
# Reduce number of memory views
--n_memory 50

# Increase voxel size (coarser pooling)
--voxel_size 0.08
```

### Slow Processing

```bash
# Disable SOR filtering
--enable_sor false

# Use larger voxel size
--voxel_size 0.06
```

### Import Errors

```bash
# Ensure mapanything environment is activated
conda activate mapanything

# Check Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

## Dependencies

- PyTorch >= 2.0
- MapAnything (primary feature extraction model)
- DINOv2 (facebookresearch/dinov2) - fallback if no model specified
- Open3D (optional, for visualization)
