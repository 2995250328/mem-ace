# Implementation Notes: BSE Memory Extraction

## Status: Complete

The BSE Memory Extraction pipeline is fully implemented with multi-strategy ray direction handling.

---

## Architecture Overview

```
run_memory_extraction.py
├── main()
│   ├── parse_args()                 # CLI argument parsing
│   ├── MapAnythingExtractor         # MapAnything model with multi-scale features
│   ├── select_memory_views()        # Memory view selection
│   ├── two_pass_processing()        # Core extraction loop
│   │   ├── Pass 1: Extract + pool + accumulate
│   │   └── Pass 2: Normalize + save
│   └── save_memory()                # Output .pt file
├── BSEPooler (bse_pooling.py)
│   ├── _voxel_hash()                # Prime encoding
│   ├── _compute_cluster_means()     # Coarse features
│   ├── _otsu_threshold()            # Adaptive threshold
│   ├── _binary_split()              # Outlier separation
│   └── pool()                       # Main pooling
└── WelfordNormalizer (welford_meter.py)
    ├── update()                     # Streaming statistics
    └── finalize()                   # Return mean/std
```

---

## Core Functions

### 1. `MapAnythingExtractor.extract()`

**Purpose**: Extract multi-scale features using MapAnything model.

**Implementation**:
```python
class MapAnythingExtractor:
    TARGET_INTERM_LAYERS = [0, 6, 12, 18]  # DPT-style layers

    def extract(self, images, depths, poses, intrinsics):
        # Prepare input views (mapanything format)
        input_views = [...]

        # Run inference with intermediate feature saving
        self.model.infer(input_views, save_filename=temp_file)

        # Load saved features
        saved_data = torch.load(temp_file)
        raw_interm = saved_data["intermediate"]  # 24 layers
        raw_final = saved_data["final"]

        # Extract target layers + final
        result = {}
        for i in range(n_views):
            feat_list = []
            for layer_idx in self.TARGET_INTERM_LAYERS:
                feat_list.append(raw_interm[layer_idx]["features"][i])
            feat_list.append(raw_final["features"][i])
            result[i] = {'features': feat_list, 'grid_H': H, 'grid_W': W}

        return result
```

### 2. `process_multiscale_features_to_grid(feat_list)`

**Purpose**: Align multi-scale features to grid resolution (DPT-style).

**Implementation**:
```python
def process_multiscale_features_to_grid(feat_list, apply_l2_norm=False):
    # 1. Reshape all features to (B, C, H, W) format
    # 2. Find max resolution among all layers
    # 3. Upsample smaller features to max resolution
    # 4. Concatenate along channel dimension
    # 5. Flatten to (grid_H * grid_W, C_total)

    return aligned_features, max_H, max_W
    # Output: [N, ~5000] features (5 layers × 1024)
```

### 3. `two_pass_processing(memory_views, features_dict, ...)`

**Purpose**: Two-pass processing with chunked memory.

**Pass 1** (Extract + Pool + Accumulate):
```python
for view_idx, view_data in enumerate(memory_views):
    # Get multi-scale features for this view
    view_info = features_dict[view_idx]
    feat_list = view_info['features']
    grid_H, grid_W = view_info['grid_H'], view_info['grid_W']

    # Process features: align to grid and concatenate
    features_flat, _, _ = process_multiscale_features_to_grid(feat_list)

    # Unproject with grid features (samples depth at grid centers)
    points, ray_dirs, colors, features_flat, camera_centers = unproject_with_grid_features(
        view_data, features_flat, grid_H, grid_W, depth_valid_range
    )

    # Optional: SOR filtering
    if enable_sor:
        inlier_mask = sor_filter(points)
        points = points[inlier_mask]
        # ... filter other arrays

    # BSE pooling
    pooled = bse_pooler.pool(points, features_flat, colors, ray_dirs, camera_centers)

    # Accumulate statistics
    welford.update(pooled['points'])

    # Collect view-level camera info
    view_camera_centers.append(camera_center)
    view_camera_rotations.append(R)
    view_camera_intrinsics.append(K)
    # ... compute Plücker main ray
```

**Pass 2** (Normalize + Save):
```python
mu, sigma = welford.finalize()

for chunk_path in chunk_paths:
    pooled = torch.load(chunk_path)
    pooled['points'] = (pooled['points'] - mu) / sigma
    # ... concatenate to final memory

# Assemble view-level camera info
view_info = {
    'camera_centers': torch.stack(view_camera_centers, dim=0),
    'camera_rotations': torch.stack(view_camera_rotations, dim=0),
    'camera_intrinsics': torch.stack(view_camera_intrinsics, dim=0),
    'plucker_main_rays': torch.stack(view_plucker_main_rays, dim=0)
}
```

### 4. `BSEPooler.pool(points, features, colors, ray_dirs, camera_centers)`

**Purpose**: Boundary-preserving bilateral clustering.

**Algorithm**:
1. Coarse voxel hashing → voxel indices
2. Compute cluster mean features
3. Otsu thresholding on feature similarities
4. Binary split: separate outliers from main cluster
5. Fine-grained pooling with multiple ray strategies

**Output**:
```python
{
    'points': P_bse,              # [M, 3] float32
    'features': F_bse,            # [M, C] float16 (~5000 dim)
    'colors': C_bse,              # [M, 3] float32
    'ray_dirs': D_pooled,         # [M, 3] based on strategy
    'ray_dirs_mean': D_mean,
    'ray_dirs_dominant': D_dominant,
    'ray_dirs_first': D_first,
    'plucker_rays': plucker,      # [M, 6]
    'camera_centers': centers,    # [M, 3]
    'cluster_sizes': sizes,       # [M] int64
}
```

for chunk_path in chunk_paths:
    pooled = torch.load(chunk_path)

    # Normalize points
    points_norm = (pooled['points'] - mu) / sigma

    final_points.append(points_norm)
    final_ray_dirs.append(pooled['ray_dirs'])
    final_features.append(pooled['features'])
    # ...

    torch.save(final_memory, args.output_path)
```

### 3. `BSEPooler.pool(points, features, colors, ray_dirs, camera_centers)`

**Purpose**: Boundary-preserving bilateral clustering.

**Algorithm**:
1. Coarse voxel hashing → voxel indices
2. Compute cluster mean features
3. Otsu thresholding on feature similarities
4. Binary split: separate outliers from main cluster
5. Fine-grained pooling with multiple ray strategies

**Output**:
```python
{
    'points': P_bse,              # [M, 3] float32
    'features': F_bse,            # [M, C] float16
    'colors': C_bse,              # [M, 3] float32
    'ray_dirs': D_pooled,         # [M, 3] based on strategy
    'ray_dirs_mean': D_mean,
    'ray_dirs_dominant': D_dominant,
    'ray_dirs_first': D_first,
    'plucker_rays': plucker,      # [M, 6]
    'camera_centers': centers,    # [M, 3]
    'cluster_sizes': sizes,       # [M] int64
}
```

---

## Ray Direction Handling

### Problem

Multi-view rays observing the same object have divergent directions. Simple averaging loses geometric information.

### Solution: Multiple Strategies

| Strategy | Formula | Use Case |
|----------|---------|----------|
| `mean` | `normalize(mean(rays))` | Baseline pooling |
| `dominant` | `argmax(similarity)` | Most representative |
| `first` | `rays[0]` | Preserve original |
| `all` | Keep all | Downstream learning |

### Plücker Encoding

```python
def compute_plucker_rays(points, ray_dirs, camera_centers):
    moment = torch.cross(camera_centers, ray_dirs, dim=1)
    plucker = torch.cat([ray_dirs, moment], dim=1)  # [N, 6]
    return plucker
```

Output contains all strategies for downstream experimentation.

---

## Output Schema (v1.2)

```python
{
    # Schema
    'schema_version': '1.2',

    # Point-level data
    'points': P_norm,           # [N, 3] float32, zero-mean unit-std
    'features': F_bse,          # [N, C] float16, multi-scale DINOv2 features (~5000 dim)
    'colors': C_bse,            # [N, 3] float32, RGB [0,1]
    'cluster_sizes': sizes,     # [N] points per cluster

    # Ray representations (point-level, pooled)
    'ray_dirs': D_strategy,     # [N, 3] selected by --ray_pool_strategy
    'ray_dirs_mean': D_mean,
    'ray_dirs_dominant': D_dom,
    'ray_dirs_first': D_first,
    'plucker_rays': plucker,    # [N, 6] (direction, moment)

    # View-level camera information (M = number of memory views)
    'view_camera_centers': centers,    # [M, 3] camera positions
    'view_camera_rotations': rots,     # [M, 3, 3] rotation matrices
    'view_camera_intrinsics': Ks,      # [M, 3, 3] intrinsic matrices
    'view_plucker_main_rays': plucker, # [M, 6] (direction, moment)

    # Normalization
    'mu': mu_scene,             # [3]
    'sigma': sigma_scene,       # scalar
}
```

---

## Testing

### Unit Test

```python
from ace_dinov2_lmc.memory_extraction import BSEPooler, WelfordNormalizer

# Test BSEPooler
pooler = BSEPooler(voxel_size=0.05, use_otsu=True, ray_pool_strategy='mean')
pooled = pooler.pool(points, features, colors, ray_dirs, camera_centers)

# Test WelfordNormalizer
welford = WelfordNormalizer()
welford.update(points_chunk)
mu, sigma = welford.finalize()
```

### Smoke Test

```bash
conda activate mapanything

python -m ace_dinov2_lmc.memory_extraction.run_memory_extraction \
    /data/xwh/7Scenes/pgt_7scenes_chess \
    output/test_memory.pt \
    --n_memory 5 \
    --device cuda:0
```

---

## File Structure

```
memory_extraction/
├── bse_pooling.py              # BSE + ray strategies
├── welford_meter.py            # Streaming normalization
├── run_memory_extraction.py    # Main script
├── extract_memory.sh           # Shell launcher
├── README.md                   # Usage guide
├── .research_state.md          # Research workflow state
├── 00_ideas/
│   └── reuse_map.md            # Reusable code index
├── 01_design/
│   ├── proposal.md             # Original proposal
│   └── proposal_zh.md
├── 02_architecture/
│   ├── system_design_incremental.md  # Final design
│   └── system_design_incremental_zh.md
└── 03_implementation/
    ├── IMPLEMENTATION_NOTES.md         # This file
    └── IMPLEMENTATION_NOTES_zh.md
```

---

## Key Implementation Details

### Adapter Pattern

Decoupled from map-anything:

```python
class FeatureExtractorAdapter:
    def extract(self, images):
        # Standalone DINOv2, no map-anything dependency
        with torch.no_grad():
            return self.dinov2(images)

class DatasetAdapter:
    def __init__(self, path, dataset_type):
        if dataset_type == "7scenes":
            self.dataset = SevenScenesDataset(path, split='train')
        # ...
```

### GPU Memory Cleanup

```python
try:
    # Process views
    for idx in memory_indices:
        # ... extraction code ...
finally:
    # Cleanup temp files
    for chunk_path in chunk_paths:
        if os.path.exists(chunk_path):
            os.remove(chunk_path)
```

### Input Validation

```python
assert depth.dim() == 2, f"Depth must be 2D, got shape {depth.shape}"
assert pose.shape == (4, 4), f"Pose must be 4x4, got {pose.shape}"
assert features.shape[0] == points.shape[0], "Feature/point count mismatch"
```

---

## Next Steps

### Stage 6: Evaluation

- Run on 7-Scenes and Indoor6 datasets
- Compare ray strategies on downstream ACE accuracy
- Ablation: BSE vs vanilla voxel pooling
- Metrics: compression ratio, boundary preservation, localization accuracy

### Potential Improvements

1. **Learned Ray Pooling**: Attention-based ray weighting
2. **Iterative Clustering**: k-way instead of binary split
3. **Multi-Scale BSE**: Hierarchical pooling
4. **Temporal Consistency**: For video sequences
