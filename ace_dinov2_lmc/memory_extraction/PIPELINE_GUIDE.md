# BSE Memory Extraction Pipeline Guide

This document provides a comprehensive explanation of the BSE (Bilateral Supervoxel Extraction) Memory Extraction pipeline for ACE DINOv2 LMC.

## Overview

The pipeline extracts a compressed scene representation from multi-view RGB-D data through:

1. **View Selection**: Selecting optimal memory views from training data
2. **Feature Extraction**: Extracting MapAnything multi-scale features from RGB images (default)
3. **Unprojection**: Converting 2D pixels + depth to 3D points with ray directions
4. **BSE Pooling**: Boundary-preserving bilateral clustering to compress point cloud
5. **Normalization**: Global scene normalization using Welford's algorithm

The output is a `.pt` file containing:
- Compressed 3D points with features and colors
- Multiple ray direction representations
- View-level camera information

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              INPUT DATASET                                  │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ Training Data: RGB images + Depth maps + Poses + Intrinsics         │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           STEP 1: VIEW SELECTION                            │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ Input: All training frames                                           │   │
│  │ Method: Uniform sampling or FPS (Furthest Point Sampling)            │   │
│  │ Output: M memory views (default: 100)                                │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        STEP 2: FEATURE EXTRACTION                           │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ Input: M RGB images                                                  │   │
│  │ Model: DINOv2 ViT-L/14 (pretrained) → MapAnything multi-scale concat │   │
│  │ Output: M feature maps [~5000-9000, H/14, W/14] (multi-scale concat) │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            STEP 3: UNPROJECTION                             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ For each memory view:                                                 │   │
│  │   1. Get valid depth pixels (within range)                            │   │
│  │   2. Back-project: P_cam = Z * K^-1 * [u, v, 1]^T                    │   │
│  │   3. World transform: P_world = R * P_cam + t                         │   │
│  │   4. Ray direction: D = normalize(P_world - camera_center)           │   │
│  │   5. Extract features and colors at valid pixels                     │   │
│  │                                                                      │   │
│  │ Output per view:                                                      │   │
│  │   - points: [N_i, 3] 3D coordinates                                   │   │
│  │   - features: [N_i, C] multi-scale DINOv2 features (C ~ 5000-9000)  │   │
│  │   - colors: [N_i, 3] RGB colors                                      │   │
│  │   - ray_dirs: [N_i, 3] unit ray directions                           │   │
│  │   - camera_centers: [N_i, 3] camera center (repeated)                │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           STEP 4: BSE POOLING                                │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ Applied per view, then merged across views:                          │   │
│  │                                                                      │   │
│  │ 4.1 Coarse Voxel Hashing                                             │   │
│  │     - Quantize points to voxel grid: voxel_id = floor(p / voxel_size) │  │
│  │     - Encode using prime numbers for uniqueness                       │   │
│  │     - Group points by voxel_id                                       │   │
│  │                                                                      │   │
│  │ 4.2 Compute Cluster Mean Features                                    │   │
│  │     - For each voxel, compute mean feature vector                    │   │
│  │     - This represents the "prototype" for the cluster                │   │
│  │                                                                      │   │
│  │ 4.3 Otsu Adaptive Thresholding                                       │   │
│  │     - Compute cosine similarity between each point and cluster mean  │   │
│  │     - Build histogram of similarities                                │   │
│  │     - Apply Otsu's method to find optimal threshold                  │   │
│  │     - If cluster is unimodal (std < threshold), skip splitting      │   │
│  │                                                                      │   │
│  │ 4.4 Binary Split                                                    │   │
│  │     - Main cluster: points with similarity >= threshold             │   │
│  │     - Outlier cluster: points with similarity < threshold           │   │
│  │     - This preserves boundary points separately                     │   │
│  │                                                                      │   │
│  │ 4.5 Fine-grained Pooling                                            │   │
│  │     - For each cluster (main + outlier):                             │   │
│  │       * points: mean position                                       │   │
│  │       * features: mean features                                     │   │
│  │       * colors: mean color                                          │   │
│  │       * ray_dirs: multiple strategies (see below)                   │   │
│  │                                                                      │   │
│  │ Output per view:                                                      │   │
│  │   - M_j pooled clusters (M_j << N_i)                                 │   │
│  │   - Each cluster has: point, feature, color, ray representations     │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      STEP 5: TWO-PASS PROCESSING                            │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ Pass 1: Extract + Pool + Accumulate                                   │   │
│  │   - Process each memory view through steps 2-4                       │   │
│  │   - Collect pooled clusters from all views                           │   │
│  │   - Accumulate global statistics (Welford: mean, variance)          │   │
│  │   - Save temporary chunks to /dev/shm                                │   │
│  │   - Collect view-level camera info (pose, intrinsics, Plücker)      │   │
│  │                                                                      │   │
│  │ Pass 2: Normalize + Assemble                                          │   │
│  │   - Finalize Welford statistics                                      │   │
│  │   - Normalize all points: (p - mu) / sigma                          │   │
│  │   - Concatenate all clusters into final memory                       │   │
│  │   - Assemble view-level camera information                           │   │
│  │                                                                      │   │
│  │ Output:                                                               │   │
│  │   - result: Dict with all pooled tensors                             │   │
│  │   - view_info: Dict with view-level camera info                      │   │
│  │   - mu, sigma: Scene normalization parameters                        │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                             OUTPUT (.pt FILE)                                │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ Schema Version: 1.2                                                  │   │
│  │                                                                      │   │
│  │ Point-level Data (N = total clusters):                               │   │
│  │   - points: [N, 3] normalized 3D coordinates                         │   │
│  │   - features: [N, C] multi-scale DINOv2 features (fp16, C ~ 5000-9000) │   │
│  │   - colors: [N, 3] RGB colors                                        │   │
│  │   - cluster_sizes: [N] points per cluster                            │   │
│  │   - ray_dirs: [N, 3] primary ray directions                          │   │
│  │   - ray_dirs_mean: [N, 3] mean + normalize strategy                  │   │
│  │   - ray_dirs_dominant: [N, 3] highest similarity strategy            │   │
│  │   - ray_dirs_first: [N, 3] first ray strategy                        │   │
│  │   - plucker_rays: [N, 6] Plücker coordinates                         │   │
│  │                                                                      │   │
│  │ View-level Camera Info (M = number of views):                        │   │
│  │   - view_camera_centers: [M, 3] camera positions                     │   │
│  │   - view_camera_rotations: [M, 3, 3] rotation matrices               │   │
│  │   - view_camera_intrinsics: [M, 3, 3] intrinsic matrices             │   │
│  │   - view_plucker_main_rays: [M, 6] main ray Plücker coords           │   │
│  │                                                                      │   │
│  │ Normalization:                                                        │   │
│  │   - mu: [3] scene mean                                               │   │
│  │   - sigma: scene standard deviation                                   │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Detailed Algorithm Explanations

### 1. View Selection

**Purpose**: Select a representative subset of frames from training data.

**Methods**:
- **Uniform Sampling**: Select every k-th frame (fallback)
- **FPS (Furthest Point Sampling)**: Select views that maximize spatial coverage

**Output**: M memory views (default: 100)

### 2. Feature Extraction

**Purpose**: Extract multi-scale semantic features from RGB images.

**Default Model**: MapAnything (DINOv2 ViT-L/14 backbone + AAT 24 layers + DPT prediction head)
- Unified pretrained model on large-scale data
- Outputs 8 intermediate transformer layers (DPT-style) + final layer
- Uses layers [2, 5, 8, 11, 14, 17, 20, 23] + final layer
- Each layer outputs 1024-dimensional features
- **Multi-scale features are concatenated** → total dimension is ~5000-9000 (not 1024)

<!-- Updated 2026-03-30: corrected feature dimension from 1024 to multi-scale concat -->

**Fallback Model**: DINOv2 standalone
- Automatically falls back when MapAnything unavailable
- Uses same DPT-style multi-scale layers
- Force via `USE_MODEL=dinov2` environment variable

**Process**:
```python
# Default: use MapAnything
if config.use_model == "mapanything":
    from mapanything.models import init_model
    # Manually build config (bypass Hydra)
    model_config = build_manual_config()
    model = init_model("mapanything", model_config)

    # Inference
    model.infer(input_views, save_filename=temp_file)

    # Load intermediate features
    saved_data = torch.load(temp_file)
    # Use 8 intermediate layers + final layer
    target_layers = [2, 5, 8, 11, 14, 17, 20, 23]
    feat_list = [saved_data["intermediate"][i] for i in target_layers]
    feat_list.append(saved_data["final"])

else:  # DINOv2 fallback
    # Load DINOv2 model
    dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
    # Extract features directly from intermediate blocks
    feat_list = extract_dinov2_intermediate_features(dinov2, image)
```

**Multi-scale feature processing** (DPT-style):
```python
# 1. Reshape all features to spatial format (if needed)
# 2. Find maximum resolution among all layers
# 3. Upsample smaller features to max resolution using bilinear interpolation
# 4. Concatenate along channel dimension
# 5. Flatten to (grid_H * grid_W, C_total)

features_flat, grid_H, grid_W = process_multiscale_features_to_grid(feat_list)
# Output: features_flat [grid_H * grid_W, ~5000], grid_H, grid_W (typically 37x37)
```

### 3. Unprojection (2D Grid to 3D)

**Purpose**: Convert grid-resolution features to 3D points using real depth.

**Key Difference**: Instead of pixel-wise unprojection, we sample depth at **grid centers**.

**Mathematical Formulation**:

Given:
- Depth map: Z[u, v] at image resolution (e.g., 518x518)
- Camera intrinsics: K (3x3 matrix)
- Camera pose: T = [R | t] (4x4 matrix)
- Grid dimensions: grid_H × grid_W (e.g., 37×37)

**Step-by-step**:

1. **Create grid center coordinates**:
   ```
   u_grid = linspace(0, img_W - 1, grid_W)
   v_grid = linspace(0, img_H - 1, grid_H)
   ```

2. **Sample depth at grid centers**:
   ```
   z_sampled = depth[v_grid_int, u_grid_int]
   ```

3. **Filter valid depth**:
   ```
   valid_mask = (z_sampled > min_depth) & (z_sampled < max_depth)
   ```

4. **Back-project to camera coordinates**:
   ```
   x_cam = (u_valid - cx) * z_valid / fx
   y_cam = (v_valid - cy) * z_valid / fy
   P_cam = [x_cam, y_cam, z_valid]
   ```

5. **Transform to world coordinates**:
   ```
   P_world = R * P_cam + t
   ```

6. **Compute ray direction**:
   ```
   camera_center = -R^T * t
   ray_dir = normalize(P_world - camera_center)
   ```

7. **Sample colors and features**:
   - Colors: Bilinear interpolation at grid centers
   - Features: Already at grid resolution, filter to valid points

**Output per view**:
- points: [N_valid, 3] - Valid 3D points (N_valid ≤ grid_H × grid_W)
- features: [N_valid, ~5000] - Multi-scale DINOv2 features
- colors: [N_valid, 3] - RGB colors
- ray_dirs: [N_valid, 3] - Unit ray directions
- camera_centers: [N_valid, 3] - Camera center (repeated)

**Compression**: Grid-based sampling reduces points by ~196x (from 518×518 to 37×37).

### 4. BSE Pooling

**Purpose**: Compress point cloud while preserving boundaries.

#### 4.1 Coarse Voxel Hashing

**Goal**: Group nearby points into voxels.

**Method**:
```python
# Quantize points to voxel grid
voxel_coords = torch.floor(points / voxel_size).long()

# Encode using prime numbers for unique IDs
voxel_id = (voxel_coords[:, 0] * P1 +
             voxel_coords[:, 1] * P2 +
             voxel_coords[:, 2] * P3)

# Group points by voxel_id
unique_ids, inverse_indices = torch.unique(voxel_id, return_inverse=True)
```

#### 4.2 Cluster Mean Features

**Goal**: Compute prototype feature for each cluster.

**Method**:
```python
# For each voxel, compute mean feature
cluster_means = torch.zeros(len(unique_ids), feature_dim)
for i in range(len(unique_ids)):
    mask = (inverse_indices == i)
    cluster_means[i] = features[mask].mean(dim=0)
```

#### 4.3 Otsu Adaptive Thresholding

**Goal**: Find optimal threshold to separate inliers from outliers.

**Method**:
1. Compute cosine similarity between each point and cluster mean:
   ```
   similarity = (features @ cluster_means.T) / (||features|| * ||cluster_means||)
   ```

2. Build histogram of similarities (256 bins)

3. Apply Otsu's method to find threshold that maximizes inter-class variance:
   ```
   For each possible threshold t:
       w0 = weight of pixels below t
       w1 = weight of pixels above t
       mu0 = mean of pixels below t
       mu1 = mean of pixels above t
       variance = w0 * w1 * (mu0 - mu1)^2

   Select t that maximizes variance
   ```

4. Unimodality check: if similarity std < 0.02, skip splitting

#### 4.4 Binary Split

**Goal**: Separate boundary points from main cluster.

**Method**:
```python
# Main cluster: high similarity points
main_mask = similarity >= threshold

# Outlier cluster: low similarity points (boundary)
outlier_mask = similarity < threshold

# This preserves geometric boundaries separately
```

#### 4.5 Fine-grained Pooling

**Goal**: Compute pooled attributes for each cluster.

**Point-level pooling**:
```python
pooled_point = points[mask].mean(dim=0)
pooled_feature = features[mask].mean(dim=0)
pooled_color = colors[mask].mean(dim=0)
```

**Ray pooling strategies**:

1. **Mean + Normalize**:
   ```python
   ray_mean = ray_dirs[mask].mean(dim=0)
   ray_pooled = F.normalize(ray_mean, dim=-1)
   ```

2. **Dominant Ray**:
   ```python
   # Select ray with highest similarity to cluster mean
   similarities = cosine_similarity(ray_dirs[mask], cluster_mean)
   ray_pooled = ray_dirs[mask][similarities.argmax()]
   ```

3. **First Ray**:
   ```python
   # Simply use the first ray in the cluster
   ray_pooled = ray_dirs[mask][0]
   ```

4. **Plücker Encoding**:
   ```python
   # 6D representation: (direction, moment)
   direction = ray_pooled
   moment = torch.cross(camera_center, direction)
   plucker = torch.cat([direction, moment])  # [6]
   ```

**Compression Ratio**: Typical compression from ~1M points to ~50K clusters (20x).

### 5. Two-Pass Processing

**Purpose**: Efficient memory management with global normalization.

#### Pass 1: Extract + Pool + Accumulate

```python
for view_idx in range(M):
    # Extract features
    features = dinov2(images[view_idx])

    # Unproject to 3D
    points, ray_dirs, colors, features, camera_centers = unproject(...)

    # Optional: SOR filtering
    if enable_sor:
        inliers = sor_filter(points)
        points = points[inliers]

    # BSE pooling
    pooled = bse_pooler.pool(points, features, colors, ray_dirs, camera_centers)

    # Accumulate statistics
    welford.update(pooled['points'])

    # Collect view-level camera info
    view_info['camera_centers'].append(camera_center)
    view_info['camera_rotations'].append(R)
    view_info['camera_intrinsics'].append(K)
    # ... compute Plücker main ray

    # Save temporary chunk
    torch.save(pooled, f"/dev/shm/chunk_{view_idx}.pt")
```

#### Pass 2: Normalize + Assemble

```python
# Finalize global statistics
mu, sigma = welford.finalize()

# Load and normalize all chunks
final_memory = {}
for chunk_path in chunk_paths:
    pooled = torch.load(chunk_path)

    # Normalize points to zero-mean, unit-variance
    pooled['points'] = (pooled['points'] - mu) / sigma

    # Concatenate to final memory
    final_memory['points'].append(pooled['points'])
    # ... same for other attributes

# Concatenate all chunks
result = {
    'points': torch.cat(final_memory['points'], dim=0),
    'features': torch.cat(final_memory['features'], dim=0),
    # ...
}

# Assemble view-level camera info
view_info = {
    'camera_centers': torch.stack(view_camera_centers, dim=0),  # [M, 3]
    'camera_rotations': torch.stack(view_camera_rotations, dim=0),  # [M, 3, 3]
    'camera_intrinsics': torch.stack(view_camera_intrinsics, dim=0),  # [M, 3, 3]
    'plucker_main_rays': torch.stack(view_plucker_main_rays, dim=0),  # [M, 6]
}
```

### 6. Welford's Algorithm

**Purpose**: Numerically stable computation of mean and variance.

**Why Welford?**:
- Traditional: `variance = E[X^2] - E[X]^2` suffers from catastrophic cancellation
- Welford: Updates mean and variance in a single pass, numerically stable

**Algorithm**:
```python
# Initialize
count = 0
mean = 0.0
M2 = 0.0  # Sum of squared differences

# Update for each new point x
count += 1
delta = x - mean
mean += delta / count
delta2 = x - mean
M2 += delta * delta2

# Finalize
variance = M2 / count
std = sqrt(variance)
```

---

## Ray Direction Handling

### Why Multiple Strategies?

When observing an object from multiple viewpoints, ray directions can vary significantly:
- A single point on a table might be observed from different angles
- Simple averaging (`mean + normalize`) can blur this multi-view information

### Available Strategies

| Strategy | Formula | Use Case | Size |
|----------|---------|----------|------|
| `mean` | `normalize(mean(rays))` | Baseline pooling | [N, 3] |
| `dominant` | `argmax(similarity)` | Most representative ray | [N, 3] |
| `first` | `rays[0]` | Preserve original direction | [N, 3] |
| `all` | Keep all rays | Downstream learning | Variable |

### Plücker Ray Encoding

**Definition**: A 6D representation of 3D lines.

```
L = (d, m)
where:
    d ∈ R^3 is the direction vector
    m ∈ R^3 is the moment vector: m = camera_center × d
```

**Properties**:
- Uniquely identifies a 3D line
- Satisfies Grassmann-Plücker constraint: d · m = 0
- Useful for geometric operations (line-line distance, intersection tests)

**In our pipeline**:
- Point-level: `plucker_rays` [N, 6] - pooled Plücker coordinates
- View-level: `view_plucker_main_rays` [M, 6] - main camera ray Plücker coordinates

---

## Usage Example

```bash
python -m ace_dinov2_lmc.memory_extraction.run_memory_extraction \
    /home/xwh/data/7Scenes/pgt_7scenes_chess \
    output/chess_memory.pt \
    --n_memory 100 \
    --voxel_size 0.05 \
    --use_otsu \
    --ray_pool_strategy mean \
    --device cuda:0
```

**Expected output**:
```
[BSE Memory] Dataset: /home/xwh/data/7Scenes/pgt_7scenes_chess
[BSE Memory] Output: output/chess_memory.pt
[BSE Memory] N_MEMORY: 100, BSE: True
[BSE Memory] Voxel size: 0.05, Otsu: True
[BSE Memory] Train dataset: 1000 samples
[BSE Memory] Selected 100 views
[BSE Memory] Using MapAnything extractor (default)
[BSE Memory] Extracting features...
[BSE Memory] Two-pass processing...
[Pass 1] Extracting and pooling...
100%|████████████████████| 100/100
[Pass 2] Normalizing and assembling...
[Save] Memory saved to output/chess_memory.pt
[Save] Schema version: 1.2
[Save] Total points: 52341
[Save] Feature dim: ~5000-9000 (multi-scale concat, not 1024)
[Save] Scene mean: [1.234, -0.567, 2.345]
[Save] Scene sigma: 1.2345
[Save] View count: 100
[BSE Memory] Done!
```

---

## Key Design Decisions

### 1. Why BSE instead of Voxel Pooling?

| Aspect | Voxel Pooling | BSE |
|--------|---------------|-----|
| Boundary preservation | ❌ Merges boundaries | ✅ Separates boundaries |
| Adaptive threshold | ❌ Fixed voxels | ✅ Otsu adaptive |
| Compression ratio | Fixed | Adaptive |
| Implementation | Simple | Moderate |

### 2. Why Two-Pass Processing?

| Alternative | Pros | Cons |
|-------------|------|------|
| Single pass | Simpler | Cannot normalize without full statistics |
| Two-pass | Global normalization, resume capability | Slightly more complex |

### 3. Why Multiple Ray Strategies?

- **Mean**: Baseline, simple
- **Dominant**: Preserves most representative direction
- **First**: No processing overhead
- **All**: Maximum information for downstream learning

### 4. Why Save View-Level Camera Info?

- Downstream Transformer can use camera poses for view-dependent reasoning
- Enables geometric consistency checks
- Supports future extensions (e.g., view selection, attention mechanisms)

---

## Troubleshooting

### Out of Memory

```bash
# Reduce number of memory views
--n_memory 50

# Increase voxel size (coarser pooling)
--voxel_size 0.08

# Reduce batch size in feature extraction
# (modify code to process in smaller batches)
```

### Slow Processing

```bash
# Disable SOR filtering
--enable_sor false

# Use larger voxel size
--voxel_size 0.06

# Disable Otsu (use fixed threshold)
--use_otsu false
```

### Poor Quality Results

- **Blurry boundaries**: Decrease `voxel_size` (e.g., 0.03)
- **Too few clusters**: Decrease `voxel_size`
- **Too many clusters**: Increase `voxel_size`
- **Noisy features**: Enable SOR filtering `--enable_sor`

---

## References

1. **ACE: Accelerated Coordinate Encoding** (CVPR 2023)
   - Original scene coordinate regression framework

2. **DINOv2: Learning Robust Visual Features without Supervision** (2023)
   - Feature extraction backbone

3. **Otsu's Method** (1979)
   - Adaptive thresholding for image binarization

4. **Welford's Algorithm** (1962)
   - Numerically stable online variance computation

5. **Plücker Coordinates** (19th century)
   - 6D representation of 3D lines
