# Complete Implementation Guide: BSE-Enhanced Memory Extraction

**Author**: Implementation Guide
**Date**: 2026-03-26
**Purpose**: Step-by-step guide to complete `run_memory_extraction.py`

---

## 1. Overview

**Goal**: Incrementally enhance map-anything memory extraction by replacing voxel pooling with BSE + Welford normalization.

**Input**: Training dataset (7-Scenes, Indoor6, etc.)
**Output**: `.pt` file with normalized coordinates, ray directions, and global statistics

**Key Changes**:
- ✅ BSEPooler: Already implemented
- ✅ WelfordNormalizer: Already implemented
- ⚠️ Main script: Needs integration

---

## 2. Architecture: 7 Core Functions

```
main()
├─> load_dataset_and_model()
├─> select_memory_views()
├─> extract_multiscale_features()
├─> two_pass_processing()
│   ├─> Pass 1: unproject_and_pool()
│   └─> Pass 2: normalize_and_save()
└─> save_memory()
```

---

## 3. Function Specifications

### 3.1 `prepare_batch_input(raw_data, device, mode="memory")`

**Purpose**: Convert dataset raw data to model input format.

**Input**:
- `raw_data`: dict from dataset `__getitem__`
  - Keys: `image`, `depth`, `pose`, `intrinsics`, `rgb_path`, etc.

**Output**:
- `processed`: dict ready for model.infer()
  - Keys: `images`, `depths`, `camera_poses`, `camera_intrinsics`

**Pseudocode**:
```python
def prepare_batch_input(raw_data, device, mode="memory"):
    # 1. Extract image (already normalized by dataset)
    image = raw_data['image'].to(device).unsqueeze(0)  # [1, 3, H, W]

    # 2. Extract depth (if available)
    depth = raw_data.get('depth', None)
    if depth is not None:
        depth = depth.to(device).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

    # 3. Extract pose [4x4 matrix]
    pose = raw_data['pose'].to(device).unsqueeze(0)  # [1, 4, 4]

    # 4. Extract intrinsics [3x3 matrix]
    intrinsics = raw_data['intrinsics'].to(device).unsqueeze(0)  # [1, 3, 3]

    return {
        'images': image,
        'depths': depth,
        'camera_poses': pose,
        'camera_intrinsics': intrinsics
    }
```

**Key Notes**:
- Dataset already applies ImageNet normalization
- Add batch dimension (unsqueeze(0))
- Handle missing depth gracefully

---

### 3.2 `extract_multiscale_features(model, input_views, device)`

**Purpose**: Call model.infer() to extract multi-scale intermediate features.

**Input**:
- `model`: Map-Anything Transformer
- `input_views`: list of processed dicts (from prepare_batch_input)

**Output**:
- `features_dict`: dict mapping view_idx to multi-scale features
  - Structure: `{0: {'features': [layer0, layer1, ..., layerN]}, 1: {...}, ...}`

**Pseudocode**:
```python
def extract_multiscale_features(model, input_views, device, save_path="/tmp/features.pt"):
    # Ensure model stores intermediate features
    if hasattr(model, "model_config"):
        model.model_config.store_info_sharing_intermediate_features = True

    # Run inference
    predictions = model.infer(
        input_views,
        memory_efficient_inference=True,
        use_amp=True,
        amp_dtype="bf16",
        save_filename=save_path,
        ignore_depth_inputs=False,  # Use real depth
        ignore_pose_inputs=False,
        ignore_calibration_inputs=False
    )

    # Load saved features
    features_dict = torch.load(save_path)
    return features_dict
```

**Key Notes**:
- `model.infer()` saves features to disk automatically
- Multi-scale features: typically 4-5 intermediate layers + 1 final layer
- Total channels: ~5120 (varies by model)

---

### 3.3 `process_multiscale_features(feat_list, target_size=None)`

**Purpose**: Align multi-scale features to unified grid and concatenate.

**Input**:
- `feat_list`: list of tensors [B, C_i, H_i, W_i] from different layers

**Output**:
- `features`: concatenated tensor [B, C_total, H_max, W_max]

**Pseudocode**:
```python
def process_multiscale_features(feat_list, target_size=None):
    if target_size is None:
        # Use largest spatial size
        target_size = max([f.shape[-2:] for f in feat_list])

    aligned_features = []
    for feat in feat_list:
        if feat.shape[-2:] != target_size:
            # Bilinear interpolation to target size
            feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
        aligned_features.append(feat)

    # Concatenate along channel dimension
    return torch.cat(aligned_features, dim=1)  # [B, C_total, H, W]
```

**Key Notes**:
- Grid resolution typically 37×37 or 74×74
- Alternative: use AnyUp for learned upsampling to image resolution

---

### 3.4 `unproject_with_real_depth(view_data, features, depth_valid_range=(0.1, 6.0))`

**Purpose**: Unproject features to 3D using real depth.

**Input**:
- `view_data`: dict with `depth`, `pose`, `intrinsics`
- `features`: [1, C, H, W] feature tensor
- `depth_valid_range`: (min, max) valid depth in meters

**Output**:
- `points`: [N, 3] world coordinates
- `ray_dirs`: [N, 3] unit ray directions
- `colors`: [N, 3] RGB colors (if available)
- `features_flat`: [N, C] features

**Pseudocode**:
```python
def unproject_with_real_depth(view_data, features, depth_valid_range=(0.1, 6.0)):
    depth = view_data['depths'][0, 0]  # [H, W]
    pose = view_data['camera_poses'][0]  # [4, 4]
    K = view_data['camera_intrinsics'][0]  # [3, 3]

    H, W = depth.shape
    C = features.shape[1]

    # Resize features to match depth resolution
    if features.shape[-2:] != (H, W):
        features = F.interpolate(features, size=(H, W), mode='bilinear', align_corners=False)

    # Create pixel grid
    u, v = torch.meshgrid(torch.arange(W), torch.arange(H), indexing='xy')
    u = u.to(depth.device).float()
    v = v.to(depth.device).float()

    # Filter valid depth
    valid_mask = (depth > depth_valid_range[0]) & (depth < depth_valid_range[1])
    u_valid = u[valid_mask]
    v_valid = v[valid_mask]
    Z_valid = depth[valid_mask]

    # Unproject to camera coordinates
    # P_cam = Z * K^-1 * [u, v, 1]^T
    K_inv = torch.inverse(K)
    uv1 = torch.stack([u_valid, v_valid, torch.ones_like(u_valid)], dim=0)  # [3, N]
    P_cam = K_inv @ uv1  # [3, N]
    P_cam = P_cam * Z_valid.unsqueeze(0)  # [3, N]

    # Transform to world coordinates
    # P_world = R * P_cam + t
    R = pose[:3, :3]
    t = pose[:3, 3]
    P_world = (R @ P_cam).T + t  # [N, 3]

    # Compute ray directions
    camera_center = t  # [3]
    ray_dirs = P_world - camera_center.unsqueeze(0)  # [N, 3]
    ray_dirs = F.normalize(ray_dirs, dim=1)

    # Extract features at valid pixels
    features_flat = features[0, :, v_valid.long(), u_valid.long()].T  # [N, C]

    # Extract colors (if available)
    if 'images' in view_data:
        rgb = view_data['images'][0]  # [3, H, W]
        colors = rgb[:, v_valid.long(), u_valid.long()].T  # [N, 3]
    else:
        colors = torch.ones(len(P_world), 3, device=P_world.device) * 0.5

    return P_world, ray_dirs, colors, features_flat
```

**Key Notes**:
- Handles sparse depth (only valid pixels)
- Camera center = translation vector t
- Ray direction = normalize(P_world - camera_center)

---

### 3.5 `two_pass_processing(memory_views, features_dict, pooler, welford, temp_dir)`

**Purpose**: Two-pass BSE + Welford normalization.

**Pass 1**: Extract → Unproject → Pool → Accumulate statistics → Save chunks
**Pass 2**: Finalize statistics → Load chunks → Normalize → Concatenate

**Pseudocode**:
```python
def two_pass_processing(memory_views, features_dict, pooler, welford, temp_dir="/dev/shm"):
    import os
    os.makedirs(temp_dir, exist_ok=True)
    chunk_paths = []

    # Pass 1: Extract + pool + accumulate
    print("[Pass 1] Extracting and pooling...")
    for view_idx, view_data in enumerate(tqdm(memory_views)):
        # Get multi-scale features for this view
        feat_list = features_dict[view_idx]['features']
        features = process_multiscale_features(feat_list)

        # Unproject with real depth
        points, ray_dirs, colors, features_flat = unproject_with_real_depth(
            view_data, features
        )

        # BSE pooling
        pooled = pooler.pool(points, features_flat, colors, ray_dirs)

        # Accumulate statistics
        welford.update(pooled['points'])

        # Save temporary chunk
        chunk_path = os.path.join(temp_dir, f"chunk_{view_idx}.pt")
        torch.save(pooled, chunk_path)
        chunk_paths.append(chunk_path)

    # Pass 2: Normalize + assemble
    print("[Pass 2] Normalizing and assembling...")
    mu_scene, sigma_scene = welford.finalize()

    all_points = []
    all_ray_dirs = []
    all_features = []
    all_colors = []

    for chunk_path in tqdm(chunk_paths):
        pooled = torch.load(chunk_path)
        # Normalize points
        pooled['points'] = (pooled['points'] - mu_scene) / sigma_scene
        all_points.append(pooled['points'])
        all_ray_dirs.append(pooled['ray_dirs'])
        all_features.append(pooled['features'])
        all_colors.append(pooled['colors'])

    # Concatenate all chunks
    final_points = torch.cat(all_points, dim=0)
    final_ray_dirs = torch.cat(all_ray_dirs, dim=0)
    final_features = torch.cat(all_features, dim=0)
    final_colors = torch.cat(all_colors, dim=0)

    # Clean up temp files
    for chunk_path in chunk_paths:
        os.remove(chunk_path)

    return final_points, final_ray_dirs, final_features, final_colors, mu_scene, sigma_scene
```

---

### 3.6 `save_memory(output_path, points, ray_dirs, features, colors, mu, sigma)`

**Purpose**: Save memory to .pt file with extended schema.

**Pseudocode**:
```python
def save_memory(output_path, points, ray_dirs, features, colors, mu, sigma):
    memory_dict = {
        'points': points.cpu().float(),           # [N, 3]
        'ray_dirs': ray_dirs.cpu().float(),       # [N, 3]
        'features': features.cpu().half(),        # [N, C] fp16
        'colors': colors.cpu().float(),           # [N, 3]
        'mu': mu.cpu().float(),                   # [3]
        'sigma': sigma                            # scalar
    }
    torch.save(memory_dict, output_path)
    print(f"[Save] Memory saved to {output_path}")
    print(f"[Save] Total points: {len(points)}")
```

---

### 3.7 `main()` - Complete Flow

**Pseudocode**:
```python
def main():
    args = parse_args()
    device = torch.device(args.device)

    # Import dependencies
    from mapanything.models import init_model
    from mapanything.datasets import SevenScenesWAI
    from mapanything.tasks.ace.memory_selection import select_optimal_memory_indices

    # Initialize modules
    welford = WelfordNormalizer()
    pooler = BSEPooler(voxel_size=args.voxel_size, use_otsu=args.use_otsu)

    # Load dataset
    train_dataset = SevenScenesWAI(args.dataset_path, split='train')

    # Select memory views
    memory_indices, _ = select_optimal_memory_indices(train_dataset, args.n_memory)

    # Prepare input views
    memory_views = []
    for idx in memory_indices:
        raw_data = train_dataset[idx]
        processed = prepare_batch_input(raw_data, device)
        memory_views.append(processed)

    # Load model (requires model config - add to args)
    # model = init_model(args.model_str, args.model_config)
    # model.to(device).eval()

    # Extract features
    # features_dict = extract_multiscale_features(model, memory_views, device)

    # Two-pass processing
    # points, ray_dirs, features, colors, mu, sigma = two_pass_processing(
    #     memory_views, features_dict, pooler, welford
    # )

    # Save
    # save_memory(args.output_path, points, ray_dirs, features, colors, mu, sigma)
```

---

## 4. Parameter Mapping: Hydra → argparse

| Original (Hydra cfg) | New (argparse) | Default | Description |
|---------------------|----------------|---------|-------------|
| `cfg.N_MEMORY` | `--n_memory` | 100 | Number of memory views |
| `cfg.model.pretrained` | `--model_checkpoint` | None | Model weights path |
| `cfg.model.model_str` | `--model_str` | Required | Model architecture string |
| `cfg.model.model_config` | `--model_config` | Required | Model config dict/path |
| `cfg.gpu_id` | `--device` | cuda:0 | Device string |
| `cfg.use_patch_based_extraction` | `--use_patch_based` | False | Grid vs pixel extraction |
| `cfg.voxel_size` | `--voxel_size` | 0.05 | BSE voxel size |
| `cfg.use_bse` | `--use_bse` | True | Enable BSE pooling |
| `cfg.use_otsu` | `--use_otsu` | True | Enable Otsu threshold |

**Updated argparse**:
```python
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', type=str)
    parser.add_argument('output_path', type=str)
    parser.add_argument('--n_memory', type=int, default=100)
    parser.add_argument('--model_str', type=str, required=True)
    parser.add_argument('--model_config', type=str, required=True)
    parser.add_argument('--model_checkpoint', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--voxel_size', type=float, default=0.05)
    parser.add_argument('--use_bse', action='store_true', default=True)
    parser.add_argument('--use_otsu', action='store_true', default=True)
    return parser.parse_args()
```

---

## 5. Testing Checkpoints

### 5.1 Dataset Loading
```bash
python -m ace_dinov2_lmc.memory_extraction.run_memory_extraction \
    datasets/7scenes_chess \
    output/test.pt \
    --n_memory 10 \
    --model_str "mapanything_transformer" \
    --model_config "configs/default.yaml"
```

**Expected**:
- ✅ Dataset loads without error
- ✅ Prints: "Train dataset: N samples"

### 5.2 View Selection
**Expected**:
- ✅ Returns exactly `n_memory` indices
- ✅ Indices are within dataset range
- ✅ Prints: "Selected 10 views"

### 5.3 Model Inference
**Expected**:
- ✅ No OOM errors
- ✅ Features saved to temp file
- ✅ Multi-scale features: 4-5 layers

### 5.4 Unprojection
**Expected**:
- ✅ Points in reasonable range (e.g., [-10, 10] for 7-Scenes)
- ✅ Ray directions are unit vectors (norm ≈ 1.0)
- ✅ Valid point count > 0

### 5.5 BSE Pooling
**Expected**:
- ✅ Compression ratio > 95% (output points << input points)
- ✅ No NaN or Inf values
- ✅ Otsu threshold in [0.5, 0.95] range

### 5.6 Welford Normalization
**Expected**:
- ✅ mu_scene ≈ scene center (e.g., [0, 0, 0] for centered scenes)
- ✅ sigma_scene ≈ scene radius (e.g., 2-5 for 7-Scenes)
- ✅ Normalized points mostly in [-3, 3] range

### 5.7 Output File
**Expected**:
- ✅ File size reasonable (e.g., 10-50 MB for 100 views)
- ✅ Can be loaded with `torch.load()`
- ✅ Contains all required keys: points, ray_dirs, features, colors, mu, sigma

---

## 6. Common Issues & Debugging

### Issue 1: OOM during model inference
**Solution**: Reduce batch size or use `memory_efficient_inference=True`

### Issue 2: Sparse depth has too few valid points
**Solution**: Adjust `depth_valid_range` or use `patch_depth_sampling='nearest_valid'`

### Issue 3: Otsu threshold always returns 0.90
**Cause**: Unimodal distribution (std < 0.02)
**Solution**: This is expected behavior - fallback to fixed threshold

### Issue 4: Normalized points have extreme values
**Cause**: Outliers in point cloud
**Solution**: Add SOR (Statistical Outlier Removal) before pooling

### Issue 5: Features dimension mismatch
**Cause**: Model config doesn't match expected architecture
**Solution**: Verify `store_info_sharing_intermediate_features=True`

---

## 7. Visualization for Debugging

```python
# Visualize unprojected point cloud
import open3d as o3d
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points.cpu().numpy())
pcd.colors = o3d.utility.Vector3dVector(colors.cpu().numpy())
o3d.io.write_point_cloud("debug_unproject.ply", pcd)

# Check Otsu histogram
import matplotlib.pyplot as plt
plt.hist(sim.cpu().numpy(), bins=256)
plt.axvline(tau, color='r', label=f'Otsu τ={tau:.3f}')
plt.legend()
plt.savefig("debug_otsu.png")
```

---

## 8. Next Steps After Implementation

1. **Unit test each function** with synthetic data
2. **Run on small dataset** (n_memory=10) first
3. **Visualize intermediate results** (point clouds, histograms)
4. **Compare with original** map-anything output
5. **Measure compression ratio** and boundary preservation
6. **Integrate with downstream** ACE training

---

## 9. Summary

**What you need to implement**:
1. ✅ Copy function pseudocode from sections 3.1-3.7
2. ✅ Add model loading logic (init_model + checkpoint)
3. ✅ Handle dataset type detection (7-Scenes vs Indoor6)
4. ✅ Add error handling and logging
5. ✅ Test on real data

**Estimated implementation time**: 2-4 hours

**Files to modify**:
- `run_memory_extraction.py` (main implementation)
- `extract_memory.sh` (update with new args)

Good luck! 🚀

