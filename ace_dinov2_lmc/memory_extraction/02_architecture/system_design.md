# System Architecture: Bilateral Supervoxel Extraction with Welford Streaming Normalization

**Author**: Senior ML Systems Architect
**Stage**: 3 — Architecture Design
**Date**: 2026-03-26

---

## 1. Architecture Overview

The BSE memory extraction system consists of three main modules:

```
┌─────────────────────────────────────────────────────────────────┐
│                    run_memory_extraction.py                      │
│                     (Main Orchestrator)                          │
└────────────┬────────────────────────────────────────────────────┘
             │
             ├──> FeatureExtractor (DINOv2 wrapper)
             │    - Load pretrained DINOv2-ViT-L/14
             │    - Extract dense patch features [H/14, W/14, 1024]
             │
             ├──> BSEPooler (bse_pooling.py)
             │    - Coarse voxel hashing
             │    - Otsu adaptive thresholding
             │    - Bilateral feature clustering
             │
             └──> WelfordNormalizer (welford_meter.py)
                  - Streaming statistics accumulation (FP64)
                  - Two-pass normalization
```

### Key Design Decisions

1. **Feature Extraction Strategy**: Use DINOv2-ViT-L/14 final layer patch tokens (no multi-scale fusion in v1)
2. **Depth Integration**: Depth Anything V2 used only for unprojection, not for feature weighting
3. **Memory Format**: Compatible with existing `dino_lmc_base` trainer buffer schema
4. **Modularity**: Each component (FeatureExtractor, BSEPooler, WelfordNormalizer) is independently testable

---

## 2. Module Specifications

### 2.1 FeatureExtractor

**Purpose**: Wrap DINOv2 model for efficient batch feature extraction.

**Interface**:
```python
class FeatureExtractor:
    def __init__(self, dinov2_path: str, device: str, freeze: bool = True)
    def extract(self, images: Tensor) -> Tensor
        # Input: [B, 3, H, W], RGB normalized with ImageNet stats
        # Output: [B, H/14, W/14, 1024], patch features
```

**Implementation Notes**:
- Reuse DINOv2 loading pattern from `trainer_dinov2_lmc.py` (lines 1-80)
- Use `torch.no_grad()` context since backbone is frozen
- Return patch tokens only (exclude CLS token)
- Output shape: `[B, num_patches_h, num_patches_w, 1024]` where `num_patches = H // 14`

**Reuse from dino_lmc_base**:
- DINOv2 model initialization pattern
- ImageNet normalization constants
- Device management

---

### 2.2 BSEPooler

**Purpose**: Bilateral supervoxel extraction with Otsu adaptive thresholding.

**Interface**:
```python
class BSEPooler:
    def __init__(self, voxel_size: float = 0.05, use_otsu: bool = True,
                 otsu_bins: int = 256, unimodal_threshold: float = 0.02)

    def pool(self, points: Tensor, features: Tensor, colors: Tensor,
             ray_dirs: Tensor) -> Dict[str, Tensor]
        # Input:
        #   points: [N, 3] world coordinates
        #   features: [N, C] DINOv2 features (fp16)
        #   colors: [N, 3] RGB colors
        #   ray_dirs: [N, 3] unit ray directions
        # Output: dict with keys {points, features, colors, ray_dirs}
```

**Core Algorithm**:
```python
def pool(self, points, features, colors, ray_dirs):
    # Step 1: Coarse voxel hashing
    cluster_ids = self._voxel_hash(points)  # [N]

    # Step 2: Compute cluster mean features (fp32 for stability)
    F_mean = self._scatter_mean(features.float(), cluster_ids)  # [M, C]

    # Step 3: Otsu adaptive thresholding
    sim = self._cosine_similarity(features.float(), F_mean[cluster_ids])  # [N]
    tau = self._otsu_threshold(sim) if self.use_otsu else 0.90

    # Step 4: Binary split
    is_outlier = (sim < tau).long()
    sub_ids = cluster_ids * 2 + is_outlier

    # Step 5: Fine-grained pooling
    return self._scatter_mean_all(points, features, colors, ray_dirs, sub_ids)
```

**Implementation Notes**:
- Use `torch.unique(return_inverse=True)` for voxel hashing
- Implement `_scatter_mean` with `index_add_` + count division (no torch_scatter dependency)
- Otsu implementation: vectorized histogram + between-class variance maximization
- Unimodal guard: skip splitting if `std(sim) < unimodal_threshold`

---

### 2.3 WelfordNormalizer

**Purpose**: Streaming global statistics computation with O(1) memory.

**Interface**:
```python
class WelfordNormalizer:
    def __init__(self)
    def update(self, points: Tensor) -> None
        # Accumulate statistics from a chunk of points
    def finalize(self) -> Tuple[Tensor, float]
        # Returns (mu_scene, sigma_scene)
    def normalize(self, points: Tensor) -> Tensor
        # Apply normalization: (points - mu) / sigma
```

**Welford Algorithm** (numerically stable variance):
```python
def update(self, points):
    for p in points:
        self.count += 1
        delta = p - self.mean
        self.mean += delta / self.count  # FP64
        self.M2 += delta * (p - self.mean)  # FP64

def finalize(self):
    mu = self.mean.float()  # FP64 -> FP32
    sigma = torch.sqrt(self.M2 / self.count).item()  # scalar
    return mu, sigma
```

**Implementation Notes**:
- `mean` and `M2` must be `torch.float64` to avoid catastrophic cancellation
- `count` is a Python int (no overflow risk for <10^15 points)
- Output `mu` and `sigma` are converted to FP32 for storage

---

## 3. Main Pipeline: run_memory_extraction.py

### 3.1 Two-Pass Processing Flow

```python
def main(args):
    # Initialize modules
    feature_extractor = FeatureExtractor(args.dinov2_path, args.device)
    bse_pooler = BSEPooler(args.voxel_size, args.use_otsu)
    welford = WelfordNormalizer()
    dataset = CamLocDatasetDINOv2(args.scene, mode=0)  # RGB-only

    # Pass 1: Extract + Pool + Accumulate Statistics
    temp_chunks = []
    for batch_idx, batch in enumerate(dataloader):
        # 1. Extract DINOv2 features
        features = feature_extractor.extract(batch['image'])  # [B, H/14, W/14, 1024]

        # 2. Unproject to 3D
        points, ray_dirs, colors = unproject_batch(batch, features)

        # 3. BSE pooling
        pooled = bse_pooler.pool(points, features, colors, ray_dirs)

        # 4. Update Welford statistics
        welford.update(pooled['points'])

        # 5. Spool to disk
        temp_path = f"/dev/shm/chunk_{batch_idx}.pt"
        torch.save(pooled, temp_path)
        temp_chunks.append(temp_path)

    # Pass 2: Normalize + Assemble
    mu_scene, sigma_scene = welford.finalize()

    final_memory = []
    for temp_path in temp_chunks:
        pooled = torch.load(temp_path)
        pooled['points'] = welford.normalize(pooled['points'])
        final_memory.append(pooled)

    # Concatenate and save
    memory = {k: torch.cat([m[k] for m in final_memory]) for k in final_memory[0]}
    memory['mu'] = mu_scene
    memory['sigma'] = sigma_scene
    torch.save(memory, args.output_path)

    # Cleanup temp files
    for temp_path in temp_chunks:
        os.remove(temp_path)
```

### 3.2 Unprojection Helper

```python
def unproject_batch(batch, features):
    """Unproject pixels to 3D world coordinates with ray directions."""
    depth = batch['depth']  # [B, H, W] from Depth Anything V2
    K = batch['intrinsics']  # [B, 3, 3]
    R = batch['rotation']  # [B, 3, 3]
    t = batch['translation']  # [B, 3]

    # Generate pixel grid
    H, W = depth.shape[1:]
    u, v = torch.meshgrid(torch.arange(W), torch.arange(H), indexing='xy')
    pixels = torch.stack([u, v, torch.ones_like(u)], dim=-1).float()  # [H, W, 3]

    # Unproject to camera coordinates
    P_cam = depth[..., None] * (K.inverse() @ pixels.unsqueeze(-1)).squeeze(-1)  # [B, H, W, 3]

    # Transform to world coordinates
    P_world = (R @ P_cam.unsqueeze(-1)).squeeze(-1) + t.unsqueeze(1).unsqueeze(1)  # [B, H, W, 3]

    # Compute ray directions
    O_cam = t  # [B, 3]
    D_raw = P_world - O_cam.unsqueeze(1).unsqueeze(1)  # [B, H, W, 3]
    D_raw = D_raw / (torch.norm(D_raw, dim=-1, keepdim=True) + 1e-6)  # normalize

    # Downsample to match DINOv2 resolution (14x)
    P_world = F.interpolate(P_world.permute(0,3,1,2), scale_factor=1/14, mode='bilinear').permute(0,2,3,1)
    D_raw = F.interpolate(D_raw.permute(0,3,1,2), scale_factor=1/14, mode='bilinear').permute(0,2,3,1)
    colors = F.interpolate(batch['image'], scale_factor=1/14, mode='bilinear').permute(0,2,3,1)

    # Flatten
    points = P_world.reshape(-1, 3)
    ray_dirs = D_raw.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    features = features.reshape(-1, 1024)

    return points, ray_dirs, colors, features
```

---

## 4. Output Format & Compatibility

### 4.1 Memory File Schema

```python
{
    "points": Tensor,      # [N, 3] float32, normalized coordinates
    "ray_dirs": Tensor,    # [N, 3] float32, unit vectors
    "features": Tensor,    # [N, 1024] float16, DINOv2 features
    "colors": Tensor,      # [N, 3] float32, RGB [0, 1]
    "mu": Tensor,          # [3] float32, scene mean
    "sigma": float,        # scalar float32, scene std
}
```

### 4.2 Compatibility with dino_lmc_base Trainer

The output format is compatible with `TrainerACEDINOv2LMC.load_memory_features()`:

**Mapping**:
- `points` → used for visibility estimation and coordinate regression targets
- `features` → loaded into LMC fusion module
- `ray_dirs` → (new field, ignored by existing trainer but available for future use)
- `mu`, `sigma` → used for denormalization during inference

**Backward Compatibility**:
- If `ray_dirs` is missing, trainer falls back to vanilla mode
- If `mu`/`sigma` are missing, trainer assumes identity normalization

---

## 5. Key Technical Details

### 5.1 DINOv2 Feature Extraction

**Model**: DINOv2-ViT-L/14 (1024-dim features)
**Source**: Pretrained weights at `/data/xwh/checkpoints/dinov2_vitl14_pretrain.pth`
**Strategy**: Single-scale, final layer patch tokens only (no multi-scale fusion in v1)

**Rationale**:
- Multi-scale fusion (like DPT) adds complexity without proven benefit for relocalization
- Final layer features are semantically rich and sufficient for BSE boundary detection
- Keeps architecture simple and ablation-friendly

**Feature Resolution**:
- Input image: 518×518 (37×37 patches at 14×14 patch size)
- Output features: [37, 37, 1024]
- After unprojection: ~1369 points per image (before BSE compression)

### 5.2 Depth Integration Strategy

**Depth Source**: Depth Anything V2 (metric depth estimation)
**Usage**: Unprojection only (not used for feature weighting)

**Rationale**:
- Depth quality varies across scenes; using it for weighting risks amplifying errors
- BSE already performs semantic-aware pooling via feature similarity
- Depth-weighted pooling can be explored in future work (v2)

**Depth Preprocessing**:
- Resize to match image resolution (518×518)
- No confidence filtering in v1 (assume all depth is valid)
- Outlier rejection handled implicitly by BSE (noisy depth → outlier cluster)

### 5.3 Performance Optimization

**Memory Management**:
- Temp files written to `/dev/shm` (RAM disk) for fast I/O
- Each chunk processed independently, GPU memory freed after spooling
- Welford accumulator: O(1) memory (only 3 FP64 tensors + 1 int)

**Batch Processing**:
- Default batch size: 16 images per chunk
- Adjustable via `--batch_size` for different GPU memory budgets
- Expected throughput: ~2-3 minutes for 1000 images on A100

**Numerical Stability**:
- Cosine similarity computed in FP32 (not FP16)
- Welford mean/M2 maintained in FP64
- Epsilon guards: 1e-6 for division by zero

---

## 6. File Organization

```
memory_extraction/
├── __init__.py                    # Empty package marker
├── feature_extractor.py           # DINOv2 wrapper
├── bse_pooling.py                 # BSE core algorithm
├── welford_meter.py               # Streaming normalization
├── run_memory_extraction.py       # Main orchestrator
└── extract_memory.sh              # Bash launcher script
```

### 6.1 Module Dependencies

```
run_memory_extraction.py
├── feature_extractor.py
│   └── torch, torchvision
├── bse_pooling.py
│   └── torch (no external deps)
├── welford_meter.py
│   └── torch (no external deps)
└── dataset_dinov2.py (from project root)
    └── torch, cv2, numpy
```

**No New Dependencies**: All modules use only PyTorch and standard library.

---

## 7. Testing Strategy

### 7.1 Unit Tests

**FeatureExtractor**:
- Input: synthetic [1, 3, 518, 518] tensor
- Expected output shape: [1, 37, 37, 1024]
- Verify: features are not all zeros, std > 0.1

**BSEPooler**:
- Input: 10k random points with synthetic features
- Expected: output size < 500 points (>95% compression)
- Verify: Otsu threshold varies with feature distribution

**WelfordNormalizer**:
- Input: 1M random points in chunks of 10k
- Expected: μ and σ match torch.mean/std to 4 decimal places
- Verify: FP64 precision prevents catastrophic cancellation

### 7.2 Integration Test

**End-to-End on 7-Scenes Chess**:
- Input: 1000 training images
- Expected output: `pooled_memory.pt` with ~50k points
- Verify: P_norm ∈ [-5, 5], compression ratio < 0.05
- Runtime: <5 minutes on A100

---

## 8. Implementation Roadmap

### Sprint 1: Welford Streaming Normalization
**Files**: `welford_meter.py`, `run_memory_extraction.py` (skeleton)
**Acceptance**: Verify μ, σ match full-batch computation to 4 decimal places

### Sprint 2: Otsu Adaptive Thresholding
**Files**: `bse_pooling.py`
**Acceptance**: Log τ_otsu for each chunk, verify it varies with contrast

### Sprint 3: Feature Extraction & Integration
**Files**: `feature_extractor.py`, complete `run_memory_extraction.py`
**Acceptance**: End-to-end test on 7-Scenes Chess, verify output format

### Sprint 4: Evaluation Metrics
**Files**: `eval_boundary_metrics.py` (in 04_evaluation/)
**Acceptance**: FVR > 0.90 for BSE, FVR < 0.50 for uniform baseline

---

## 9. Open Questions & Future Work

**Answered in this design**:
- ✅ Feature extraction: DINOv2-ViT-L/14 final layer only
- ✅ Depth usage: Unprojection only, no weighting
- ✅ Multi-scale: Not in v1 (keep simple)
- ✅ Compatibility: Output format matches dino_lmc_base

**Deferred to v2**:
- Multi-scale feature fusion (DPT-style)
- Depth confidence weighting
- Iterative k-means (k>2) for complex boundaries
- SuperPoint integration (mentioned in proposal but not critical for BSE validation)

