# System Architecture: BSE-Enhanced Map-Anything Memory Extraction

**Author**: Senior ML Systems Architect
**Stage**: 3 — Architecture Design (Incremental Enhancement)
**Date**: 2026-03-26

---

## Design Philosophy

This is an **incremental enhancement** of the existing Map-Anything memory extraction pipeline, NOT a new method. We preserve the proven components (view selection, multi-scale features, real depth) and upgrade ONLY the pooling and normalization steps.

**What stays the same:**
- View selection via `select_optimal_memory_indices` (FPS-based spatial coverage)
- Multi-scale feature extraction via `model.infer()` (Map-Anything Transformer)
- Real depth unprojection (dense and sparse support)
- Dataset loaders, visualization helpers

**What changes:**
- Pooling: Uniform voxel pooling → BSE (Bilateral Supervoxel Extraction) with Otsu adaptive thresholding
- Normalization: None → Welford streaming global normalization (zero-mean, unit-std)
- Output: Add ray directions, global statistics (μ_scene, σ_scene)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│              run_memory_extraction.py (main coordinator)         │
└────────────┬────────────────────────────────────────────────────┘
             │
             ├──> select_optimal_memory_indices (UNCHANGED)
             │    - FPS-based view selection
             │    - N_MEMORY views (e.g., 100)
             │
             ├──> model.infer() (UNCHANGED)
             │    - Map-Anything Transformer
             │    - Multi-scale intermediate features
             │    - Real depth support
             │
             ├──> unproject_with_real_depth (UNCHANGED)
             │    - Plenoptic unprojection
             │    - Sparse depth handling
             │
             ├──> BSEPooler (NEW - replaces voxel_pooling_optimized)
             │    - Coarse voxel hash
             │    - Otsu adaptive thresholding
             │    - Bilateral feature clustering
             │
             └──> WelfordNormalizer (NEW)
                  - Streaming statistics (FP64)
                  - Two-pass normalization
```

---

## Module Specifications

### 1. BSEPooler (NEW)

**Purpose**: Replace `voxel_pooling_optimized` with boundary-preserving bilateral clustering.

**Location**: `memory_extraction/bse_pooling.py`

**Interface**:
```python
class BSEPooler:
    def __init__(self, voxel_size: float = 0.05, use_otsu: bool = True,
                 otsu_bins: int = 256, unimodal_threshold: float = 0.02)

    def pool(self, points: Tensor, features: Tensor, colors: Tensor,
             ray_dirs: Tensor) -> Dict[str, Tensor]
        # Input:
        #   points: [N, 3] world coordinates
        #   features: [N, C] multi-scale features (fp16)
        #   colors: [N, 3] RGB colors
        #   ray_dirs: [N, 3] unit ray directions
        # Output: dict with keys {points, features, colors, ray_dirs}
```

**Algorithm**:
```python
def pool(self, points, features, colors, ray_dirs):
    # Step 1: Coarse voxel hash
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

**Implementation notes**:
- Use `torch.unique(return_inverse=True)` for voxel hash
- Use `index_add_` + count division for `_scatter_mean` (NO torch_scatter dependency)
- Otsu: vectorized histogram + inter-class variance maximization
- Unimodal protection: skip split if `std(sim) < unimodal_threshold`

---

### 2. WelfordNormalizer (NEW)

**Purpose**: O(1) memory streaming global statistics computation.

**Location**: `memory_extraction/welford_meter.py`

**Interface**:
```python
class WelfordNormalizer:
    def __init__(self)
    def update(self, points: Tensor) -> None
        # Accumulate statistics from one chunk of points
    def finalize(self) -> Tuple[Tensor, float]
        # Return (mu_scene, sigma_scene)
    def normalize(self, points: Tensor) -> Tensor
        # Apply normalization: (points - mu) / sigma
```

**Welford algorithm** (numerically stable variance):
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

**Implementation notes**:
- `mean` and `M2` must be `torch.float64` to avoid catastrophic cancellation
- `count` is Python int (no overflow risk for <10^15 points)
- Output `mu` and `sigma` converted to FP32 for storage

---

### 3. Main Pipeline (MODIFIED)

**File**: `memory_extraction/run_memory_extraction.py`

**Changes from original `map-anything/mapanything/tasks/run_memory_extraction.py`**:

| Component | Original | Enhanced |
|-----------|----------|----------|
| Config system | Hydra `@hydra.main` | argparse CLI |
| View selection | `select_optimal_memory_indices` | **UNCHANGED** |
| Model inference | `model.infer()` | **UNCHANGED** |
| Feature extraction | Multi-scale intermediate layers | **UNCHANGED** |
| Depth handling | Real depth unprojection | **UNCHANGED** |
| Pooling | `voxel_pooling_optimized` | `BSEPooler.pool()` |
| Ray direction | Not stored | **NEW**: compute and store |
| Normalization | None | **NEW**: Welford global normalization |
| Output format | `{pooled_points, pooled_features}` | Extended schema (see below) |

**Two-pass processing**:
```python
# Pass 1: Extract + pool + accumulate statistics
welford = WelfordNormalizer()
bse_pooler = BSEPooler(voxel_size=0.05, use_otsu=True)

for batch in dataloader:
    # UNCHANGED: view selection, model.infer(), feature extraction
    memory_indices = select_optimal_memory_indices(train_dataset, N_MEMORY)
    predictions = model.infer(input_views, ...)
    features = process_multiscale_features(predictions)

    # UNCHANGED: unprojection with real depth
    points, ray_dirs, colors = unproject_with_real_depth(batch, features)

    # CHANGED: BSE pooling instead of voxel pooling
    pooled = bse_pooler.pool(points, features, colors, ray_dirs)

    # NEW: accumulate global statistics
    welford.update(pooled['points'])
    torch.save(pooled, f"/dev/shm/chunk_{i}.pt")

# Pass 2: Normalize + assemble
mu_scene, sigma_scene = welford.finalize()
for chunk in temp_chunks:
    pooled = torch.load(chunk)
    pooled['points'] = welford.normalize(pooled['points'])
    final_memory.append(pooled)

# Save with extended schema
torch.save({
    'points': final_points,
    'ray_dirs': final_ray_dirs,
    'features': final_features,
    'colors': final_colors,
    'mu': mu_scene,
    'sigma': sigma_scene,
}, output_path)
```

---

## Output Format

**Extended schema** (backward compatible):
```python
{
    "points":   [N, 3] float32,      # Normalized coordinates (zero-mean, unit-std)
    "ray_dirs": [N, 3] float32,      # Unit ray directions (NEW)
    "features": [N, C] float16,      # Multi-scale features (C ~5120)
    "colors":   [N, 3] float32,      # RGB [0, 1]
    "mu":       [3] float32,         # Scene mean (NEW)
    "sigma":    float32,             # Scene std (NEW)
}
```

**Backward compatibility**: Downstream code expecting `pooled_points`/`pooled_features` keys can add aliases or use a simple adapter.

---

## File Organization

```
memory_extraction/
├── __init__.py
├── bse_pooling.py            # NEW: BSE algorithm
├── welford_meter.py          # NEW: Streaming normalization
├── run_memory_extraction.py  # MODIFIED: Enhanced pipeline
└── extract_memory.sh         # MODIFIED: Adapted bash script
```

**No new dependencies**: All modules use only PyTorch and standard library.

---

## Implementation Roadmap

### Sprint 1: Welford Streaming Normalization
**File**: `welford_meter.py`
- Implement `WelfordNormalizer` class
- Unit test: verify numerically stable variance on synthetic data
- Integration test: run on 7-Scenes Chess, verify `mu` and `sigma` are reasonable

### Sprint 2: Otsu Adaptive Thresholding
**File**: `bse_pooling.py` (partial)
- Implement `_otsu_threshold()` function
- Unit test: verify on synthetic bimodal/unimodal distributions
- Ablation: compare fixed τ=0.90 vs Otsu on real similarity histograms

### Sprint 3: BSE Pooling
**File**: `bse_pooling.py` (complete)
- Implement full `BSEPooler` class
- Replace `voxel_pooling_optimized` in `run_memory_extraction.py`
- Add CLI flags: `--use_bse`, `--bse_tau`, `--use_otsu`

### Sprint 4: Integration & Evaluation
**File**: `run_memory_extraction.py` (complete)
- Two-pass processing with Welford
- Extended output schema
- Bash script adaptation
- Evaluation metrics: FVR, SCD (see proposal.md)

---

## Key Technical Details

**Multi-scale feature handling**:
- Map-Anything Transformer outputs 4-5 intermediate layers + 1 final layer
- Total channels: ~5120 (varies by model architecture)
- Grid resolution: 37×37 or 74×74 depending on model
- Processing: Grid alignment (default) or AnyUp upsampling (optional)

**Real depth handling**:
- Supports both dense and sparse depth
- Sparse depth: Only unproject valid pixels (depth > 0)
- No depth prediction fallback (use real depth only)

**Performance optimizations**:
- Temp files written to `/dev/shm` (RAM disk)
- Welford accumulator: O(1) memory
- Numerical stability: cosine similarity in FP32, Welford in FP64

---

## Migration from map-anything

**Dependencies to preserve**:
- `mapanything.models.init_model`
- `mapanything.datasets.{SevenScenesWAI, Indoor6WAI}`
- `mapanything.tasks.ace.memory_selection.select_optimal_memory_indices`

**Dependencies to replace**:
- Hydra config → argparse
- Hardcoded paths → CLI arguments
- Visualization helpers → optional (keep for debugging)

**Backward compatibility**:
- Output `.pt` format matches map-anything for downstream ACE training
- Can load existing map-anything extracted memories

---

## Answered Questions

✅ Feature extraction: Multi-scale Map-Anything Transformer (UNCHANGED)
✅ Depth usage: Real depth for unprojection (UNCHANGED)
✅ View selection: `select_optimal_memory_indices` (UNCHANGED)
✅ Pooling: Upgrade from uniform voxel to BSE with Otsu
✅ Normalization: Add Welford global normalization
✅ Compatibility: Extended output format, backward compatible

**Deferred to future**: Learned memory extraction (SceneTok-inspired) for depth-free datasets
