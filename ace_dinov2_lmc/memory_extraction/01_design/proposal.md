# Title: Bilateral Supervoxel Extraction for Scene-Agnostic Memory Compression in Visual Relocalization

## 1. Problem Formulation

Given a set of N training images {I_i} with known poses {T_i ∈ SE(3)} and camera intrinsics {K_i},
the memory extraction pipeline produces a compact scene memory M used by the downstream
ACE DINOv2 LMC relocalization system.

**Current pipeline (Paradigm A — vanilla voxel pooling):**
- Unproject each pixel to 3D: P_raw ∈ R^{H×W×3}
- Voxel-pool features F ∈ R^{H×W×C} into a fixed grid → M_voxel

**Limitations:**
- Uniform voxel grid ignores semantic boundaries → blurs features at object edges
- No global metric normalization → scene-dependent coordinate scale breaks zero-shot transfer
- No ray direction storage → downstream Transformer cannot reason about view-dependent effects
- Tight coupling to mapanything/Hydra ecosystem → cannot run standalone on ace_depth datasets

**Target formulation (Paradigm C — BSE with global normalization):**

Input: {I_i, T_i, K_i}
Output: M = {P_norm, D_bse, F_bse, C_bse, μ_scene, σ_scene}

Where:
- P_norm = (P_raw − μ_scene) / σ_scene  ∈ R^{N×3}  (zero-mean, unit-std)
- D_bse  = normalize(P_raw − O_cam)     ∈ R^{N×3}  (unit ray directions)
- F_bse  ∈ R^{N×C}  (DINOv2 features after bilateral pooling)
- C_bse  ∈ R^{N×3}  (RGB colors after pooling)
- μ_scene ∈ R^3, σ_scene ∈ R  (scene statistics for denormalization)

Objective: maximize semantic boundary preservation while achieving >95% point compression.

## 2. Literature Landscape & Motivation

### Representative Methods

**ACE (Cavallari et al., CVPR 2023)**
- Assumption: scene coordinates can be regressed from local patch features without explicit 3D map
- Limitation: no persistent scene memory; each scene requires full retraining from scratch
- Gap: cannot leverage cross-scene geometric priors

**Map-Anything / ACE-G (internal, 2025)**
- Assumption: a pre-extracted voxel-pooled memory M_voxel can guide iterative LMC training
- Limitation: uniform voxel grid → semantic boundary blurring; Hydra/mapanything coupling
- Gap: memory quality directly limits relocalization accuracy; no principled feature-geometry alignment

**PointNeRF (Xu et al., CVPR 2022)**
- Assumption: neural point clouds with per-point features enable view synthesis
- Limitation: requires dense depth; per-scene optimization; not designed for relocalization
- Gap: feature pooling strategy not transferable to sparse relocalization setting

### Research Gap

Existing memory extraction pipelines treat all 3D points uniformly during pooling.
This ignores the fact that semantic boundaries (edges between objects) carry disproportionately
more discriminative information for relocalization. Furthermore, scene-specific coordinate scales
prevent zero-shot transfer of the downstream regressor.

**This idea addresses two gaps simultaneously:**
1. Semantic-aware pooling: bilateral clustering preserves high-frequency boundaries
2. Global metric normalization: enables scene-agnostic downstream training

### Why Non-Trivial

- Bilateral filtering in 3D point clouds requires efficient GPU implementation without torch_scatter
- **Adaptive threshold selection**: Fixed τ risks over/under-segmentation across scenes with varying contrast. Otsu's method provides principled, data-driven threshold selection.
- **Chunked processing with global normalization**: Computing μ_scene, σ_scene on chunked data requires streaming algorithms (Welford) to maintain mathematical consistency.
- Migration from Hydra/mapanything to standalone argparse requires careful dependency analysis

## 3. Design Space & Selected Architecture

### Paradigm A — Uniform Voxel Pooling (current baseline)
- Simple grid quantization, average features per voxel
- Pro: O(N) time, no hyperparameters beyond voxel_size
- Con: blurs semantic boundaries; no feature-geometry alignment

### Paradigm B — FPS + kNN Pooling
- Farthest Point Sampling selects representative points, kNN aggregates neighbors
- Pro: uniform spatial coverage
- Con: O(N²) FPS is prohibitively slow for >1M points; loses boundary structure

### Paradigm C — Bilateral Supervoxel Extraction (SELECTED)
- Two-level clustering: coarse geometric hash → fine bilateral split by feature similarity
- **Adaptive threshold**: Uses Otsu's method to find optimal split point τ_otsu from cosine similarity histogram (no fixed hyperparameter)
- Pro: preserves semantic boundaries; O(N) vectorized; no new dependencies; adapts to scene contrast
- Con: assumes bimodal similarity distribution (mitigated by unimodal guard)

**Justification for Paradigm C:**
- Theoretical: bilateral filtering is the principled way to preserve discontinuities
- Practical: can be implemented with index_add_ (already in codebase), no torch_scatter needed
- Adaptive: Otsu's method automatically finds the optimal threshold for each chunk based on local feature distribution
- Ablation-friendly: Otsu can be disabled with --use_otsu False to fall back to fixed τ

### Detailed Pipeline Steps

**Step 1 — Plenoptic Unprojection**

For each valid pixel (u, v) with depth Z, intrinsics K, rotation R, translation t:
```
P_cam = Z · K⁻¹ · [u, v, 1]ᵀ
P_raw = R · P_cam + t          # world coordinates [N, 3]
O_cam = t                       # camera center in world [3]
D_raw = (P_raw - O_cam) / ‖P_raw - O_cam‖₂   # unit ray direction [N, 3]
```

**Step 2 — Coarse Geometric Hashing**
```
V_idx = floor(P_raw / voxel_size).long()   # [N, 3]
_, cluster_ids = torch.unique(V_idx, dim=0, return_inverse=True)  # [N]
```

**Step 3 — Bilateral Feature Clustering (the boundary-preserving split)**

Must avoid Python loops — fully vectorized:
```
# Compute cluster mean features via index_add_ (no torch_scatter needed)
F_fp32 = F_raw.float()                          # cast to fp32 for numerical safety
F_mean = zeros(M, C).index_add_(0, cluster_ids, F_fp32) / count  # [M, C]

# Cosine similarity of each point to its cluster mean
F_i    = F_fp32                                 # [N, C]
F_ci   = F_mean[cluster_ids]                    # [N, C]
sim_i  = (F_i * F_ci).sum(1) / (‖F_i‖ · ‖F_ci‖ + 1e-6)  # [N], fp32

# Otsu adaptive thresholding (finds optimal split automatically)
hist, bins = histogram(sim_i, bins=256, range=[0, 1])
τ_otsu = otsu_threshold(hist, bins)             # maximizes between-class variance

# Unimodal guard: skip splitting if distribution is too uniform
if std(sim_i) < 0.02:
    τ_otsu = -1.0  # force all points to main cluster

# Binary split: outlier = semantically different from cluster majority
is_outlier = (sim_i < τ_otsu).long()            # [N], 0 or 1
sub_id = cluster_ids * 2 + is_outlier           # [N]
```

**Otsu's Method**: Treats similarity distribution as a mixture of two classes (main cluster vs. boundary). Searches for threshold that maximizes inter-class variance, equivalent to minimizing intra-class variance. Proven optimal for bimodal distributions.

**Step 4 — Fine-grained Pooling**
```
# scatter_mean on {P_raw, D_raw, F_raw, colors} using sub_id
P_bse, D_bse, F_bse, C_bse = scatter_mean_all(sub_id)
D_bse = D_bse / (‖D_bse‖₂ + 1e-6)             # re-normalize ray dirs
```

**Step 5 — Global Metric Normalization**

Computed AFTER BSE pooling (on P_bse, not P_raw — avoids outlier contamination):
```
μ_scene = mean(P_bse, dim=0)                    # [3]
σ_scene = std(‖P_bse - μ_scene‖₂).item()       # scalar
P_norm  = (P_bse - μ_scene) / σ_scene           # [N, 3]
```

This ensures P_norm ∈ [-3, 3] for typical indoor scenes (3σ coverage).

### OOM Mitigation: Welford Streaming Normalization

For large scenes (>1000 images at high resolution), full unprojection exceeds GPU memory.
Use Welford's online algorithm for numerically stable, O(1) memory global statistics:

```
Pass 1 (per-chunk streaming): for each batch of B images:
    unproject → D_raw → BSE (Steps 1-4) → P_bse_chunk

    # Welford online update (FP64 for numerical stability)
    for each point p in P_bse_chunk:
        count += 1
        delta = p - mean
        mean += delta / count
        M2 += delta * (p - mean)  # running sum of squared deviations

    # Spool chunk to disk (temporary .pt file)
    save(P_bse_chunk, F_bse_chunk, D_bse_chunk, f"temp_chunk_{i}.pt")

Pass 2 (global scaling): after all chunks processed:
    μ_scene = mean                          # [3], FP64 → FP32
    σ_scene = sqrt(M2 / count)              # scalar, FP64 → FP32

    # Stream normalize each chunk
    for each temp_chunk_{i}.pt:
        load(P_bse_chunk, F_bse_chunk, D_bse_chunk)
        P_norm_chunk = (P_bse_chunk - μ_scene) / σ_scene
        append to final buffer

    # Save final memory
    save(P_norm, F_bse, D_bse, C_bse, μ_scene, σ_scene, "pooled_memory.pt")
    cleanup temp files
```

**Key advantages**:
- O(1) memory: only stores running statistics, not all points
- Numerically stable: Welford's algorithm avoids catastrophic cancellation in variance computation
- Mathematically exact: produces identical μ, σ as full batch computation (up to FP64 precision)
- No double-pooling: BSE runs once per chunk, normalization is a simple scalar operation

Default: B = 16 images per chunk, temp files written to /dev/shm (RAM disk) for fast I/O.

### Adaptive Fallback

If len(P_bse) > 100,000 after BSE (insufficient compression):
- Double voxel_size and rerun Steps 2-4
- Log warning with original and new point counts

## 4. Evaluation & Validation Plan

### Datasets
- **7-Scenes Chess**: smoke test (small, fast, well-understood)
- **Indoor6 scene3, scene4a**: main ablation (existing eval results in dino_lmc_base)

### Metrics
- **Compression ratio**: len(P_bse) / total_input_pixels (target: <0.05)
- **FVR (Feature Variance Retention)**: Var(F_bse) / Var(F_raw) — measures information preservation in feature space (target: >0.90)
- **SCD (Semantic Chamfer Distance)**: Chamfer distance in joint space [P_xyz, λ·F_dino] between raw and pooled point clouds (target: <0.5× baseline)
- **Relocalization accuracy**: median translation error (cm) and rotation error (°) on Indoor6
- **Normalization check**: assert P_norm ∈ [-5, 5] (99.9% of points)

### Baselines
1. `--use_bse False` (vanilla voxel pooling, current behavior)
2. `--use_bse True --use_otsu True` (proposed: Otsu-adaptive BSE)
3. `--use_bse True --use_otsu False --bse_tau 0.90` (fixed threshold BSE, for comparison)

### Ablation Studies
- Otsu vs. fixed τ: compare FVR and SCD on Indoor6 scene3 (varying contrast)
- voxel_size sweep: {0.03, 0.05, 0.10} on 7-Scenes Chess
- Unimodal guard: measure false split rate on homogeneous regions (flat walls)
- Welford precision: verify μ_scene, σ_scene match full-batch computation to 4 decimal places

## 5. Expected Failure Modes & Engineering Risks

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| torch_scatter not in ace env | High | Use index_add_ + count division (proven in existing code) |
| OOM during unprojection of large scenes | Medium | Welford streaming with O(1) memory, temp files to /dev/shm |
| FP16 numerical instability in cosine similarity | High | Cast to fp32 before dot product; add ε=1e-6 to denominator |
| FP32 precision loss in Welford M2 accumulator | High | Use FP64 for mean and M2, convert to FP32 only at output |
| Otsu fails on unimodal distributions | Medium | Unimodal guard: skip splitting if std(sim) < 0.02 |
| Otsu assumes bimodal distribution | Medium | Acceptable for DINOv2 features (semantic boundaries create natural bimodality) |
| Disk I/O bottleneck in temp file spooling | Low | Write to /dev/shm (RAM disk) instead of HDD; async torch.save |
| Output >100k points (insufficient compression) | Low | Adaptive fallback: double voxel_size and retry once |
| mapanything import paths break | High | Replace all mapanything.* imports with direct equivalents |
| WAI dataset format incompatibility | Medium | Test with ace backend first; WAI support is optional |

## 6. Reuse Plan

### Inherited from dino_lmc_base
- `../trainer_dinov2_lmc.py`: DINOv2 model loading pattern (lines 1-80 show import structure)
- `../options_dinov2_lmc.py`: argparse builder pattern, `_strtobool`, `--data_backend` arg
- `../train_ace_dinov2_lmc.py`: script entry point structure (sys.path, device setup)

### From Project Root
- `../../dataset_dinov2.py`: `CamLocDatasetDINOv2` — RGB loading, intrinsics, pose loading
- `../../ace_util.py`: `to_homogeneous` — coordinate transforms

### New Files (this idea only)
- `memory_extraction/__init__.py` — empty package marker
- `memory_extraction/bse_pooling.py` — standalone BSE algorithm (no external deps beyond torch)
- `memory_extraction/run_memory_extraction.py` — migrated + enhanced extraction script
- `memory_extraction/extract_memory.sh` — adapted bash launcher (no Hydra)

### Migration Source (read-only reference)
- `map-anything/mapanything/tasks/run_memory_extraction.py`
- `map-anything/bash_scripts/ace/fps_memory.sh`

## 7. Implementation Roadmap

Three incremental sprints — each independently testable:

**Sprint 1 — Welford Streaming Normalization (resolve mathematical contradiction)**
- Implement WelfordMeter class with FP64 accumulators (count, mean, M2)
- Rewrite main loop in run_memory_extraction.py for two-pass processing
- Pass 1: stream chunks, update Welford state, spool to temp files
- Pass 2: load temp files, apply normalization, assemble final memory
- Acceptance: verify μ_scene, σ_scene match full-batch computation to 4 decimal places

**Sprint 2 — Otsu Adaptive Thresholding (resolve hyperparameter critique)**
- Implement vectorized otsu_threshold(sim_tensor, bins=256) function
- Add unimodal guard: skip splitting if std(sim) < 0.02
- Replace `is_outlier = sim < 0.90` with `is_outlier = sim < τ_otsu`
- Log τ_otsu for each chunk to observe dynamic adaptation
- Acceptance: verify τ_otsu varies across chunks with different contrast levels

**Sprint 3 — Quantitative Evaluation Metrics (provide rebuttal ammunition)**
- Implement eval_boundary_metrics.py script
- Compute FVR = Var(F_bse) / Var(F_raw) for boundary preservation
- Compute SCD in joint [P, λ·F] space using Chamfer distance
- Generate comparison table: Uniform Voxel vs. Otsu-BSE
- Acceptance: FVR > 0.90 for Otsu-BSE, FVR < 0.50 for Uniform baseline
