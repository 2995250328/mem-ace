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
- The binary split (main/outlier) must be calibrated (τ) to avoid over-segmentation
- Global normalization must be computed before pooling to avoid data leakage
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
- Pro: preserves semantic boundaries; O(N) vectorized; no new dependencies
- Con: binary split may be too coarse for complex boundaries (acceptable for v1)

**Justification for Paradigm C:**
- Theoretical: bilateral filtering is the principled way to preserve discontinuities
- Practical: can be implemented with index_add_ (already in codebase), no torch_scatter needed
- Ablation-friendly: τ exposed as CLI arg; BSE can be disabled with --use_bse False

### Core Algorithm: BSE

```
Step 1: coarse_voxel_hash(P, voxel_size) → cluster_ids  [O(N)]
Step 2: scatter_mean(F, cluster_ids) → F_cluster_mean   [O(N) via index_add_]
Step 3: cosine_sim(F_i, F_cluster_mean[cluster_ids[i]]) → sim_i  [O(N)]
Step 4: is_outlier_i = (sim_i < τ)
Step 5: sub_id_i = cluster_ids[i] * 2 + is_outlier_i
Step 6: scatter_mean({P,D,F,C}, sub_id) → pooled output  [O(N)]
Step 7: L2-normalize pooled D
```

### Global Metric Normalization

Computed AFTER unprojection, BEFORE pooling:
```
μ_scene = mean(P_raw, dim=0)          # [3]
σ_scene = std(P_raw - μ_scene).item() # scalar
P_norm  = (P_raw - μ_scene) / σ_scene
```

This ensures P_norm ∈ [-3, 3] for typical indoor scenes (3σ coverage).

## 4. Evaluation & Validation Plan

### Datasets
- **7-Scenes Chess**: smoke test (small, fast, well-understood)
- **Indoor6 scene3, scene4a**: main ablation (existing eval results in dino_lmc_base)

### Metrics
- **Compression ratio**: len(P_bse) / total_input_pixels (target: <0.05)
- **Boundary preservation**: qualitative Open3D visualization
- **Relocalization accuracy**: median translation error (cm) and rotation error (°) on Indoor6
- **Normalization check**: assert P_norm ∈ [-5, 5] (99.9% of points)

### Baselines
1. `--use_bse False` (vanilla voxel pooling, current behavior)
2. `--use_bse True --bse_tau 0.90` (proposed BSE)
3. `--use_bse True --bse_tau 0.70` (aggressive split, ablation)
4. `--use_bse True --bse_tau 0.95` (conservative split, ablation)

### Ablation Studies
- τ sweep: {0.70, 0.80, 0.90, 0.95} on Indoor6 scene3
- voxel_size sweep: {0.03, 0.05, 0.10} on 7-Scenes Chess
- With/without global normalization (P_raw vs P_norm as input to downstream trainer)

## 5. Expected Failure Modes & Engineering Risks

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| torch_scatter not in ace env | High | Use index_add_ + count division (proven in existing code) |
| OOM during unprojection of large scenes | Medium | Chunked processing (chunk_size=500k) |
| τ=0.90 over-segments flat walls | Medium | Expose as --bse_tau; default conservative |
| Output >100k points (insufficient compression) | Low | Adaptive fallback: double voxel_size and retry |
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
