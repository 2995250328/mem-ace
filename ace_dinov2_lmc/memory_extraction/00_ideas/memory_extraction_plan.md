# Plan: BSE Memory Extraction Migration & Implementation

## Context

Design doc: `memory.txt` (same folder)
Source to migrate: `map-anything/mapanything/tasks/run_memory_extraction.py`
Target: `ace_dinov2_lmc/memory_extraction/` (new subfolder, does NOT touch existing files)

Goal: upgrade memory extraction from vanilla voxel pooling (Paradigm A) to
Bilateral Supervoxel Extraction (BSE) with global metric normalization (Paradigm C).

---

## Target File Structure

```
ace_dinov2_lmc/
├── 00_ideas/
│   ├── memory.txt                        ← design doc (unchanged)
│   └── memory_extraction_plan.md         ← this file
├── trainer_dinov2_lmc.py                 ← existing (unchanged)
├── options_dinov2_lmc.py                 ← existing (unchanged)
└── memory_extraction/                    ← NEW
    ├── __init__.py
    ├── bse_pooling.py                    ← BSE algorithm (standalone)
    ├── run_memory_extraction.py          ← migrated + enhanced
    └── extract_memory.sh                 ← adapted bash script
```

---

## Critical Files (read before implementing)

| File | Role |
|------|------|
| `map-anything/mapanything/tasks/run_memory_extraction.py` | Source to migrate (read-only) |
| `map-anything/bash_scripts/ace/fps_memory.sh` | Bash script to adapt |
| `ace_dinov2_lmc/trainer_dinov2_lmc.py` | DINOv2 loading pattern to reuse |
| `ace_depth/dataset_dinov2.py` | Dataset loader to reuse |

---

## Phase 1 — Migrate + Minimal Additions

**`memory_extraction/run_memory_extraction.py`**

Replace mapanything dependencies:
- `mapanything.models.init_model` → load DINOv2 directly (reuse trainer_dinov2_lmc.py pattern)
- `mapanything.datasets.SevenScenesWAI / Indoor6WAI` → use `dataset.py` / `dataset_dinov2.py`
- `@hydra.main` → plain `argparse`

Keep all helpers: `save_pcd_with_open3d`, `clean_point_cloud_sor`,
`voxel_pooling_optimized`, `_save_depth_on_rgb_vis`

Additions:
- Compute `O_cam = t`, `D_raw = normalize(P_raw - O_cam)`
- After voxel pooling: compute `mu_scene`, `sigma_scene`, normalize `P_bse`
- Save extended `.pt` format (see Output Format below)
- Add `--output_dir` arg (default: `ace_dinov2_lmc/04_evaluation/`)

**`memory_extraction/extract_memory.sh`**
- Adapted from `fps_memory.sh`, remove Hydra, use argparse CLI

---

## Phase 2 — Vectorized Geometric Hashing

**`memory_extraction/bse_pooling.py`**

```python
def coarse_voxel_hash(points: Tensor, voxel_size: float) -> Tensor:
    """Returns cluster_ids via torch.unique on quantized coords."""
    quantized = torch.floor(points / voxel_size).long()
    _, cluster_ids = torch.unique(quantized, dim=0, return_inverse=True)
    return cluster_ids  # [N]
```

Replace chunked `index_add_` loop in `voxel_pooling_optimized` with single-pass
vectorized version using `cluster_ids`.

NOTE: Use `index_add_` + count division (NOT `torch_scatter`) — already proven
in existing `voxel_pooling_optimized`, avoids new dependency.

---

## Phase 3 — Bilateral Feature Clustering (BSE core)

**`memory_extraction/bse_pooling.py`** — add `bse_pooling()`

```python
def bse_pooling(points, ray_dirs, features, colors,
                voxel_size=0.05, tau=0.90, chunk_size=500_000) -> dict:
    """
    1. coarse_voxel_hash → cluster_ids
    2. cluster mean features via index_add_ / count
    3. cosine similarity of each point to its cluster mean (fp32)
    4. is_outlier = sim < tau → sub_cluster_id = cluster_id*2 + is_outlier
    5. fine-grained scatter_mean on sub_cluster_ids
    6. L2-normalize ray_dirs after pooling
    """
```

**Integration in `run_memory_extraction.py`:**
- `--use_bse` flag (default: False for backward compat)
- `--bse_tau` (default: 0.90)
- `--bse_voxel_size` (default: 0.05)
- Adaptive fallback: if output > 100k points, double voxel_size and retry
- Cast to fp32 before cosine similarity, add ε=1e-6 to denominator

---

## Output Format (`.pt` schema)

```python
{
    "points":   P_norm,      # [N, 3] float32, normalized (zero-mean, unit-std)
    "ray_dirs": D_bse,       # [N, 3] float32, unit vectors
    "features": F_bse,       # [N, C] float16, DINOv2 features
    "colors":   C_bse,       # [N, 3] float32, RGB [0,1]
    "mu":       mu_scene,    # [3]    float32, scene mean (for denorm)
    "sigma":    sigma_scene, # scalar float32, scene std
}
```

Backward compat: add both `points`/`pooled_points` and `features`/`pooled_features`
key aliases if needed by downstream code.

---

## Verification Checklist

- [ ] Phase 1 smoke test: run on 7-Scenes Chess, verify `P_norm` in `[-3, 3]`
- [ ] Phase 3 compression: `len(P_bse) / total_input_pixels < 0.05`
- [ ] Boundary preservation: visualize with Open3D
- [ ] Regression: feed new `.pt` to `trainer_dinov2_lmc.py`, verify loss converges
- [ ] Ablation: `--use_bse False` vs `--use_bse True` on Indoor6 scene3/scene4a

---

## Design Notes

- Binary split (main/outlier) is acceptable for v1; can extend to iterative k-means later
- τ=0.90 exposed as `--bse_tau` for ablation
- OOM risk during unprojection → chunked processing (designed in memory.txt)
- No new trainable parameters — pure algebraic operations
