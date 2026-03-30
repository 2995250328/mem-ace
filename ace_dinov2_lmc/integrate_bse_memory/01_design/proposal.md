# Title: Integrating BSE Memory into DINOv2-LMC Training Pipeline

## 1. Problem Formulation

**Goal**: Replace the simple-voxel memory loading in DINOv2-LMC trainer with BSE (Bilateral Supervoxel Extraction) pooled memory, to validate whether BSE's boundary-preserving supervoxel clustering produces better scene representations for camera relocalization.

**Input**: BSE memory file (`*_bse.pt`) containing:
- `points` [N, 3]: 3D point positions from BSE supervoxel centroids
- `features` [N, C]: DINOv2/MapAnything features per point (fp16)
- `mu` [3]: scene center
- `colors` [N, 3], `cluster_sizes` [N], ray data (optional)

**Output**: A memory dict consumed by `TrainerACEDINOv2LMC` with fields:
- `pooled_points` [N, 3], `pooled_features` [N, D], `scene_center` [3], etc.

**Objective**: Minimal adapter layer that maps BSE format → pooled format. No changes to LMC compressor, fusion, training loop, or regression head.

## 2. Literature Landscape & Motivation

**Context**: Scene memory representations for visual relocalization typically use:
1. **Uniform voxelization**: Divide 3D space into regular grid, average features per cell (current approach in dino_lmc_base)
2. **Bilateral Supervoxel Extraction (BSE)**: Cluster points by spatial proximity AND feature similarity, preserving geometric boundaries

**Research gap**: While BSE is known to produce better segmentation boundaries in point cloud processing, it has not been validated as a memory representation for ACE-style coordinate regression. The key question is whether boundary-aware clustering leads to more discriminative memory features that improve relocalization accuracy.

**Why non-trivial**:
- BSE produces different point distributions (non-uniform density)
- Feature dimensions and format differ from the current pooled memory
- Missing fields (all_scale_tokens, layers_idx) need graceful handling
- Must maintain numerical compatibility with existing LMC compressor

## 3. Design Space & Selected Architecture

### Option A: Extend `load_memory_features()` with BSE format detection
- Add a branch in the existing format detection logic (`if "pooled_points" in payload` → currently handled)
- Detect BSE format by checking for `"points" in payload and "features" in payload and "schema_version" in payload`
- Map fields inline: rename, cast types, construct missing fields
- **Pros**: Single entry point, no trainer changes, format auto-detection
- **Cons**: Modifies shared utility function

### Option B: Separate adapter function + new CLI flag
- Add `load_bse_memory()` as a separate function in `utils_lmc.py`
- Add `--memory_format bse|pooled` flag to options parser
- In trainer, call the appropriate loader based on flag
- **Pros**: Clean separation, old format untouched
- **Cons**: Requires trainer change (loading dispatch), adds CLI surface

### Option C: Offline conversion script
- Write a standalone script that converts BSE → pooled format
- Run before training, produces a pooled-format .pt file
- **Pros**: Zero runtime code changes
- **Cons**: Extra pipeline step, disk space, harder to iterate

### Selected: Option A — Extend `load_memory_features()`

**Justification**:
1. The function already has format detection logic (pooled vs intermediate)
2. Adding a BSE branch is natural and self-contained
3. No changes to trainer, options, or LMC modules
4. Auto-detection means no new CLI flags needed
5. The function's return contract (dict with `pooled_points`, `pooled_features`, etc.) is preserved exactly

**Modification scope**:
- `utils_lmc.py`: Add ~30 lines to `load_memory_features()` for BSE format detection and field mapping
- All other files: **zero changes**

### Detailed Adapter Logic

```
1. Detect BSE format: "points" in payload AND "features" in payload AND "schema_version" in payload
2. Field mapping:
   - points → pooled_points (rename)
   - features → pooled_features (rename, cast fp16→float32)
   - colors → pooled_colors (rename)
   - mu → scene_center (rename)
3. Construct missing optional fields:
   - all_poses: Build [M, 3, 4] from view_camera_rotations + view_camera_centers (if available)
   - all_intrinsics: Map from view_camera_intrinsics (if available)
   - all_scale_tokens: None (warn as missing, same as current behavior when absent)
   - layers_idx: [] (trainer defaults to num_layers=4)
4. Validate shapes (reuse existing validation logic)
5. Return same dict format as pooled path
```

### Feature Dimension Compatibility

BSE features are 1024-dim (from DINOv2 ViT-L/14 or MapAnything). The trainer's `layers_idx` defaults to `[]` when absent, which triggers `num_layers=4`, giving `feature_dim=1024/4=256`. This is consistent with how the old pooled format worked without explicit `layers_idx`.

If the user wants single-layer behavior (`num_layers=1`, `feature_dim=1024`), they can pass `--num_layers 1` or we can add a `layers_idx` hint in the BSE memory file in the future. For now, the default 4-layer split is the safe choice.

## 4. Evaluation & Validation Plan

### Datasets
- **7-Scenes**: chess, heads, fire (representative small scenes)
- **Indoor6**: scene3 (representative large scene)

### Metrics
- **Primary**: pct<5cm (percentage of frames within 5cm/5°) — used for `best_metric`
- **Secondary**: median translation error, median rotation error

### Baselines
- Old pooled memory (simple voxelization) — existing results
- BSE memory (this work) — same training config, only memory differs

### Ablation (future, not this PR)
- BSE voxel size (0.03, 0.05, 0.08, 0.10)
- Number of memory views (10, 20, 40)
- MapAnything vs DINOv2 feature extractor

### Validation Steps
1. Unit test: adapter correctly maps all BSE fields
2. Shape test: output dict passes `preflight_memory_features()` checks
3. Training smoke test: 2-iteration run completes without error
4. Full comparison: run both memories on same scene, compare metrics

## 5. Expected Failure Modes & Engineering Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Feature dim mismatch (BSE 1024 vs expected) | Low | High | Validate in adapter, log feature_dim |
| Missing all_scale_tokens changes LMC behavior | Medium | Low | Trainer already handles None gracefully (defaults to dim=1024) |
| Different point density changes visibility stats | Medium | Low | Auto-mode fallback will switch global→local if needed |
| fp16→float32 precision loss | Very Low | Very Low | Features are already fp16 in old format too |
| BSE file has 0 points (empty scene) | Low | High | Reuse existing empty-check in validation |

## 6. Reuse Plan

### Files Modified (minimal scope)
| File | Change | Lines |
|---|---|---|
| `utils_lmc.py` | Add BSE format branch to `load_memory_features()` | ~30 lines added |

### Files NOT Modified (preserved exactly)
- `trainer_dinov2_lmc.py` — LMC training logic, S1/S2 loop
- `train_ace_dinov2_lmc.py` — training entry point
- `options_dinov2_lmc.py` — CLI argument parser
- `test_ace_dinov2_lmc.py` — testing script
- `ace_compressor.py` — GeoLMC compressor
- `ace_fusion.py` — LMCFeatureFusion
- `ace_loss.py` — ReproLoss
- `ace_network_dinov2.py` — DINOv2 Regressor

### Inherited from Parent Ideas
- From `dino_lmc_base`: Full trainer infrastructure (unchanged)
- From `memory_extraction`: BSE memory file format specification (read-only reference)
