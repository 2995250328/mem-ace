# System Architecture: Dual-Branch Memory Extraction

**Author**: Senior ML Systems Architect
**Stage**: 3 — Architecture Design (Revised)
**Date**: 2026-03-26

---

## Architecture Overview: Two Parallel Branches

```
                    run_memory_extraction.py
                            |
                ┌───────────┴───────────┐
                |                       |
         Branch A (Primary)      Branch B (Fallback)
      Map-Anything Pipeline    DINOv2 Single-Scale
      ├─ Multi-scale features  ├─ DINOv2-ViT-L/14
      ├─ Real depth support    ├─ Depth Anything V2
      ├─ View selection        ├─ No view selection
      └─ Proven effective      └─ Depth-free option
```

**Design Rationale**:
- **Branch A**: Validated map-anything approach with multi-scale features + real depth
- **Branch B**: Generic framework for depth-free datasets (future work)
- User selects branch via `--backend` flag: `mapanything` (default) or `dinov2`

---

## Branch A: Map-Anything Pipeline (Primary)

### A.1 Core Components

**1. View Selection** (`select_optimal_memory_indices`)
- Reuse: `mapanything.tasks.ace.memory_selection.select_optimal_memory_indices`
- Input: Full training dataset
- Output: N_MEMORY indices (e.g., 100 views) covering scene geometry
- Strategy: FPS-based spatial coverage + pose diversity

**2. Model Inference** (`model.infer`)
- Model: Map-Anything Transformer (loaded via `mapanything.models.init_model`)
- Input: Selected views with RGB, depth, pose, intrinsics
- Output: Multi-scale intermediate features saved to `.pt` file
- Key parameters:
  - `memory_efficient_inference=True`
  - `use_amp=True, amp_dtype="bf16"`
  - `ignore_depth_inputs`: False (use real depth)
  - `ignore_pose_inputs`: False (use GT poses)
  - `ignore_calibration_inputs`: False (use intrinsics)

**3. Multi-Scale Feature Processing**
- Features: Intermediate layers + final layer from Map-Anything Transformer
- Processing options:
  - **Grid alignment** (default): Align all scales to max grid size, concatenate
  - **AnyUp upsampling** (optional): Learned upsampling to image resolution
- Output: Concatenated features at grid resolution or image resolution

**4. BSE Pooling + Welford Normalization**
- Same as Branch B design (reuse BSEPooler and WelfordNormalizer)
- Input: Unprojected 3D points with multi-scale features
- Output: Compressed memory with normalized coordinates

### A.2 Data Flow

```python
# Step 1: View Selection
memory_indices = select_optimal_memory_indices(train_dataset, N_MEMORY)

# Step 2: Prepare Input Views
input_views = []
for idx in memory_indices:
    raw_data = train_dataset[idx]
    processed = prepare_batch_input(raw_data, device, mode="memory")
    input_views.append(processed)

# Step 3: Model Inference (saves multi-scale features to .pt)
predictions = model.infer(
    input_views,
    memory_efficient_inference=True,
    use_amp=True,
    amp_dtype="bf16",
    save_filename=features_path,
    ignore_depth_inputs=False,  # Use real depth
)

# Step 4: Load Features and Unproject
features_dict = torch.load(features_path)
for view_idx, view_data in enumerate(input_views):
    # Extract multi-scale features for this view
    feat_list = features_dict[f'view_{view_idx}']['features']

    # Align to grid or upsample to image resolution
    if use_anyup:
        features = process_multiscale_features_anyup(feat_list, ...)
    else:
        features = process_multiscale_features_to_grid(feat_list)

    # Unproject with real depth
    points, ray_dirs, colors = unproject_with_real_depth(view_data, features)

    # BSE pooling
    pooled = bse_pooler.pool(points, features, colors, ray_dirs)
    welford.update(pooled['points'])
    save_temp_chunk(pooled)

# Step 5: Welford Normalization (same as Branch B)
mu, sigma = welford.finalize()
normalize_and_save_final_memory(mu, sigma)
```

### A.3 Key Technical Details

**Real Depth Handling**:
- Supports both dense and sparse depth
- Sparse depth: Only unproject valid pixels (depth > 0)
- No depth prediction fallback in Branch A (use Branch B if depth unavailable)

**Multi-Scale Feature Dimensions**:
- Typical: 4-5 intermediate layers + 1 final layer
- Total channels: ~5120 (varies by model architecture)
- Grid resolution: Typically 37×37 or 74×74 depending on model

**View Selection Strategy**:
- `select_optimal_memory_indices` uses FPS on camera positions
- Ensures spatial coverage of scene geometry
- Default N_MEMORY: 100 views (adjustable via config)

---

## Branch B: Learned Memory Extraction (Future Work)

**Purpose**: Generic framework for datasets without real depth, based on fully learned scene representation.

**Key Differences from Branch A**:
- No view selection (processes all training images)
- No depth dependency (neither real nor predicted depth)
- Fully learned feature compression and scene representation
- Reference: SceneTok project's learned scene tokenization approach

**When to Use**:
- Dataset has no depth annotations
- Map-anything model not available
- Need fully end-to-end learned approach

**Design Direction** (to be refined):
- Borrow SceneTok's scene tokenization ideas
- Learned 3D scene representation instead of geometric unprojection
- End-to-end training of feature extraction and compression
- Decoupled from ACE training pipeline

**Implementation**: Deferred until Branch A is validated. Detailed design in future `system_design_branch_b_learned.md`.

---

## Unified File Organization

```
memory_extraction/
├── __init__.py
├── backends/
│   ├── __init__.py
│   ├── mapanything_backend.py    # Branch A implementation
│   └── dinov2_backend.py          # Branch B implementation
├── common/
│   ├── bse_pooling.py             # Shared BSE algorithm
│   ├── welford_meter.py           # Shared normalization
│   └── utils.py                   # Shared utilities
├── run_memory_extraction.py       # Main entry (dispatches to backend)
└── extract_memory.sh              # Bash launcher
```

**Backend Interface** (both branches implement):
```python
class MemoryBackend:
    def select_views(self, dataset, n_views) -> List[int]
    def extract_features(self, views) -> Dict[str, Tensor]
    def unproject(self, view, features) -> Tuple[Tensor, ...]
```

---

## Implementation Roadmap (Revised)

### Sprint 1: Shared Infrastructure
- `bse_pooling.py`, `welford_meter.py` (same as before)
- Backend interface definition

### Sprint 2: Branch A (Map-Anything) - Priority
- `mapanything_backend.py`
- Integrate `select_optimal_memory_indices`
- Wrap `model.infer` with proper error handling
- Multi-scale feature processing (grid alignment + optional AnyUp)

### Sprint 3: Branch B (DINOv2) - Fallback
- `dinov2_backend.py`
- Feature extractor + Depth Anything V2 integration

### Sprint 4: Evaluation
- `eval_boundary_metrics.py`
- Compare Branch A vs Branch B on scenes with real depth

---

## Migration from map-anything

**Dependencies to Preserve**:
- `mapanything.models.init_model`
- `mapanything.datasets.{SevenScenesWAI, Indoor6WAI}`
- `mapanything.tasks.ace.memory_selection.select_optimal_memory_indices`

**Dependencies to Replace**:
- Hydra config → argparse
- Hardcoded paths → CLI arguments
- Visualization helpers → optional (keep for debugging)

**Backward Compatibility**:
- Output `.pt` format matches map-anything for downstream ACE training
- Can load existing map-anything extracted memories
