# Migration Plan: Memory Compression + Two-Stage Training → DINO ACE

## Context

ace_depth 已有完整的 DINOv2 ACE 实现（encoder + head + trainer + dataset）。`tasks/ace/__init__.py` 已声明了对 `compressor.py`、`fusion.py`、`utils.py` 的导入，但这些文件尚未创建。

目标：从 `map-anything/mapanything/tasks/ace/` 迁入 memory 压缩/融合模块和两阶段训练逻辑，在 ace_depth 中形成完整的 DINO ACE + LMC 流程。

## Data Flow

```
Image → DINOv2Encoder (frozen, 1024-dim, 14x) → Dense Features (B, 1024, H/14, W/14)
                                                       ↓
Memory Bank (.pt, loaded from disk) → GeoLMC → Latent Tokens (B, K, C)
                                                       ↓
LMCFeatureFusion(query=DenseFeats, memory=LatentTokens) → Fused Features
                                                       ↓
ACEHead_Homogeneous_Mean → 3D Coords + Confidence (B, 5, H/14, W/14)
                                                       ↓
ReproLoss (dyntanh) + Scale Regularization → Loss
```

## Two-Stage Iterative Training

```
for iter_idx in range(15):
    Stage 1 (S1): Train compressor + fusion online (500 steps, 2000 for iter 0)
        - Freeze encoder, train compressor + fusion + head(0.1x LR)
        - Forward: encoder → compressor → fusion → head → repro loss

    Stage 2 (S2): Train head offline on buffer (14K steps, 42K for final iter)
        - Freeze encoder + compressor + fusion
        - Collect buffer with fused features
        - Train head on buffer with chunk schedule
```

## Files to Create/Modify

### Phase 1: Core Modules (copy from map-anything, fix imports)

Source base: `/home/xwh/project/map-anything/mapanything/tasks/ace/`
Target base: `/home/xwh/project/ace_depth/tasks/ace/`

| # | File | Action | Source | Key Changes |
|---|------|--------|--------|-------------|
| 1 | `tasks/ace/compressor.py` | CREATE | `compressor.py` | Copy verbatim. Self-contained (torch only). Contains GeoLMC, FourierPositionEncoding, geometric bias modules. |
| 2 | `tasks/ace/fusion.py` | CREATE | `fusion.py` | Fix import: `mapanything.tasks.ace.compressor` → `tasks.ace.compressor`. Contains LMCFusionBlock, LMCFeatureFusion. |
| 3 | `tasks/ace/regression_head.py` | CREATE | `regression_head.py` | Copy verbatim. Contains ACEHead_Homogeneous_Mean (recommended), PointwiseBlock, FiLM1x1, load_regression_head. |
| 4 | `tasks/ace/loss_utils.py` | CREATE | `loss_utils.py` | Fix import: `mapanything.tasks.ace.common` → `tasks.ace.common`. Contains ReproLoss (dyntanh), _loss_fn, scale regularization. |
| 5 | `tasks/ace/buffer.py` | CREATE | `buffer.py` | Fix imports. Copy BufferTensors, FeatureReplayBuffer, BufferDataset. Adapt `_collect_buffer` → `_collect_buffer_dinov2` for CamLocDatasetDINOv2 tuple format + LMCInferenceWrapper. |
| 6 | `tasks/ace/utils.py` | CREATE | `utils.py` | Trim heavily. Keep `load_memory_features` + create `LMCInferenceWrapper` adapted for DINOv2Encoder. Remove mapanything-specific imports. |
| 7 | `tasks/ace/common.py` | MODIFY | `common.py` | Add from source: `_invert_c2w_to_w2c`, `strip_module_prefix`, `unwrap_model` (~30 lines). Keep existing functions. |
| 8 | `tasks/ace/training_utils.py` | CREATE | `training_utils.py` | Keep `MetricLogger`, `reset_head_weights`. Remove hydra/omegaconf deps, use plain args. |
| 9 | `tasks/ace/checkpoint.py` | CREATE | `checkpoint.py` | Simplified version: `save_checkpoint(path, compressor, fusion, head)`, `load_checkpoint_weights(path, ...)`. Remove OmegaConf deps. |
| 10 | `tasks/ace/__init__.py` | MODIFY | — | Extend with imports for regression_head, loss_utils, buffer, training_utils, checkpoint. |

### Phase 2: Integration (new files connecting LMC to DINOv2)

| # | File | Action | Pattern | Description |
|---|------|--------|---------|-------------|
| 11 | `trainer_dinov2_lmc.py` | CREATE | `trainer_dinov2.py` + map-anything `train_ace.py` | Main two-stage iterative trainer. Uses DINOv2Encoder + GeoLMC + LMCFeatureFusion + ACEHead_Homogeneous_Mean. Implements `_stage1_train_lmc()` and `_stage2_train_head()`. |
| 12 | `train_ace_dinov2_lmc.py` | CREATE | `train_ace_dinov2.py` | CLI entry point. Adds args: `--memory_path`, `--lmc_mode`, `--num_latent_tokens`, `--iterations`, `--lmc_warmup_steps`, `--lmc_train_steps`, `--head_reset_strategy`. |
| 13 | `test_ace_dinov2_lmc.py` | CREATE | `test_ace_dinov2.py` | Test script. Loads memory + checkpoint → builds LMCInferenceWrapper → head → DSAC* RANSAC. |

### Files NOT Modified

- `ace_network_dinov2.py` — DINOv2Encoder class reused directly, Head class untouched (new head in regression_head.py)
- `dataset_dinov2.py` — reused as-is for data loading
- `ace_loss.py` — kept for backward compat with non-LMC variants
- `trainer_dinov2.py` — kept for non-LMC single-stage training
- `train_ace_dinov2.py` / `test_ace_dinov2.py` — kept for non-LMC usage

## Key Integration Points

1. **Encoder output**: `DINOv2Encoder.forward()` → `(B, 1024, H/14, W/14)` — compatible with LMCFeatureFusion (accepts `(B, N_q, C)` after flatten+transpose)
2. **Memory bank**: Loaded via `load_memory_features(path, device)` → dict with `pooled_points`, `pooled_features`, `scene_center`, `all_scale_tokens`
3. **Feature dim**: DINOv2 outputs 1024-dim; memory bank may have different dim. LMCFeatureFusion handles this via separate `query_feature_dim` / `memory_feature_dim` params.
4. **Head output**: ACEHead_Homogeneous_Mean outputs 5 channels (xyz + w + confidence). For DSAC*, slice `[:, :3]` after dehomogenization.
5. **Dataset adapter**: CamLocDatasetDINOv2 returns tuples `(image, mask, pose, pose_inv, K, K_inv, coords, filename)`. Buffer collection and S1 training need to unpack this format (vs map-anything's dict format).

## Implementation Order

```
Step 1:  tasks/ace/compressor.py          (no deps)
Step 2:  tasks/ace/common.py              (add utility functions)
Step 3:  tasks/ace/fusion.py              (depends on compressor)
Step 4:  tasks/ace/regression_head.py     (no deps)
Step 5:  tasks/ace/loss_utils.py          (depends on common)
Step 6:  tasks/ace/utils.py               (depends on compressor, fusion)
Step 7:  tasks/ace/buffer.py              (depends on common, utils)
Step 8:  tasks/ace/training_utils.py      (depends on regression_head)
Step 9:  tasks/ace/checkpoint.py          (depends on common)
Step 10: tasks/ace/__init__.py            (extend imports)
Step 11: trainer_dinov2_lmc.py            (depends on all above + ace_network_dinov2 + dataset_dinov2)
Step 12: train_ace_dinov2_lmc.py          (depends on trainer)
Step 13: test_ace_dinov2_lmc.py           (depends on utils, regression_head, ace_network_dinov2)
```

## Verification

1. Import test: `python -c "from tasks.ace import GeoLMC, LMCFeatureFusion, load_memory_features"` — verifies all modules load
2. Forward pass test: Construct full pipeline with dummy data, verify output shapes
3. Training smoke test: `python train_ace_dinov2_lmc.py datasets/7scenes_chess output/test_lmc.pt --memory_path <memory.pt> --iterations 1 --lmc_warmup_steps 10 --epochs 1` — verify one full S1+S2 cycle runs without error
4. Inference test: `python test_ace_dinov2_lmc.py datasets/7scenes_chess output/test_lmc.pt --memory_path <memory.pt>` — verify DSAC* produces pose estimates
