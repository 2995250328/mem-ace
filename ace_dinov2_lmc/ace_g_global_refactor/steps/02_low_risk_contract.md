# Step 02: Low-Risk ACE-G Global Contract

Status: done.

## Goal

Make the current ACE-G global baseline easier to audit without changing its
default numerical training path. This step only explicitizes contracts that were
already implicit.

## Implemented Measures

- Key layer selection is explicit through `lmc_key_slice_idx`.
- Default key slice remains legacy-compatible:
  - `num_layers <= 1`: use the full feature.
  - `num_layers > 1`: default to `slice 2`, matching the previous effective
    `_get_layer_slice(..., 1)` behavior for five-layer MapAnything memory.
- Invalid explicit key slice indices raise `ValueError` instead of silently
  falling back.
- Training `lmc_config` now records:
  - `requested_lmc_mode`
  - `effective_lmc_mode`
  - `layers_idx`
  - `lmc_key_slice_idx`
  - `lmc_key_layer_label`
- Evaluation passes `lmc_key_slice_idx` back into `GeoLMC`; old checkpoints
  without this field use the same legacy-compatible default.
- S1/S2 compressor trainability is explicit:
  - S1 sets compressor parameters trainable.
  - S2 sets compressor parameters frozen.
  - `_compress_memory()` restores the compressor's previous train/eval mode
    instead of forcing train mode.

## Non-Goals

- No loss-path refactor.
- No S1 sampled ReproLoss time-axis change.
- No fusion architecture change.
- No change to default key slice for existing five-layer memory.

## Verification Points

- `options_dinov2_lmc.py`: exposes `--lmc_key_slice_idx`.
- `/home/xwh/project/ace_depth/ace_compressor.py`: resolves and validates
  explicit zero-based key slice semantics.
- `trainer_dinov2_lmc.py`: logs and checkpoints requested/effective mode,
  layer metadata, key slice, and compressor trainability boundary.
- `test_ace_dinov2_lmc.py`: reconstructs `GeoLMC` with the checkpoint key slice.
