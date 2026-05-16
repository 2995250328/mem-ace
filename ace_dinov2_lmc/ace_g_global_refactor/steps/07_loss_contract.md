# Step 07 - Shared Reprojection / Invalid-Loss Contract

Status: `[done]`

## Purpose

Reduce hidden loss-path drift before changing ACE-G fusion or compressor
architecture.

This step is a compatibility-preserving refactor. It does not change the ACE
reprojection formula, invalid target construction, S1/S2 time-axis policy, S2
compressor freeze rule, fusion behavior, or any training preset.

## Changed Code

Main implementation:

- `trainer_dinov2_lmc.py`

New shared helper:

- `_compute_reprojection_invalid_loss_contract(...)`

The helper owns the common ACE loss core:

- scene coordinates to camera coordinates
- camera coordinates to pixel reprojection
- L1/L2 reprojection error
- depth / hard-clamp / nonfinite invalid mask
- valid ReproLoss call
- invalid camera-space proxy loss
- common stats

The caller still owns stage-specific policy. This is intentional.

## Preserved Stage Policies

S1 full-map:

- Keeps `image_mask_B1HW`.
- Keeps nuclear outlier exclusion.
- Keeps `s1_full_map_max_points` valid-point subsampling.
- Keeps the historical normalizer `N = B * H * W`.
- Keeps S1 full-map step selection:
  `mapany_flow_profile ? _monotonic_repro_step() : s1_step`.
- Does not use `loss_invalid_max_delta` clamp.
- Keeps invalid `nan_to_num(..., posinf=1e4, neginf=1e4)`.

S1 sampled / sampled-buffer:

- Keeps `_resolve_s1_sampled_loss_step(s1_step)`.
- Keeps historical compatibility defaults:
  `fixed_zero`, `per_iter`, or `global_monotonic`.
- Keeps normalizer `batch_size`.
- Does not use `loss_invalid_max_delta` clamp.
- Keeps invalid `nan_to_num(..., posinf=1e4, neginf=1e4)`.

Standard S2:

- Keeps existing `repro_step_mode` policy:
  `global_monotonic`, `per_iter`, or `legacy_rewind`.
- Keeps normalizer `batch_size`.
- Keeps `loss_invalid_max_delta` per-component clamp.
- Keeps invalid `nan_to_num(..., posinf=0.0, neginf=0.0)`.
- Optimizer, scheduler, finite-loss guard, gradient clipping, and logging remain
  in the S2 step.

ACE-G S2-G:

- Keeps the only intended difference from standard S2: raw features are fused
  before the head.
- Keeps R1/R2 fusion trainability behavior.
- Uses the same shared reprojection / invalid-loss helper as standard S2 after
  `pred_scene_coords_b3HW` has been produced.
- Optimizer, scheduler, finite-loss guard, gradient clipping, and logging remain
  in the S2-G step.

## Verification

Completed lightweight checks:

- `python -m py_compile trainer_dinov2_lmc.py`
- Synthetic S1 parity check:
  - copied the old S1 full-map and S1 sampled formulas into a temporary script
  - compared loss and valid/invalid masks against the new helper
  - result: `S1 helper parity OK`
- Synthetic S2 parity check:
  - copied the old S2 formula into a temporary script
  - compared loss, valid/invalid masks, and reprojection error against the new
    helper
  - result: `S2 helper parity OK`

## Non-Goals

- No new loss formula.
- No S1/S2 behavior unification beyond code sharing.
- No architecture changes.
- No new CLI flag.
- No preset change.
- No full training run in this step.

## Follow-Up

If future experiments suggest S1 should also use `loss_invalid_max_delta`, that
must be a separate explicit ablation because it changes historical S1 behavior.
