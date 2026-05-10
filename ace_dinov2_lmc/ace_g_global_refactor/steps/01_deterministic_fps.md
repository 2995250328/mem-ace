# Step 01: Deterministic FPS

Status: done.

## Goal

Remove the largest stochastic source in ACE-G global compression: the random
first point in farthest point sampling. For the same memory and same checkpoint,
S1, S2, and test-time compression should select the same latent coordinates by
default.

## Implemented Measure

- `farthest_point_sampling()` accepts an explicit `start_policy`.
- Default policy is `farthest_from_center`.
- `legacy_random` remains available as an opt-in rollback path.
- CLI exposes `--lmc_fps_start_policy`.
- Training saves `lmc_fps_start_policy` into `lmc_config`.
- Evaluation reconstructs `GeoLMC` from checkpoint `lmc_config` and consumes
  the saved policy.

## Verified Code Points

- `/home/xwh/project/ace_depth/ace_compressor.py:43`
- `options_dinov2_lmc.py:537`
- `trainer_dinov2_lmc.py:1020`
- `trainer_dinov2_lmc.py:1033`
- `trainer_dinov2_lmc.py:1077`
- `test_ace_dinov2_lmc.py:287`

## Residual Note

Old checkpoints that were trained before this field existed do not store the
original sampled latent coordinates. They now default to deterministic
`farthest_from_center` during evaluation unless manually patched to use
`legacy_random`, but that still cannot recover the original random first point.
