# Step 03: S1 Sampled Loss Step Mode

Status: done.

## Goal

Make sampled S1 ReproLoss time-axis behavior explicit without changing the
legacy default behavior.

## Implemented Measures

- Added `--s1_loss_step_mode` with explicit choices:
  - `fixed_zero`: always use ReproLoss step 0.
  - `per_iter`: use the current S1 update index inside the current LMC
    iteration.
  - `global_monotonic`: use the trainer's monotonic global ReproLoss step.
- CLI default is `None`, which resolves to the legacy behavior:
  - regular profiles resolve to `fixed_zero`;
  - `lmc_profile=mapany_flow_v1` resolves to `global_monotonic`.
- Training logs both requested and resolved modes.
- Checkpoints record:
  - `s1_loss_step_mode`
  - `requested_s1_loss_step_mode`
- Both sampled S1 paths are covered:
  - S1 raw-buffer mode;
  - online encoder sampled mode.

## Non-Goals

- No reprojection/invalid-loss refactor.
- No change to full-map S1 scheduling.
- No default behavior change for existing commands.

## Verification Points

- `options_dinov2_lmc.py`: exposes `--s1_loss_step_mode`.
- `trainer_dinov2_lmc.py`: resolves legacy-compatible default, records config,
  logs mode, and routes sampled S1 loss through `_resolve_s1_sampled_loss_step()`.
