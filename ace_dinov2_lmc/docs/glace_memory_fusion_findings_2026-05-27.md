# GLACE Memory Fusion Findings - 2026-05-27

This note records the observed behavior of the current GLACE memory-fusion design on `wayspots_bears`.

## Runs observed

### 1. `guarded_lowlr_r2_v1`
- Post-train aggregated median: `1.02 deg, 3.15 cm`
- `25cm/5deg`: `95.34%`
- `10cm/5deg`: `89.31%`
- `5cm/5deg`: `77.24%`
- `2cm/2deg`: `12.93%`
- `1cm/1deg`: `1.03%`
- Avg time: `169.69 ms`

### 2. `probe_ultralow_drift_v1`
- Post-train aggregated median: `1.02 deg, 3.11 cm`
- `25cm/5deg`: `95.86%`
- `10cm/5deg`: `88.79%`
- `5cm/5deg`: `76.72%`
- `2cm/2deg`: `13.28%`
- `1cm/1deg`: `1.72%`
- Avg time: `158.57 ms`

## What this suggests

1. The current fusion path does not consistently improve the main metric family relative to the existing GLACE baseline.
2. Lowering the learning rate and tightening the guard did not produce a clear, stable gain in `pct5`.
3. The compressed-memory signal is influencing the model, but the observed effect is not translating into stronger pose accuracy in a reliable way.
4. The current design likely needs a different fusion strategy or a different way of injecting memory into the GLACE decoder path.

## Code paths involved

- `TrainerACEDINOv2LMC._create_regressor()`
- `TrainerACEDINOv2LMC.__init__()`
- `TrainerACEDINOv2LMC._setup_s2_optimizer_and_schedule()`
- `TrainerACEDINOv2LMC._run_s2_polish_phase()`
- `TrainerACEDINOv2LMC._training_step_ace_g()`

## Reference files

- `run_config.json`
- `best_checkpoint_meta.json`
- `training_full_log.txt`
- `best_K64_it28_wayspots_bears_glace_lmc_frozen_guard_active_eval_log.txt`

## Decision status

- Do not continue scaling the current fusion design by just increasing iterations.
- Revisit the fusion / injection design before running the next experiment family.
