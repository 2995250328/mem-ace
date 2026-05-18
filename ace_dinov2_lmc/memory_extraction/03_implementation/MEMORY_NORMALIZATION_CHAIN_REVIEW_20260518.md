# Memory Normalization Chain Review - 2026-05-18

## Scope

This note answers two concrete questions:

1. How is normalization computed and stored in the memory file?
2. Are the saved normalization parameters actually used by the current training/eval paths?

The short answer is: the memory file stores normalization metadata, but the common C0 ACE-G presets currently load BSE memory back in world coordinates and intentionally drop `normalization_mu` / `normalization_sigma` from the active training path. The saved params are still used by the explicit normalized BSE path and by the C1 reference-normalized path.

## Memory File Contents

The extended memory writer computes a scalar scene normalizer from the final merged world point cloud:

```text
mu_scene = mean(points_world)
sigma_scene = sqrt(mean(mean((points_world - mu_scene)^2, over N), over xyz))
points_norm = (points_world - mu_scene) / sigma_scene
```

Implementation references:

- `memory_extraction/run_memory_extraction.py:3210`: recomputes `mu_scene` / `sigma_scene` after global merge.
- `memory_extraction/run_memory_extraction.py:3426`: reconstructs `points_world` from `points_norm`.
- `memory_extraction/run_memory_extraction.py:3433`: saves both normalized and world-space aliases.

Important saved fields:

| Field | Space | Meaning |
| --- | --- | --- |
| `points`, `points_norm` | normalized scene space | `(points_world - normalization_mu) / normalization_sigma` |
| `points_world` | world | raw metric scene coordinates |
| `pooled_points` | world | legacy training alias, intentionally world-space |
| `normalization_mu`, `mu` | world | point-cloud centroid used for normalization |
| `normalization_sigma`, `sigma` | scalar | RMS scene scale |
| `scene_center` | world | mean camera center, used as ACE/LMC anchor |
| `scene_center_cam` | world | selected-memory-view camera-center mean |
| `points_ref` | reference camera | C1 diagnostic/target field |
| `points_ref_norm` | reference-normalized | C1 normalized target field |
| `normalization_ref.mu_ref` / `sigma_ref` | reference camera | C1 reference-space normalizer |

`scene_center` and `normalization_mu` are not the same concept. `normalization_mu` is the point-cloud centroid. `scene_center` is the mean camera center kept for ACE-compatible head anchoring and geometry checks.

## C1 Reference Normalization

C1 adds a separate coordinate transform before normalization:

```text
points_ref = (points_world - C_ref) @ R_ref
points_ref_norm = (points_ref - mu_ref) / sigma_ref
```

The selected reference is `all_poses[0]` after memory-view ordering. The memory stores both `T_ref_c2w_world` and `T_world_to_ref`, plus `scene_center_ref` and `scene_center_ref_norm`.

Runtime C1 recovery in training:

```text
head output: points_ref_norm
points_ref = output * (sigma_ref / alpha) + mu_ref
points_world = R_ref @ points_ref + C_ref
if full-pipeline normalization is active:
    training_world = (points_world - normalization_mu) / normalization_sigma
```

Implementation references:

- `memory_extraction/run_memory_extraction.py:3282`: `_compute_points_ref_norm`.
- `memory_extraction/run_memory_extraction.py:3292`: world to reference transform.
- `memory_extraction/run_memory_extraction.py:3355`: writes `points_ref` / `points_ref_norm`.
- `trainer_dinov2_lmc.py:174`: builds runtime C1 contract.
- `trainer_dinov2_lmc.py:268`: recovers C1 head output for reprojection loss.
- `test_ace_dinov2_lmc.py:163`: recovers C1 predictions to raw world coordinates for DSAC/eval.

`--c1_ref_norm_alpha` is a weaker-normalization ablation. At load time it multiplies `points_ref_norm` and `scene_center_ref_norm`; recovery divides by `alpha`, so the metric world coordinates remain consistent.

## What Training Actually Uses

The decisive logic is in `utils_lmc.load_memory_features()`.

### Current common C0 ACE-G presets

The current training presets set `bse_denorm_to_world=True`, for example `train_ace_dinov2_lmc.py:69` and `train_ace_dinov2_lmc.py:92`.

For C0 BSE memory with `points_world` present, the loader does:

```text
pooled_points = points_world
scene_center = saved world camera-mean scene_center
normalization_mu = None
normalization_sigma = None
```

References:

- `utils_lmc.py:353`: BSE world-points path.
- `utils_lmc.py:366`: drops active normalizer by setting `norm_mu_out = None`.
- `trainer_dinov2_lmc.py:1217`: no normalizer means raw world coordinates and raw depth thresholds.

So for the current C0 low-risk ACE-G path, the memory file contains `normalization_mu` / `normalization_sigma`, but training does not use them. This is intentional: it makes BSE memory behave like pooled world-coordinate memory and isolates BSE/ASB effects from full coordinate normalization.

### Explicit normalized C0 BSE path

If `bse_denorm_to_world=False` and the memory is not `pool_mode=pooled`, the loader does:

```text
pooled_points = points_norm
scene_center = zeros(3)
normalization_mu = saved mu
normalization_sigma = saved sigma
```

Then trainer activates full-pipeline normalization:

- `coord_mu` / `coord_sigma` are set from memory.
- `depth_min`, `depth_max`, `depth_target` are divided by `sigma`.
- training buffer inverse-pose translations are normalized.
- head mean is aligned to zero scene center.
- checkpoint stores `normalization_mu` / `normalization_sigma`.
- eval denormalizes predicted scene coordinates by `pred_world = pred_norm * sigma + mu`.

References:

- `utils_lmc.py:410`: normalized BSE load branch.
- `trainer_dinov2_lmc.py:1225`: active normalized training setup.
- `trainer_dinov2_lmc.py:2572`: buffer pose normalization.
- `trainer_dinov2_lmc.py:2192`: normalized head mean handling.
- `test_ace_dinov2_lmc.py:479`: eval-time de-normalization for non-C1 normalized checkpoints.

This is the "training-chain normalization interface" that was previously tested and found slightly worse than no normalization.

### C1 path

C1 is handled before `bse_denorm_to_world`, so a C1 memory with `points_ref_norm` uses reference-normalized memory even if the preset has `bse_denorm_to_world=True`.

References:

- `utils_lmc.py:283`: C1 branch has priority.
- `utils_lmc.py:294`: `pooled_points = points_ref_norm`.
- `utils_lmc.py:330`: keeps `normalization_mu` / `normalization_sigma` active.
- `trainer_dinov2_lmc.py:237`: C1 output space becomes `points_ref_norm`.

This means C1 normalized runs exercise two related but distinct mechanisms:

1. reference-normalized head target (`points_ref_norm`);
2. full-pipeline world normalization for reprojection loss if memory `normalization_mu/sigma` are passed through.

Eval recovers C1 output directly to raw world coordinates, so it does not additionally apply `normalization_mu/sigma` after C1 recovery.

## Other "Normalization" Knobs

These are separate from target-coordinate normalization:

| Knob | Location | Effect |
| --- | --- | --- |
| `--pe_normalize_input` | `ace_compressor.py:34` | Divides Fourier PE input by per-forward `coords.std(dim=1)`. This only affects compressor positional encoding. It is not the saved memory normalizer. |
| `--lmc_fusion_geometry_mode value_only_norm/geokey_norm` | `ace_fusion.py:127` | Divides memory latent coordinates by `fusion_scene_scale` before fusion PE. |
| `--lmc_fusion_scene_scale_source memory_points_p95` | `trainer_dinov2_lmc.py:95` | Computes scene scale as P95 radius of `pooled_points - scene_center`. This is runtime geometry scaling, not target normalization. |

These knobs can improve geometry feature scale stability without changing the coordinate target/loss space.

## Evidence From Existing Runs

The historical result is consistent with the code reading:

- Current repaired C0 single-memory baseline remains stronger than same-recipe C1 on `scene2a`: `memory_extraction/03_implementation/PROGRESS_reference_consistent_memory.md` records repaired C0 `pct5=76.65`, median translation `3.0150 cm`, and states the completed C1 run does not beat repaired C0.
- The 2026-05-08 C1 aux-ref pair was negative for expansion: mean `acc5=71.98` versus control `73.54`, median translation `3.4068 cm` versus `3.3975 cm`.
- The C1 logs confirm the normalized path was actually active: `utils_lmc` logs `C1 REF-NORM path`, and trainer logs `Buffer poses normalized: sigma=3.3494`.

Interpretation: a small degradation from full/normalized-coordinate training is expected and does not prove the math is wrong. Normalization changes the numerical target scale and loss conditioning. It helps multi-scene scale balance conceptually, but can slightly compress single-scene fine metric precision unless compensated by weaker normalization (`alpha > 1`), auxiliary metric loss, or a better loss schedule.

## Current Mental Model

Use this decision table when reading or launching runs:

| Memory/run mode | Active memory coords | Active saved `mu/sigma`? | Eval recovery |
| --- | --- | --- | --- |
| `pool_mode=pooled` | world | no | raw world |
| C0 BSE + `bse_denorm_to_world=True` | world / `points_world` | no | raw world |
| C0 BSE + `bse_denorm_to_world=False` | `points_norm` | yes | `pred * sigma + mu` |
| C1 + `points_ref_norm` | `alpha * points_ref_norm` | yes for training loss chain; `normalization_ref` for C1 recovery | `points_ref_norm -> points_ref -> world` |
| fusion A1/A2 | whatever memory coords above, plus scene-scale PE input | no target-space change | same as run mode |

Practical conclusion: for the current default C0 ACE-G baselines, the stored memory normalizer is metadata/fallback, not an active training signal. For normalized ablations and C1, it is active and should be treated as part of the checkpoint contract.

## Follow-Up Checks Worth Keeping

- When comparing "normalized vs unnormalized", log the loader branch explicitly (`BSE WORLD-POINTS`, `BSE normalized points`, or `C1 REF-NORM`) and do not infer from memory contents alone.
- For C1 reports, always record `c1_ref_norm_alpha`, `normalization_ref.sigma_ref`, and world `normalization_sigma`; they affect different transforms.
- If revisiting normalized training, prefer a small matrix on one scene: C0 world, C0 normalized, C1 alpha=1, C1 alpha=2, optionally C1 + aux metric loss. The metric to watch is `acc5` and median translation, while ensuring `acc25` does not collapse.
