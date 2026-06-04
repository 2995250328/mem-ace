# SquareBench Stage2 Global Recovery Log

This is the running log for all attempts related to fixing the poor `wayspots_squarebench` Stage2 behavior. Keep appending new trials here before turning any result into paper-facing claims.

## Current Diagnosis

`wayspots_squarebench` is not a data or Stage1-memory failure case. The local route is strong:

- ACE baseline is strong on SquareBench.
- ACE-FCN-LMC Stage1 is slightly better than ACE at Acc5.
- ACE-FCN memory sanity passed: pooled points 36094, cosine margin 0.677, nn3d/random3d 0.143.
- Raw Stage2 `glace_concat` collapses badly on SquareBench.
- GLACE baseline itself is worse than ACE on this scene, so blindly injecting GLACE global features amplifies a weak global prior.

Working interpretation: the problem is negative transfer from GLACE global conditioning, not bad sparse depth, bad ACE-FCN memory, or a broken Stage1 local predictor.

## Fixed Reference Results

All numbers are post-train aggregated unless noted otherwise.

| Scene | Method | Acc50 | Acc25 | Acc10 | Acc5 | Median |
|---|---|---:|---:|---:|---:|---|
| SquareBench | ACE | 99.653 | 99.480 | 69.844 | 52.166 | 0.689 deg / 4.816 cm |
| SquareBench | GLACE | 94.974 | 94.801 | 67.244 | 42.634 | 0.654 deg / 5.855 cm |
| SquareBench | ACE-FCN-LMC Stage1 | 99.827 | 99.310 | 70.020 | 53.030 | 0.655 deg / 4.786 cm |
| SquareBench | Stage2 raw `glace_concat` | 78.683 | 78.680 | 49.740 | 32.580 | 0.768 deg / 10.098 cm |
| Bears | ACE | 97.931 | 95.172 | 86.207 | 73.276 | 1.111 deg / 3.506 cm |
| Bears | GLACE | 97.414 | 94.310 | 87.069 | 77.069 | 1.011 deg / 3.080 cm |
| Bears | ACE-FCN-LMC Stage1 | 97.069 | 94.830 | 91.030 | 77.070 | 1.059 deg / 3.214 cm |
| Bears | Stage2 raw `glace_concat` | 98.103 | 97.760 | 93.280 | 83.280 | 0.946 deg / 3.072 cm |

The tension is clear: raw global helps Bears but breaks SquareBench.

## Attempt Summary

| Family | Variant | SquareBench Acc25 / Acc10 / Acc5 | Bears Acc25 / Acc10 / Acc5 | Outcome |
|---|---|---:|---:|---|
| Scalar gate | `glace_g001_learn` | 77.99 / 49.39 / 32.76 | 97.07 / 92.93 / 81.72 | Bears ok, SquareBench still collapses. |
| Scalar gate bounded | `glace_g001_max001_learn` | 99.83 / 70.19 / 51.65 | 95.17 / 90.86 / 79.48 | Safe for SquareBench, weak for Bears. |
| Scalar gate bounded | `glace_g001_max01_learn` | 91.51 / 59.10 / 37.44 | 96.21 / 92.07 / 82.41 | Better Bears, still damages SquareBench. |
| Consistency | `glace_g001_max01_cons01` | 85.79 / 55.63 / 35.70 | 98.62 / 93.10 / 82.24 | Does not protect SquareBench. |
| Consistency | `glace_g001_max01_cons1` | 87.69 / 57.54 / 36.05 | 97.24 / 91.90 / 82.59 | Still poor on SquareBench. |
| Guard loss | `glace_g001_max01_guard01` | 89.08 / 53.38 / 32.41 | 97.41 / 92.76 / 82.24 | Guard did not prevent negative transfer. |
| Guard loss | `glace_g001_max01_guard1` | 89.60 / 57.89 / 38.65 | 97.24 / 93.10 / 83.45 | Bears good, SquareBench bad. |
| Manual select | SquareBench safe max=0.001, Bears strong max=0.1 | 99.65 / 71.58 / 48.70 | 97.59 / 92.41 / 81.90 | Works only as manual per-scene selection; not paper-clean. |
| Residual identity | `glace_residual_identity` | 99.48 / 70.36 / 52.69 | n/a | Identity path matches Stage1, proving eval/training reconstruction can be safe. |
| Residual unified | smoke / basefix | 99.48 / 70.54 / 52.51 | n/a | Safe but no real global benefit. |
| Residual unified | it12 | 99.31 / 69.67 / 53.03 | 96.03 / 90.69 / 78.10 | Degenerates to local/Stage1 behavior; global effectively unused. |
| FiLM | `glace_film_unified` | 99.83 / 70.88 / 50.61 | 95.17 / 89.66 / 77.93 | Safe-ish, but worse than desired and weak on Bears. |
| FiLM control | `random_film` | 99.65 / 69.84 / 48.18 | n/a | Similar to GLACE FiLM; little evidence GLACE semantics are used. |
| FiLM control | `zero_film` | 99.31 / 70.36 / 50.78 | n/a | Comparable to GLACE/random FiLM; reinforces degeneration conclusion. |
| Concat gate + L1 | `glace_concat_gate_unified` | 86.66 / 57.37 / 34.66 | 96.90 / 91.03 / 81.55 | Bears keeps useful global gain; SquareBench still collapses. L1 alone is not a unified solution. |

## Detailed Notes

### 1. Raw `glace_concat` is the original failure

Raw Stage2 uses the Stage1 local stack with GLACE global feature concatenated into the final head. It improves Bears substantially, but SquareBench drops from Stage1 Acc5 53.03 to 32.58 and Acc10 from 70.02 to 49.74. This is the failure this log tracks.

### 2. Scalar gate confirms the tradeoff

The bounded scalar gate proved that SquareBench can be protected when the maximum global contribution is tiny. With `gate_max=0.001`, SquareBench returns near Stage1. But Bears loses the larger global benefit. With `gate_max=0.1`, Bears improves while SquareBench collapses again. This means a single fixed scalar cap is not enough unless it learns real reliability, not just amplitude.

### 3. Consistency and guard losses did not solve it

Stage1 consistency and pixel guard losses were meant to stop Stage2 from moving too far from the local teacher. In practice, the tried weights did not protect SquareBench. Some earlier reliability/adaptive variants also became numerically unstable, with logs showing `Valid: 0.0%`, `naninf=4096`, and NaN gate diagnostics. Current conclusion: the implemented consistency/guard route is not a clean solution in its current form.

### 4. Residual identity/basefix fixed implementation safety but not usefulness

The residual identity check initially hit a no-grad failure when every trainable/global residual path was disabled. After the basefix, identity evaluation matched Stage1-like performance on SquareBench. This was important because it proved the Stage2 wrapper itself can be safe. However, the full residual-unified run mostly collapsed to the local predictor: SquareBench is safe, Bears loses global gains. Therefore residual-unified is a good sanity architecture but not yet a useful global-conditioning method.

### 5. FiLM did not extract useful GLACE semantics

`glace_film_unified` was introduced to condition the frozen local representation via a zero-initialized FiLM adapter instead of raw concat. It preserved SquareBench better than raw concat, but random/zero controls were close. Bears also regressed relative to raw Stage2. The likely conclusion is that this FiLM implementation behaves mostly like a conservative capacity/control path, not like effective GLACE semantic conditioning.

### 6. Current active implementation: concat gate with L1

The current active direction is returning to `glace_concat`, but adding a learnable bounded scalar gate with explicit L1 regularization:

```bash
--ace_lmc_global_gate_init 0.01
--ace_lmc_global_gate_learnable True
--ace_lmc_global_gate_max 0.1
--ace_lmc_global_gate_l1_weight 0.001
```

Implemented pieces:

- `options_dinov2_lmc.py`: added `--ace_lmc_global_gate_l1_weight`.
- `trainer_dinov2_lmc.py`: added scalar gate L1 computation and logging (`gScalar`, `gScalarL1`) in Stage2 / ACE-G paths.
- `scripts/run_squarebench_stage2_global_gate_matrix.sh`: added `glace_concat_gate_unified` variant and defaulted the script to it.

Completed result on 2026-06-04:

| Scene | Acc25 | Acc10 | Acc5 | Median | Output |
|---|---:|---:|---:|---|---|
| Bears | 96.90 | 91.03 | 81.55 | 0.91 deg / 2.75 cm | `/data/xwh/ace_dinov2_lmc/04_evaluation/wayspots_ace_fcn_lmc_suite/20260530_181745/stage2_concat_gate_l1_bears_it12_glace_concat_gate_unified/.../post_train_eval.txt` |
| SquareBench | 86.66 | 57.37 | 34.66 | 0.71 deg / 7.97 cm | `/data/xwh/ace_dinov2_lmc/04_evaluation/wayspots_ace_fcn_lmc_suite/20260530_181745/stage2_concat_gate_l1_squarebench_it12_glace_concat_gate_unified/.../post_train_eval.txt` |

Interpretation: explicit scalar-gate L1 regularization is not enough. It preserves a moderate Bears improvement, but SquareBench remains far below Stage1 and ACE. The failure is therefore not solved by a single learnable bounded scalar gate with weak sparsity pressure.

## Result Interpretation So Far

1. SquareBench is a valid negative-transfer stress case, not a broken dataset case.
2. GLACE global information is scene-dependent: helpful on Bears, harmful on SquareBench.
3. A manually selected gate cap can make each scene look reasonable, but this is not a principled paper method.
4. Conservative residual/FiLM variants prevent catastrophic damage by suppressing global information, but then they do not deliver the original Stage2 benefit.
5. The concat-gate-L1 result confirms that a weak scalar sparsity prior is insufficient; the next useful experiment must learn reliability from a richer signal than a single global scalar amplitude.

## Chronological Updates

### 2026-06-04 - `glace_concat_gate_unified`

Command / variant:

```bash
SCENE=wayspots_bears MATRIX_SUBDIR=stage2_concat_gate_l1_bears_it12 VARIANTS_STR='glace_concat_gate_unified' ...
SCENE=wayspots_squarebench MATRIX_SUBDIR=stage2_concat_gate_l1_squarebench_it12 VARIANTS_STR='glace_concat_gate_unified' ...
```

Result:

| Scene | Acc50 | Acc25 | Acc10 | Acc5 | Median | Notes |
|---|---:|---:|---:|---:|---|---|
| Bears | n/a | 96.90 | 91.03 | 81.55 | 0.91 deg / 2.75 cm | Better than Stage1 Acc5 77.07, but below raw Stage2 Acc5 83.28. |
| SquareBench | n/a | 86.66 | 57.37 | 34.66 | 0.71 deg / 7.97 cm | Still much worse than Stage1 Acc5 53.03 and ACE Acc5 52.17. |

Conclusion:

- This route does not solve the unified reliability problem.
- The scalar gate can still admit harmful GLACE conditioning on SquareBench even with L1.
- Treat this as another failed automatic-selection attempt, not as a final method.

## Paths

Main suite root:

```text
/data/xwh/ace_dinov2_lmc/04_evaluation/wayspots_ace_fcn_lmc_suite/20260530_181745
```

Important result dirs:

```text
stage2_global_gate_matrix_squarebench_gatefix
stage2_global_gate_matrix_squarebench_gatefix_bounded
stage2_global_gate_matrix_squarebench_gatefix_bounded01
stage2_global_reliability_squarebench_cons01
stage2_global_reliability_squarebench_cons1
stage2_global_reliability_squarebench_guard01
stage2_global_reliability_squarebench_guard1
stage2_reliability_select_squarebench_safe
stage2_residual_identity_check_squarebench
stage2_residual_unified_squarebench_basefix_smoke
stage2_residual_unified_squarebench_basefix_it12
stage2_film_squarebench_it12
stage2_film_squarebench_controls_it12
stage2_concat_gate_l1_squarebench_it12
```

Bears comparison dirs:

```text
stage2_global_gate_matrix_bears_gatefix
stage2_global_gate_matrix_bears_gatefix_bounded
stage2_global_gate_matrix_bears_gatefix_bounded01
stage2_global_reliability_bears_cons01
stage2_global_reliability_bears_cons1
stage2_global_reliability_bears_guard01
stage2_global_reliability_bears_guard1
stage2_reliability_select_bears_strong
stage2_residual_unified_bears_basefix_it12
stage2_film_bears_it12
stage2_concat_gate_l1_bears_it12
```

## Append Template

Add new results in this format:

````markdown
### YYYY-MM-DD - <short variant name>

Command / variant:

```bash
<command>
```

Result:

| Scene | Acc50 | Acc25 | Acc10 | Acc5 | Median | Notes |
|---|---:|---:|---:|---:|---|---|
| SquareBench |  |  |  |  |  |  |
| Bears |  |  |  |  |  |  |

Conclusion:

- ...
````
