# Review Report: SquareBench Stage2 Global Reliability Experiments

## 1. Executive Summary

The SquareBench recovery experiments show that `wayspots_squarebench` is a valid negative-transfer stress case rather than a broken data, memory, or Stage1 case. The local route is strong: ACE is already competitive, ACE-FCN-LMC Stage1 slightly improves over ACE at Acc5, and ACE-FCN memory sanity checks pass. The failure appears when Stage2 injects GLACE global features through `glace_concat`.

The central finding is scene-dependent global reliability:

- On Bears, raw Stage2 GLACE global conditioning improves Acc5 from Stage1 77.070 to 83.280.
- On SquareBench, the same raw Stage2 conditioning drops Acc5 from Stage1 53.030 to 32.580.

Therefore, the problem is not simply how much global feature to inject. The problem is that the model currently lacks a reliable mechanism to decide when GLACE global information is trustworthy.

## 2. Source and Metric Provenance

Primary source:

```text
stage2_global_reliability/02_review/squarebench_recovery_log.md
```

The log states that all fixed reference numbers are post-train aggregated unless otherwise noted. The main suite root for the underlying run directories is:

```text
/data/xwh/ace_dinov2_lmc/04_evaluation/wayspots_ace_fcn_lmc_suite/20260530_181745
```

Important SquareBench result directories listed by the log:

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

Important Bears comparison directories listed by the log:

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

Metrics are Acc50, Acc25, Acc10, Acc5, and median rotation/translation error. Higher Acc values are better; lower median errors are better.

## 3. Fixed Reference Results

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

The key deltas are:

- SquareBench raw Stage2 vs Stage1: Acc5 drops by 20.450 points.
- Bears raw Stage2 vs Stage1: Acc5 improves by 6.210 points.

This opposite behavior is the core reliability problem.

## 4. Experiment Matrix Summary

| Family | Variant | SquareBench Acc25 / Acc10 / Acc5 | Bears Acc25 / Acc10 / Acc5 | Outcome |
|---|---|---:|---:|---|
| Scalar gate | `glace_g001_learn` | 77.99 / 49.39 / 32.76 | 97.07 / 92.93 / 81.72 | Bears remains acceptable, but SquareBench still collapses. |
| Scalar gate bounded | `glace_g001_max001_learn` | 99.83 / 70.19 / 51.65 | 95.17 / 90.86 / 79.48 | Safe for SquareBench, but weak for Bears. |
| Scalar gate bounded | `glace_g001_max01_learn` | 91.51 / 59.10 / 37.44 | 96.21 / 92.07 / 82.41 | Better for Bears, but still damages SquareBench. |
| Consistency | `glace_g001_max01_cons01` | 85.79 / 55.63 / 35.70 | 98.62 / 93.10 / 82.24 | Does not protect SquareBench. |
| Consistency | `glace_g001_max01_cons1` | 87.69 / 57.54 / 36.05 | 97.24 / 91.90 / 82.59 | Still poor on SquareBench. |
| Guard loss | `glace_g001_max01_guard01` | 89.08 / 53.38 / 32.41 | 97.41 / 92.76 / 82.24 | Guard loss does not prevent negative transfer. |
| Guard loss | `glace_g001_max01_guard1` | 89.60 / 57.89 / 38.65 | 97.24 / 93.10 / 83.45 | Bears is good, SquareBench remains bad. |
| Manual select | SquareBench safe max=0.001, Bears strong max=0.1 | 99.65 / 71.58 / 48.70 | 97.59 / 92.41 / 81.90 | Works as manual per-scene selection, but is not paper-clean. |
| Residual identity | `glace_residual_identity` | 99.48 / 70.36 / 52.69 | n/a | Identity path matches Stage1 and proves wrapper safety. |
| Residual unified | smoke / basefix | 99.48 / 70.54 / 52.51 | n/a | Safe but no real global benefit. |
| Residual unified | it12 | 99.31 / 69.67 / 53.03 | 96.03 / 90.69 / 78.10 | Degenerates to local/Stage1 behavior; global is effectively unused. |
| FiLM | `glace_film_unified` | 99.83 / 70.88 / 50.61 | 95.17 / 89.66 / 77.93 | Safe-ish, but weak on Bears and below desired result. |
| FiLM control | `random_film` | 99.65 / 69.84 / 48.18 | n/a | Similar to GLACE FiLM, giving little evidence that GLACE semantics are used. |
| FiLM control | `zero_film` | 99.31 / 70.36 / 50.78 | n/a | Comparable to GLACE/random FiLM; reinforces degeneration conclusion. |
| Concat gate + L1 | `glace_concat_gate_unified` | 86.66 / 57.37 / 34.66 | 96.90 / 91.03 / 81.55 | Bears keeps some global gain, but SquareBench still collapses. L1 alone is not a unified solution. |

## 5. Analysis by Attempt Family

### 5.1 Raw `glace_concat`

Raw Stage2 is the original failure mode. It concatenates GLACE global features into the final head on top of the Stage1 local stack. This improves Bears but heavily damages SquareBench. The important conclusion is that raw global conditioning is not uniformly bad; it is scene-dependent.

### 5.2 Scalar Gates

The scalar gate family proves the tradeoff. A tiny cap such as `gate_max=0.001` protects SquareBench, but sacrifices the larger Bears gain. A larger cap such as `gate_max=0.1` recovers Bears but reintroduces SquareBench collapse. Therefore, a fixed scalar amplitude cap cannot solve the unified reliability problem unless it learns reliability from additional signals.

### 5.3 Consistency and Guard Losses

Stage1 consistency and pixel guard losses were intended to keep Stage2 close to the local teacher. In the tested configurations, they do not protect SquareBench. Earlier adaptive/reliability variants also showed numerical instability with `Valid: 0.0%`, `naninf=4096`, and NaN gate diagnostics. The current consistency/guard route is not clean enough to become the final method without a redesign.

### 5.4 Residual Identity and Residual Unified

Residual identity/basefix experiments are valuable implementation checks. They show that the Stage2 wrapper can safely reconstruct Stage1-like behavior when the residual/global path is disabled or effectively zero. However, the full residual-unified run mostly degenerates to the local predictor: SquareBench stays safe, but Bears loses the global benefit. This architecture solves the safety problem but not the usefulness problem.

### 5.5 FiLM and FiLM Controls

The FiLM path conditions the frozen local representation through a zero-initialized adapter instead of raw concat. It protects SquareBench better than raw concat, but random and zero controls are close, and Bears performance is weak. This suggests the current FiLM implementation acts mostly as a conservative control path rather than an effective GLACE semantic-conditioning path.

### 5.6 Concat Gate with L1

The latest `glace_concat_gate_unified` experiment returns to concat but adds a learnable bounded scalar gate with explicit L1 regularization:

```bash
--ace_lmc_global_gate_init 0.01
--ace_lmc_global_gate_learnable True
--ace_lmc_global_gate_max 0.1
--ace_lmc_global_gate_l1_weight 0.001
```

Completed result:

| Scene | Acc25 | Acc10 | Acc5 | Median |
|---|---:|---:|---:|---|
| Bears | 96.90 | 91.03 | 81.55 | 0.91 deg / 2.75 cm |
| SquareBench | 86.66 | 57.37 | 34.66 | 0.71 deg / 7.97 cm |

This preserves a moderate Bears improvement but leaves SquareBench far below Stage1 and ACE. The weak scalar sparsity prior is insufficient; a single learnable bounded scalar gate still admits harmful GLACE conditioning.

## 6. Failed or Insufficient Routes

The following routes should not receive more broad sweeps in their current form:

1. **Single scalar gate or scalar gate max sweeps.** Manual per-scene selection works, but it is not a principled paper method.
2. **Simple Stage1 consistency or guard loss.** The tested losses do not prevent SquareBench collapse and have shown instability in earlier variants.
3. **Conservative residual or FiLM as the final method.** They protect SquareBench mostly by suppressing global information, which removes the original Bears benefit.
4. **Concat gate with weak L1.** It keeps part of the Bears gain but still fails badly on SquareBench.

These routes remain useful as ablations that motivate reliability-aware global conditioning.

## 7. Scientific Interpretation

The experiments support the following paper-facing interpretation:

1. SquareBench is a genuine negative-transfer stress case.
2. GLACE global information is helpful on Bears but harmful on SquareBench.
3. ACE-FCN-LMC Stage1 is a strong and stable local memory route.
4. Raw GLACE global concat is not safe as a unified method.
5. Existing scalar or conservative controls do not solve the central reliability problem.

The correct research problem is therefore not generic feature fusion. It is scene- or sample-dependent global reliability estimation.

## 8. Recommended Next Directions

### 8.1 Local-Global Disagreement Gate

Use disagreement between local-only and global-conditioned predictions as the reliability signal. Candidate signals include:

- Stage1 local reprojection error.
- Difference between Stage1 coordinates and Stage2 coordinates.
- Global-conditioned residual norm.
- Per-pixel reprojection delta.
- Local-only validity or confidence.

If global conditioning strongly disagrees with a confident local predictor, down-weight or reject the global path.

### 8.2 Dual-Head Selection or Mixture

Maintain two prediction paths:

1. Local-only Stage1/Stage2 head.
2. Local+global head.

Then select or mix them using a learned or non-learned reliability rule. The selection can be per-scene, per-image, or per-pixel. This directly matches the Bears/SquareBench split and avoids forcing one fusion behavior onto all scenes.

### 8.3 RANSAC/Post-Hoc Reliability Selection

For each query, run both local-only and local+global predictions, then choose the pose using SCR-native confidence signals:

- RANSAC inlier count.
- Pose score.
- Reprojection residual.
- Coordinate validity ratio.

This is likely the most practical next experiment. It aligns with the SCR pipeline and may preserve Bears gains while protecting SquareBench, at the cost of additional inference time.

## 9. Implications for the Final Paper

SquareBench should not be hidden. It is valuable as a reviewer-facing negative-transfer analysis. It proves that:

- Stage1 local memory is strong.
- The memory/data path is not broken.
- Global conditioning can be harmful when global priors are weak.
- Reliability control is necessary for a paper-clean Stage2 method.

If the final method does not solve reliability robustly, raw Stage2 `glace_concat` should be presented as an ablation rather than the unified method. The stronger narrative is:

> LMC Stage1 provides a stable local memory compressor. GLACE global conditioning can provide additional gains when reliable, but it requires explicit reliability selection or gating to avoid negative transfer.

## 10. Recommended Short-Term Experiment List

1. **Dual inference with RANSAC inlier selection.** Compare local-only vs local+global per query on Bears and SquareBench.
2. **Non-learned local-global disagreement gate.** Use coordinate delta, residual norm, or reprojection delta as the reliability criterion before adding learnable complexity.
3. **Per-image gate instead of scene scalar gate.** Scene-level scalar gates are too coarse for a paper-clean solution.
4. **Ablation consolidation.** Keep scalar gate, consistency, residual, FiLM, and concat-gate-L1 results as evidence that reliability is necessary.

## 11. Stage-2 Review Verdict

The review rejects scalar-amplitude control as the final reliability method. The next architecture stage should design a reliability mechanism based on local-global disagreement, dual-head selection, or RANSAC/post-hoc pose selection. The goal is to retain Bears' useful global gains while recovering SquareBench to Stage1-level or better performance.