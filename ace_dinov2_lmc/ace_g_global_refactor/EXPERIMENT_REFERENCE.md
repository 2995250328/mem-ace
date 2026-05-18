# ACE-G Global Refactor Experiment Reference

Updated: 2026-05-18

Purpose: this is the reference document for all follow-up ACE-G global refactor
experiments. It records completed experiments, key metrics, baseline decisions,
and "do not rerun unchanged" rules. Before launching a new run, check this file
first, then append the new result here or link to the detailed comparison note.

Metric convention:
- `pct5` / `acc5`: percentage of test frames within 5 cm / 5 deg.
- `pct2` / `acc2`: percentage of test frames within 2 cm / 2 deg.
- `med_t`: median translation error in cm.
- `med_r`: median rotation error in degrees.
- When a row says "5-run median", it is the median over repeated post-train
  eval seeds/hypotheses as recorded in the source note.

## Current Baseline Decisions

| Decision | Status | Evidence | Do not repeat unchanged |
| --- | --- | --- | --- |
| Global-track baseline should be `forceglobal_s1_periter` | Accepted for next stage | 3090 completed true-global per-iter beats B0 on post-train `pct5`, `pct10_5`, `pct2`, `med_t`, and runtime | Do not rerun `forceglobal_fixedzero` except as a negative control |
| `s1_loss_step_mode=per_iter` is useful | Accepted | 3090 true-global and 4090 effective-local both improve the useful operating point | Do not go back to `fixed_zero` as default |
| A1 `value_only_norm` is a weak positive/mixed signal | Diagnostic only | Slightly improves `pct2`, attention is sharper, but no clear `pct5` win over accepted key2 baseline | Do not promote to default yet |
| A2 `geokey_norm` is negative as default | Rejected as default | Clear metric drop vs A1/A0 | Do not rerun A2 unchanged |
| C1 / aux-ref / alpha scaling did not beat controls on `scene2a` | Negative or inconclusive | Historical C1, alpha=2, and aux-ref results trail paired controls on `acc5` | Do not expand before a stronger paired win |

## Recent Scene2a C0_P4 Baseline/Fusion Matrix

Source notes:
- `04_evaluation/train_compare/COMPARE_combined_3090_4090_baseline_decision_20260517.md`
- `04_evaluation/train_compare/COMPARE_forceglobal_vs_B0_completed_20260517.md`
- `04_evaluation/train_compare/COMPARE_fusion_A0_A1_A2_runtime_diag_20260518.md`

Shared recent setup where applicable:
- Scene / variant: `scene2a/c0_p4`
- `lmc_mode=global`
- `lmc_key_slice_idx=2`
- `lmc_fps_start_policy=farthest_from_center`
- `post_train_hypotheses=256` for repeated eval rows

### 3090 Completed Baseline Triplet

Memory: `memory_extraction/04_evaluation/memory_extract/scene2a/40v_v0.05_bilinear_bse_ut0.02_noaug_asb_adaptive_pool1.0_rgate4_prepair8_sor_gm_l2/20260427_141401/memory_bse.pt`

Aggregation: 5-seed post-train median.

| ID | Run | Effective mode | S1 step | pct5 | pct10_5 | pct2 | pct1 | med_t | med_r | avg ms | Decision |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| B0-3090 | `B0_baseline` | local fallback | fixed_zero default | 80.93 | 94.55 | 30.35 | 7.39 | 2.8713 | 0.2837 | 167.66 | Legacy control |
| FGPI-3090 | `forceglobal_s1_periter` | true global | per_iter | 82.10 | 95.72 | 33.07 | 7.00 | 2.7977 | 0.2891 | 158.59 | Current global-track baseline |
| FGFZ-3090 | `forceglobal_fixedzero` | true global | fixed_zero | 74.32 | 92.22 | 29.18 | 7.00 | 2.9779 | 0.3169 | 162.20 | Negative control |

Do not rerun unchanged:
- `B0-3090`
- `FGFZ-3090`
- `FGPI-3090` unless a strict same-code regeneration is needed.

### 4090 Completed Baseline Triplet

Memory: `memory_extraction/04_evaluation/memory_extract/scene2a/40v_v0.05_bilinear_bse_ut0.02_noaug_asb_adaptive_pool1.0_rgate4_prepair8_sor_gm_l2/20260508_103423/memory_bse.pt`

Aggregation: completed post-train summaries in the combined note.

| ID | Run | Effective mode | S1 step | pct5 | pct10_5 | pct2 | pct1 | med_t | med_r | avg ms | Decision |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| B0-4090 | `4090_baseline_20260509` | local fallback | fixed_zero default | 75.49 | 93.39 | 29.96 | 6.23 | 2.8986 | 0.2992 | 101.75 | Legacy local fallback control |
| ELPI-4090 | `4090_s1_periter` | local fallback | per_iter | 80.16 | 94.94 | 28.40 | 7.39 | 2.8205 | 0.2892 | 136.83 | Per-iter useful, but not true global |
| FGFZ-4090 | `4090_forceglobal_fixedzero` | true global | fixed_zero | 76.65 | 94.94 | 26.46 | 6.61 | 3.0464 | 0.3204 | 107.99 | Negative/mixed control |
| FGPI-4090 | `4090_forceglobal_s1_periter` | true global | per_iter | 83.66 | 95.33 | 34.24 | 6.23 | 2.6233 | 0.2831 | 168.31 | Clean 4090 true-global per-iter reference |

FGPI-4090 source:
- `04_evaluation/train_compare/indoor6_full_baselines_4090_forceglobal_s1_periter/.../20260517_112300_...`
- Aggregation: 5-run post-train median over seeds `1305,2026,4242,7777,9001`.

Do not rerun unchanged:
- `B0-4090`
- `ELPI-4090`
- `FGFZ-4090`
- `FGPI-4090`

### Fusion Geometry A0/A1/A2

Memory: `memory_extraction/04_evaluation/memory_extract/scene2a/40v_v0.05_bilinear_bse_ut0.02_noaug_asb_adaptive_pool1.0_rgate4_prepair8_sor_gm_l2/20260508_103423/memory_bse.pt`

Shared contract:
- `lmc_mode=global`
- `lmc_auto_mode_by_visibility=False`
- `lmc_key_slice_idx=2`
- `s1_loss_step_mode=per_iter`

Important caveat:
- A0 used `batch_size=5120`.
- A1/A2 used `batch_size=10240`.
- Therefore A0 is a compatibility/control check, not a perfect train-control
  against A1/A2.

| ID | Fusion mode | Train batch | Eval protocol | hyp | pct5 | pct2 | med_t | med_r | Decision |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| A0-post | `value_only_raw` | 5120 | train post-eval single run | 64 | 79.77 | 33.46 | 2.7546 | 0.2682 | Compatibility baseline |
| A0-reeval | `value_only_raw` | 5120 | 5 repeated runs | 256 | 79.77 | 33.07 | 2.6086 | 0.2734 | Already repeated |
| A0-diag | `value_only_raw` | 5120 | runtime diag single run | 256 | 80.16 | 33.46 | 2.63 | 0.27 | Diagnostics complete |
| A1 | `value_only_norm` | 10240 | 5-run median | 256 | 80.54 | 35.41 | 2.6117 | 0.2803 | Keep as diagnostic/mixed signal |
| A2 | `geokey_norm` | 10240 | 5-run median | 256 | 77.04 | 28.02 | 3.0951 | 0.3062 | Reject as default |

Accepted key2 reference in same note:

| Reference | pct5 | pct2 | med_t | med_r |
| --- | ---: | ---: | ---: | ---: |
| accepted key2 baseline | 83.66 | 34.24 | 2.6233 | 0.2831 |

Runtime attention diagnostics:

| ID | Fusion mode | p_norm_std | Effective tokens read | Key read |
| --- | --- | ---: | --- | --- |
| A0 | `value_only_raw` | 3.4163 | about 57-58 | raw geometry scale is large |
| A1 | `value_only_norm` | 0.3874 | about 47-49 | scale normalization works and sharpens attention |
| A2 | `geokey_norm` | 0.3874 | about 44-51 | learned `key_geo_scale=-0.027288`; metrics drop |

Do not rerun unchanged:
- A0, A1, A2 on `scene2a/c0_p4`.
- If new code needs a smoke check, run only one diagnostic/eval command, not
  the whole A0/A1/A2 matrix.

Next related run:
- CPE: A1-style fusion with compressor PE scene-scale, not a repeat of A1.
  Use `--lmc_fusion_geometry_mode value_only_norm`
  and `--lmc_compressor_pe_scale_mode scene_scale`.

## Historical Memory / C0 / C1 Results

Source:
- `memory_extraction/03_implementation/KEY_RESULTS_reference_consistent_memory.md`

These rows use the historical "best scanned metric per experiment directory"
policy, not necessarily the same strict repeated-eval protocol as the recent
3090/4090 baseline triplets.

| ID | Scope | HW / status | acc25 | acc10 | acc5 | acc2 | med_r | med_t | Decision |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| scene2a_phase0_c0_rerun | single-memory C0 baseline rerun | 3090 validated | 100.00 | 91.05 | 73.15 | 24.51 | 0.3456 | 3.3740 | Superseded by repaired C0 |
| scene2a_phase4_c0_20260422_122243 | repaired single-memory C0 | 3090 validated | 99.61 | 91.44 | 76.65 | 26.46 | 0.3212 | 3.0150 | Strong historical C0 |
| scene2a_c1_20260426_113134 | single-memory C1 same recipe | 3090 validated | 99.61 | 91.05 | 71.98 | 23.35 | 0.3444 | 3.4055 | Negative vs repaired C0 |
| scene2a_c1_alpha2_20260502_183531 | C1 normalized target alpha=2 | 3090 exploratory | 99.61 | 91.83 | 73.54 | 25.68 | 0.3424 | 3.1096 | Did not beat same-date alpha=1 |
| scene2a_c1_alpha1_ctrl_20260502_184405 | C1 alpha=1 same-date control | 3090 exploratory | 99.61 | 91.44 | 75.49 | 24.90 | 0.3407 | 3.1607 | Control stronger than alpha=2 on acc5 |
| scene2a_c1_auxref01_detfps_20260508_000246 | C1 + deterministic FPS + aux_ref=0.1 | 4090 exploratory; interrupted before final post-train | 100.00 | 91.44 | 71.60 | 23.35 | 0.3540 | 3.2366 | Negative vs paired control |
| scene2a_c1_detfps_ctrl_20260508_000323 | C1 + deterministic FPS control | 4090 exploratory; interrupted before final post-train | 100.00 | 91.83 | 73.93 | 27.24 | 0.3580 | 3.1805 | Paired control stronger |
| scene1_c1_20260428_101453 | cross-scene single-memory C1 | 3090 validated | 97.00 | 89.86 | 70.84 | 26.78 | 0.5546 | 3.3743 | Historical reference |
| scene4a_c1_20260428_101453 | cross-scene single-memory C1 | 3090 validated | 100.00 | 96.20 | 75.32 | 38.61 | 0.6420 | 2.8059 | Historical reference |
| scene5_c1_20260429_184410 | cross-scene single-memory C1 | 3090 partial/local validated | 98.11 | 87.03 | 61.56 | 15.80 | 0.6706 | 4.3124 | Historical reference |
| scene6_c1_20260429_184410 | cross-scene single-memory C1 | 3090 validated/local summaries | 90.40 | 84.21 | 67.49 | 32.20 | 0.6256 | 3.3777 | Historical reference |
| scene3_fallback60_phase4_c0_20260423_111442 | diagnostic 60-view fallback C0 | 3090 diagnostic | 97.14 | 88.57 | 69.52 | 22.22 | 0.6379 | 3.3900 | Diagnostic reference |
| scene3_c1_20260426_122510 | diagnostic single-memory C1 | 3090 diagnostic | 98.10 | 89.52 | 68.89 | 25.40 | 0.6379 | 3.4885 | Diagnostic reference |
| scene3_cluster_ensemble_eval | query-time C0 cluster ensemble | 3090 validated | 99.05 | 92.06 | 68.89 | 29.84 | 0.6327 | 3.0805 | Ensemble reference |
| scene3_c1_cluster01_20260501_213155 | cluster-local C1 / ensemble host | 3090 validated | 97.78 | 89.21 | 70.16 | 25.40 | 0.6487 | 3.2640 | Best listed scene3 C1 cluster row |
| scene3_c1_cluster02_20260501_213200 | cluster-local C1 | 3090 validated | 96.83 | 87.62 | 67.94 | 24.44 | 0.6517 | 3.3525 | Historical reference |

Do not rerun unchanged:
- Same-recipe `scene2a` C1.
- `scene2a` C1 alpha=2.
- `scene2a` aux-ref 0.1 unless a strict repeated-eval pair first shows a win.
- Unchanged `scene3` cluster-local diagnostics.

## Active / Pending Experiments

These are not completed results yet.

| ID | Purpose | Key flags | Priority | Notes |
| --- | --- | --- | --- | --- |
| CPE-A1-scene2a | Test compressor PE scene-scale on top of A1-style fusion | `--lmc_fusion_geometry_mode value_only_norm --lmc_fusion_scene_scale_source memory_points_p95 --lmc_compressor_pe_scale_mode scene_scale` | High | New run, not a repeat of A1 |
| CPE-A1-diag | Same as above with runtime diagnostics | add `--lmc_log_runtime_stats True --lmc_runtime_stats_interval 20 --lmc_runtime_stats_max_pixels 4096` | High | Run one diagnostic job; no need to block all other jobs |
| B1-scalar-mix-scene2a | Test multi-layer key learned scalar mix while keeping value as all-layer concat | `--lmc_key_feature_mode scalar_mix --lmc_key_slice_idx 2` | High | New structural run; do not combine with CPE in the first pass |
| B1-scalar-mix-diag | Same as B1 with runtime diagnostics | add `--lmc_log_runtime_stats True --lmc_runtime_stats_interval 20 --lmc_runtime_stats_max_pixels 4096` | High | Use to inspect `lmc_key_mix_weights` and token usage |
| Cross-scene CPE | Check whether CPE effect generalizes | same CPE flags on selected non-scene2a scenes | Medium | Launch after scene2a CPE has a useful signal |

## Duplicate-Prevention Rules

1. Do not rerun A0/A1/A2 unchanged on `scene2a/c0_p4`; they are already done.
2. Do not rerun `forceglobal_fixedzero` as a candidate default.
3. Do not expand `geokey_norm` until there is a new mechanism, not just the
   same flag set.
4. Do not expand C1 aux-ref or alpha-scaling before a paired repeated-eval win.
5. New experiments should change exactly one primary mechanism when possible:
   fusion geometry, compressor PE scaling, S1 step mode, or auto-global mode.
6. Every new completed run should append:
   - exact command or `run_command.txt` path,
   - scene / variant / memory path,
   - checkpoint path,
   - eval protocol,
   - metrics table,
   - decision: promote, keep diagnostic, reject, or rerun needed.
