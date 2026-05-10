# Key Results: Reference-Consistent Memory

This file is the canonical place to record validated key results for the
reference-consistent memory workstream.

Rules:

- Only record results that were actually run and checked.
- Prefer one-line conclusions plus exact paths.
- Keep old entries; append new ones instead of overwriting history.
- Use this file for experiment milestones, not implementation notes.
- For future key experiments, default to `5` repeated eval passes when runtime
  is cheap enough; keep all `eval_summary_*.txt` files.
- For each experiment block, it is allowed to report the best `acc25`, best
  median errors, and best `acc5` from different runs inside that block, as
  long as the source run for each metric is stated explicitly.

## 0. Experiment Summary Table

This table is the main experiment table. Each row is one key experiment node,
and the metrics are the best-of values inside that experiment's own
`*_eval_log.txt` and `eval_summary_*.txt` files. For future runs, the intended
default is best-of over `5` eval repeats whenever evaluation is cheap enough.

| Experiment | Scope | Best acc25 | Best Median Rot (deg) | Best Median Trans (cm) | Best acc5 | Metric Sources | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| scene2a_phase0_c0_rerun | single-memory baseline rerun | 100.00 | 0.3534 | 3.3740 | 73.15 | scanned `*_eval_log.txt` + `eval_summary_*.txt`; acc25: iter 6; median rot/trans + acc5: iter 28 | validated |
| scene2a_phase4_c0_20260422_122243 | repaired single-memory C0 | 99.61 | 0.3212 | 3.0150 | 76.65 | scanned `*_eval_log.txt` + `eval_summary_*.txt`; acc25: iter 3; median rot: iter 28; median trans + acc5: iter 27 | validated |
| scene2a_c1_20260426_113134 | single-memory C1 same recipe | 99.61 | 0.3444 | 3.4055 | 71.98 | scanned `*_eval_log.txt` + `eval_summary_*.txt`; acc25: iter 4; median rot + acc5: iter 26; median trans: iter 10 | validated |
| scene3_fallback60_phase4_c0_20260423_111442 | diagnostic 60-view fallback single-memory | 96.83 | 0.6379 | 3.3900 | 69.52 | scanned `*_eval_log.txt` + `eval_summary_*.txt`; acc25: iter 17; median rot/trans + acc5: iter 28 | validated diagnostic |
| scene3_c1_20260426_122510 | diagnostic single-memory C1 same recipe | 98.10 | 0.6379 | 3.4885 | 68.89 | scanned `*_eval_log.txt` + `eval_summary_*.txt`; acc25 + median trans: iter 28; median rot: iter 23; acc5: iter 25 | validated diagnostic |
| scene1_c1_20260428_101453 | cross-scene single-memory C1 same recipe | 97.00 | 0.5614 | 3.3743 | 70.84 | scanned `*_eval_log.txt` + `eval_summary_*.txt`; acc25: iter 25/28; median rot/trans + acc5: iter 28 | validated |
| scene4a_c1_20260428_101453 | cross-scene single-memory C1 same recipe | 100.00 | 0.6420 | 2.8059 | 75.32 | scanned `*_eval_log.txt` + `eval_summary_*.txt`; acc25: iter 13; median rot: iter 24; median trans: iter 10; acc5: iter 26 | validated |
| scene2a_c1_auxref01_4090_20260508 | 4090 exploratory aux-ref paired eval | 99.61 | 0.3469 | 3.3428 | 72.37 | 5 saved deterministic eval summaries; best values across seeds | exploratory negative |
| scene2a_c1_ctrl_4090_20260508 | 4090 exploratory control paired eval | 99.22 | 0.3629 | 3.3501 | 73.93 | 5 saved deterministic eval summaries; best values across seeds | exploratory control |
| scene3_cluster01_20260425_162835 | cluster-local single run | 97.46 | 0.6768 | 3.6337 | 66.67 | scanned `*_eval_log.txt` + `eval_summary_*.txt`; acc25 + median trans + acc5: iter 28; median rot: iter 18 | validated |
| scene3_cluster02_20260425_162840 | cluster-local single run | 96.19 | 0.6748 | 3.3586 | 67.94 | scanned `*_eval_log.txt` + `eval_summary_*.txt`; acc25: iter 16; median rot/trans + acc5: iter 28 | validated |
| scene3_cluster_ensemble_eval | query-time cluster ensemble | 99.05 | 0.6327 | 3.0805 | 68.89 | scanned ensemble `eval_summary_*.txt` | validated |
| scene3_c1_cluster01_20260501_213155 | cluster-local C1 single run | 93.65 | 0.7050 | 3.6440 | 65.71 | scanned `*_eval_log.txt` + `eval_summary_*.txt`; post-train eval | validated |
| scene3_c1_cluster02_20260501_213200 | cluster-local C1 single run | 96.19 | 0.6588 | 3.4461 | 67.30 | scanned `*_eval_log.txt` + `eval_summary_*.txt`; post-train eval | validated |
| scene3_c1_cluster_ensemble_eval | query-time cluster-local C1 ensemble | 97.78 | 0.6657 | 3.3866 | 68.57 | scanned ensemble `eval_summary_*.txt` | validated |
| shared_model_multi_memory | one shared checkpoint over multiple memories | - | - | - | - | not implemented yet | pending |

## 0.1 Milestone Table

This table tracks non-metric milestones and supporting artifacts.

| Area | Stage | Artifact / Run | Key Result | Status |
| --- | --- | --- | --- | --- |
| scene2a | Q1a reference-swap | `reference_swap_report_20260421_151731.json` | best ref `1181`, strong alt `2850`, old default `1682` weaker | validated |
| scene2a | Repaired extraction C0 | `20260423_103144` | ref `2850`, swap `3909 -> 4294`, coverage max `3.715 -> 2.338 m` | validated |
| scene2a | Single-memory C1 schema | `20260426_113134` | `C1` fields complete; loader now returns ref-frame geometry + ref-frame scene center | validated |
| scene2a | Single-memory C1 end-to-end | `20260426_114419` | first full `C1` train/eval complete; works end-to-end but trails repaired `C0` on `acc5` and median trans | validated |
| scene3 | Single-forward negative case | `20260423_105122` / `20260423_111442` | coverage/probe fail, decision=`cluster_branch`; single-forward remains infeasible | validated negative case |
| scene3 | Single-memory C1 diagnostic | `20260426_123243` | `C1` train/eval path runs end-to-end; remains diagnostic, not a policy success case | validated diagnostic |
| scene1 | Single-memory C1 cross-scene | `20260428_101456` | same-recipe `C1` run validated; best `acc5=70.84` | validated |
| scene4a | Single-memory C1 cross-scene | `20260428_101456` | same-recipe `C1` run validated; best `acc5=75.32` | validated |
| scene2a | 4090 exploratory aux-ref pair | `20260508_000246` vs `20260508_000323` | 5-seed eval: aux-ref mean `acc5=71.98` vs control `73.54`; do not expand aux-ref setting | exploratory negative |
| scene3 | Phase 4 cluster fallback extraction | `20260425_130557` | built `memory_bse.clustered.pt` + two `cluster_local` memories | validated |
| scene3 | Cluster ensemble eval | `cluster_ensemble_eval` | `2cm/2deg=29.84`, selection `106/209` | validated |
| scene3 | Cluster-local C1 extraction | `20260430_222156` | both `cluster_01/02` packages carry full `C1` ref-space + recovery metadata | validated |
| scene3 | Cluster-local C1 ensemble eval | `20260501_213155` | `acc25=97.78`, `acc5=68.57`, selection `104/211`; close to C0 cluster ensemble but still slightly weaker | validated |

## 1. scene2a_train

### 1.1 Q1a reference-swap diagnostic

Result:

- best tested reference: `1181`
- close second and also strong: `2850`
- default historical root `1682` is significantly worse on translation

Artifact:

- `/home/xwh/project/ace_depth/ace_dinov2_lmc/memory_extraction/04_evaluation/reference_swap/scene2a_train/reference_swap_report_20260421_151731.json`

Conclusion:

- `scene2a_train` needed candidate gating; one fixed root heuristic was not
  reliable enough.

### 1.2 Full Phase 0 rerun baseline

Training run:

- `/home/xwh/project/ace_depth/ace_dinov2_lmc/04_evaluation/train_compare/memory_pooled_vs_asb/indoor6_ace/scene2a/dino_ace_lmc_ace_g/scene2a_asb40_phase0_c0_rerun_20260422_aceg_fS2cie_global_res518_buf2.6M_F7.7M_K64_it28_ep24_bs5120_spi_s1buf_sp384_onecycle_improved`

Best metrics:

- `25cm/5deg = 99.2218%`
- `best_iter = 28`
- `5cm/5deg = 73.1518%`
- `2cm/2deg = 19.0661%`
- `1cm/1deg = 4.2802%`
- median rotation = `0.3534 deg`
- median translation = `3.3740 cm`

Conclusion:

- The older `pct5=56.03` result was under-trained and should no longer be used
  as the main baseline.

### 1.3 Repaired C0 extraction with structured report

Extraction:

- `/home/xwh/project/ace_depth/ace_dinov2_lmc/memory_extraction/04_evaluation/memory_extract/scene2a/40v_v0.05_bilinear_bse_ut0.02_noaug_asb_adaptive_pool1.0_rgate4_prepair8_sor_gm_l2/20260423_103144`

Key facts:

- selected reference = `2850`
- repair swap = `3909 -> 4294`
- coverage mean: `0.7163 -> 0.6089 m`
- coverage p95: `1.9432 -> 1.5291 m`
- coverage max: `3.7155 -> 2.3385 m`
- probe mean translation: about `0.064 m`
- graph health stable:
  - component count `4 -> 4`
  - isolated count `0 -> 0`

Conclusion:

- The combined gate + repair path fixed the root-choice failure and improved
  long-tail coverage in a measurable way.

### 1.4 Best validated downstream C0 run

Training run:

- `/home/xwh/project/ace_depth/ace_dinov2_lmc/04_evaluation/train_compare/memory_pooled_vs_asb/indoor6_ace/scene2a/dino_ace_lmc_ace_g/scene2a_asb40_phase4_c0_20260422_122243_aceg_fS2cie_global_res518_buf2.6M_F7.7M_K64_it28_ep24_bs5120_spi_s1buf_sp384_onecycle_improved`

Best metrics:

- `25cm/5deg = 98.4436%`
- `best_iter = 27`
- `5cm/5deg = 76.6537%`
- `2cm/2deg = 26.0700%`
- `1cm/1deg = 5.0584%`
- median rotation = `0.3266 deg`
- median translation = `3.0150 cm`

Conclusion:

- This is still the current best validated single-memory C0 downstream result.

### 1.5 Single-memory C1 extraction schema verification

Extraction:

- `/home/xwh/project/ace_depth/ace_dinov2_lmc/memory_extraction/04_evaluation/memory_extract/scene2a/40v_v0.05_bilinear_bse_ut0.02_noaug_asb_adaptive_pool1.0_rgate4_prepair8_sor_gm_l2/20260426_113134`

Schema checks:

- `contract_mode = C1`
- `reference_index = 2850`
- `points_world`, `points_ref`, `points_ref_norm` all present
- `scene_center`, `scene_center_world`, `scene_center_ref`,
  `scene_center_ref_norm` all present
- `scene_center_contract_space = reference_camera_normalized`
- `conditioning_reference.T_ref_c2w_world` and `T_world_to_ref` present
- `normalization_ref.mu_ref` and `sigma_ref` present

Loader check:

- `load_memory_features()` now returns:
  - `scene_center = scene_center_ref_norm`
  - `scene_center_world` preserved separately
  - `pooled_points` interpreted in the C1 contract path

Conclusion:

- The single-memory C1 schema is internally consistent and is now validated by
  a real end-to-end ACE-G train/eval run.

### 1.6 Single-memory C1 downstream validation

Training run:

- `/home/xwh/project/ace_depth/ace_dinov2_lmc/04_evaluation/train_compare/memory_pooled_vs_asb_c1/indoor6_ace/scene2a/dino_ace_lmc_ace_g/20260426_114419_aceg_fS2cie_global_res518_buf2.6M_F7.7M_K64_it28_ep24_bs5120_spi_s1buf_sp384_onecycle_improved`

Best metrics inside this experiment block:

- `25cm/5deg = 99.61%` at iter `4`
- `5cm/5deg = 71.9844%` at iter `26`
- median rotation = `0.3444 deg` at iter `26`
- median translation = `3.4055 cm` at iter `10`

Comparison to repaired C0:

- `acc25` ties the repaired C0 peak (`99.61%`)
- median rotation is close but worse (`0.3444` vs `0.3212 deg`)
- median translation is worse (`3.4055` vs `3.0150 cm`)
- `acc5` is materially worse (`71.98` vs `76.65`)

Conclusion:

- `C1` now works end-to-end on `scene2a`, but on the current same-recipe
  selected set it does not beat the repaired `C0` baseline.

### 1.7 scene2a experiment best-of summary

This summary is allowed to mix different runs within the `scene2a` experiment
group, and it is based on all recorded eval-log entries in this experiment
block, not only the checkpoint tagged `BEST`.

| Metric | Best Value | Source |
| --- | --- | --- |
| `acc25` | `100.0` | `scene2a_asb40_phase0_c0_rerun_20260422_...`, iter `6` |
| median rotation | `0.3212 deg` | `scene2a_asb40_phase4_c0_20260422_122243_...`, iter `28` |
| median translation | `3.0150 cm` | `scene2a_asb40_phase4_c0_20260422_122243_...` |
| `acc5` | `76.6537` | `scene2a_asb40_phase4_c0_20260422_122243_...`, iter `27` |

### 1.8 4090 exploratory aux-ref paired re-evaluation

Aux-ref run:

- `/home/xwh/project/ace_depth/ace_dinov2_lmc/04_evaluation/train_compare/memory_pooled_vs_asb_c1/indoor6_ace/scene2a/dino_ace_lmc_ace_g/20260508_000246_aceg_fS2cie_global_res518_buf2.6M_F7.7M_K64_it28_ep24_bs10240_spi_s1buf_sp384_onecycle_improved`
- checkpoint: `best_K64_it28_scene2a_c1_auxref01_detfps_20260508_000244.pt`
- config: `--c1_aux_ref_loss_weight 0.1`, `--lmc_fps_start_policy farthest_from_center`, `buffer_size_final=7680000`, `buffer_on_cpu=false`

Control run:

- `/home/xwh/project/ace_depth/ace_dinov2_lmc/04_evaluation/train_compare/memory_pooled_vs_asb_c1/indoor6_ace/scene2a/dino_ace_lmc_ace_g/20260508_000323_aceg_fS2cie_global_res518_buf2.6M_F7.7M_K64_it28_ep24_bs10240_spi_s1buf_sp384_onecycle_improved`
- checkpoint: `best_K64_it28_scene2a_c1_detfps_ctrl_20260508_000321.pt`
- config: `--c1_aux_ref_loss_weight 0.0`, `--lmc_fps_start_policy farthest_from_center`, `buffer_size_final=7680000`, `buffer_on_cpu=false`

5-seed deterministic eval summary:

| Group | Mean Median Rot | Mean Median Trans | Mean acc25 | Mean acc10 | Mean acc5 |
| --- | --- | --- | --- | --- | --- |
| aux-ref `0.1` | `0.3561 deg` | `3.4068 cm` | `99.61` | `89.57` | `71.98` |
| control `0.0` | `0.3660 deg` | `3.4038 cm` | `99.22` | `88.64` | `73.54` |

Conclusion:

- Aux-ref improves coarse metrics (`acc25`, `acc10`) and median rotation in
  this exploratory 4090 pair, but loses clearly on `acc5` and does not improve
  median translation.
- This does not pass the expansion check. Do not run an all-scene aux-ref sweep
  from this setting.

## 2. scene3_train

### 2.1 Single-forward negative policy example

40-view extraction:

- `/home/xwh/project/ace_depth/ace_dinov2_lmc/memory_extraction/04_evaluation/memory_extract/scene3/40v_v0.05_bilinear_bse_ut0.02_noaug_asb_adaptive_pool1.0_rgate4_prepair8_sor_gm_l2/20260423_105122`

Key facts:

- decision = `cluster_branch`
- selected fallback reference = `613`
- repair made no swap
- coverage = `mean 0.745 / p95 2.449 / max 5.633 m`
- best fallback probe translation = `mean 0.195 / q90 0.363 / max 0.517 m`

60-view diagnostic:

- `/home/xwh/project/ace_depth/ace_dinov2_lmc/memory_extraction/04_evaluation/memory_extract/scene3/60v_v0.05_bilinear_bse_ut0.02_noaug_asb_adaptive_pool1.0_rgate6_prepair12_sor_gm_l2/20260423_111442`

Key facts:

- decision still = `cluster_branch`
- selected fallback reference = `3136`
- coverage = `mean 0.699 / p95 2.449 / max 5.633 m`
- best probe translation = `mean 0.148 / q90 0.254 / max 0.440 m`

Conclusion:

- `scene3_train` is the first concrete negative policy example; single-forward
  remained infeasible even after increasing the view budget.

### 2.2 Diagnostic single-memory fallback training

Training run:

- `/home/xwh/project/ace_depth/ace_dinov2_lmc/04_evaluation/train_compare/memory_pooled_vs_asb/indoor6_ace/scene3/dino_ace_lmc_ace_g/scene3_asb60_phase4_fallback_c0_20260423_111442_aceg_fS2cie_global_res518_buf2.6M_F7.7M_K64_it28_ep24_bs5120_spi_s1buf_sp384_onecycle_improved`

Best-selection eval:

- `25cm/5deg = 95.87%`
- `5cm/5deg = 69.52%`
- `10cm/5deg = 88.57%`
- median rotation = `0.6379 deg`
- median translation = `3.3900 cm`

Automatic post-train re-eval:

- `25cm/5deg = 97.14%`
- `5cm/5deg = 67.94%`
- `10cm/5deg = 88.25%`
- median rotation = `0.6495 deg`
- median translation = `3.4289 cm`

Conclusion:

- This run is useful only as a diagnostic fallback; it is not a valid
  single-forward policy success case.

### 2.2a Diagnostic single-memory C1 training

Training run:

- `/home/xwh/project/ace_depth/ace_dinov2_lmc/04_evaluation/train_compare/memory_pooled_vs_asb_c1/indoor6_ace/scene3/dino_ace_lmc_ace_g/20260426_123243_aceg_fS2cie_global_res518_buf2.6M_F7.7M_K64_it28_ep24_bs5120_spi_s1buf_sp384_onecycle_improved`

Best metrics inside this experiment block:

- `25cm/5deg = 98.10%` at iter `28`
- `5cm/5deg = 68.8889%` at iter `25`
- median rotation = `0.6379 deg` at iter `23`
- median translation = `3.4885 cm` at iter `28`

Conclusion:

- The single-memory `C1` train/eval path is valid on `scene3`, but this
  remains a diagnostic run on a scene that still structurally wants the
  cluster path.

## 3. Cross-scene single-memory C1 validation

### 3.1 scene1

Training run:

- `/home/xwh/project/ace_depth/ace_dinov2_lmc/04_evaluation/train_compare/memory_pooled_vs_asb_c1/indoor6_ace/scene1/dino_ace_lmc_ace_g/20260428_101456_aceg_fS2cie_global_res518_buf2.6M_F7.7M_K64_it28_ep24_bs10240_spi_s1buf_sp384_onecycle_improved`

Best metrics inside this experiment block:

- `25cm/5deg = 97.00%` at iter `25/28`
- `5cm/5deg = 70.8385%` at iter `28`
- median rotation = `0.5614 deg` at iter `28`
- median translation = `3.3743 cm` at iter `28`

Conclusion:

- Same-recipe single-memory `C1` is now validated on `scene1`.

### 3.2 scene4a

Training run:

- `/home/xwh/project/ace_depth/ace_dinov2_lmc/04_evaluation/train_compare/memory_pooled_vs_asb_c1/indoor6_ace/scene4a/dino_ace_lmc_ace_g/20260428_101456_aceg_fS2cie_global_res518_buf2.6M_F7.7M_K64_it28_ep24_bs10240_spi_s1buf_sp384_onecycle_improved`

Best metrics inside this experiment block:

- `25cm/5deg = 100.00%` at iter `13`
- `5cm/5deg = 75.3165%` at iter `26`
- median rotation = `0.6420 deg` at iter `24`
- median translation = `2.8059 cm` at iter `10`

Conclusion:

- Same-recipe single-memory `C1` is now validated on `scene4a`.

### 2.3 Phase 4 offline cluster fallback extraction

Extraction:

- `/home/xwh/project/ace_depth/ace_dinov2_lmc/memory_extraction/04_evaluation/memory_extract/scene3/60v_v0.05_bilinear_bse_ut0.02_noaug_asb_adaptive_pool1.0_rgate6_prepair12_cfb2_sor_gm_l2/20260425_130557`

Artifacts present:

- `cluster_fallback_plan.json`
- `memory_bse.clustered.pt`
- `memory_bse.cluster_01.pt`
- `memory_bse.cluster_02.pt`

Key facts:

- decision = `cluster_branch`
- built `2` cluster-local memories
- both clusters marked `low_confidence=true`
- failure reason = `disconnected_covis_graph`

Conclusion:

- The Phase 4 offline construction path is valid and observable, even though
  scene3 remains structurally difficult.

### 2.4 Per-cluster training results

cluster01 run:

- `/home/xwh/project/ace_depth/ace_dinov2_lmc/04_evaluation/train_compare/memory_pooled_vs_asb/indoor6_ace/scene3/dino_ace_lmc_ace_g/20260425_162835_aceg_fS2cie_global_res518_buf2.6M_F7.7M_K64_it28_ep24_bs5120_spi_s1buf_sp384_onecycle_improved`

Best:

- `25cm/5deg = 97.4603%`
- `5cm/5deg = 66.67%`
- `10cm/5deg = 86.98%`
- median rotation = `0.7083 deg`
- median translation = `3.6337 cm`

cluster02 run:

- `/home/xwh/project/ace_depth/ace_dinov2_lmc/04_evaluation/train_compare/memory_pooled_vs_asb/indoor6_ace/scene3/dino_ace_lmc_ace_g/20260425_162840_aceg_fS2cie_global_res518_buf2.6M_F7.7M_K64_it28_ep24_bs5120_spi_s1buf_sp384_onecycle_improved`

Best:

- `25cm/5deg = 95.8730%`
- `5cm/5deg = 67.94%`
- `10cm/5deg = 86.03%`
- median rotation = `0.6748 deg`
- median translation = `3.3586 cm`

Conclusion:

- Both clusters are individually usable; cluster02 is slightly stronger alone.

### 2.5 Joint cluster ensemble evaluation

Output directory:

- `/home/xwh/project/ace_depth/ace_dinov2_lmc/04_evaluation/train_compare/memory_pooled_vs_asb/indoor6_ace/scene3/dino_ace_lmc_ace_g/cluster_ensemble_eval`

Metrics:

- `25cm/5deg = 99.05%`
- `10cm/5deg = 92.06%`
- `5cm/5deg = 68.89%`
- `2cm/2deg = 29.84%`
- median rotation = `0.6327 deg`
- median translation = `3.0805 cm`

Selection counts:

- `cluster01 = 106`
- `cluster02 = 209`

Conclusion:

- The sub-memory route has now passed an initial validation:
  cluster split + aggregated evaluation does outperform either single cluster
  model on `scene3`.

### 2.5a Cluster-local C1 extraction field validation

Extraction:

- `/home/xwh/project/ace_depth/ace_dinov2_lmc/memory_extraction/04_evaluation/memory_extract/scene3/60v_v0.05_bilinear_bse_ut0.02_noaug_asb_adaptive_pool1.0_rgate6_prepair12_cfb2_sor_gm_l2/20260430_222156`

Field checks on both `memory_bse.cluster_01.pt` and `memory_bse.cluster_02.pt`:

- `contract_mode = C1`
- `points_ref` present
- `points_ref_norm` present
- `scene_center_world`, `scene_center_ref`, `scene_center_ref_norm` present
- `conditioning_reference.T_ref_c2w_world` and `T_world_to_ref` present
- `normalization_ref.mu_ref` and `sigma_ref` present

Conclusion:

- The clustered package path now carries the full `C1` contract metadata and
  is directly consumable by the current trainer without schema patching.

### 2.5b Cluster-local C1 per-cluster training results

cluster01 run:

- `/home/xwh/project/ace_depth/ace_dinov2_lmc/04_evaluation/train_compare/memory_pooled_vs_asb_c1/indoor6_ace/scene3/dino_ace_lmc_ace_g/20260501_213155_aceg_fS2cie_global_res518_buf2.6M_F7.7M_K64_it28_ep24_bs5120_spi_s1buf_sp384_onecycle_improved`

Post-train eval:

- `25cm/5deg = 93.65%`
- `10cm/5deg = 85.71%`
- `5cm/5deg = 65.71%`
- `2cm/2deg = 20.00%`
- `1cm/1deg = 3.81%`
- median rotation = `0.7050 deg`
- median translation = `3.6440 cm`

cluster02 run:

- `/home/xwh/project/ace_depth/ace_dinov2_lmc/04_evaluation/train_compare/memory_pooled_vs_asb_c1/indoor6_ace/scene3/dino_ace_lmc_ace_g/20260501_213200_aceg_fS2cie_global_res518_buf2.6M_F7.7M_K64_it28_ep24_bs5120_spi_s1buf_sp384_onecycle_improved`

Post-train eval:

- `25cm/5deg = 96.19%`
- `10cm/5deg = 85.40%`
- `5cm/5deg = 67.30%`
- `2cm/2deg = 21.27%`
- `1cm/1deg = 5.40%`
- median rotation = `0.6588 deg`
- median translation = `3.4461 cm`

Conclusion:

- `cluster02` remains the stronger single branch, and both clustered `C1`
  branches are individually usable.

### 2.5c Cluster-local C1 ensemble evaluation

Output files:

- `/home/xwh/project/ace_depth/ace_dinov2_lmc/04_evaluation/train_compare/memory_pooled_vs_asb_c1/indoor6_ace/scene3/dino_ace_lmc_ace_g/20260501_213155_aceg_fS2cie_global_res518_buf2.6M_F7.7M_K64_it28_ep24_bs5120_spi_s1buf_sp384_onecycle_improved/eval_summary_scene3_ensemble.txt`
- `/home/xwh/project/ace_depth/ace_dinov2_lmc/04_evaluation/train_compare/memory_pooled_vs_asb_c1/indoor6_ace/scene3/dino_ace_lmc_ace_g/20260501_213155_aceg_fS2cie_global_res518_buf2.6M_F7.7M_K64_it28_ep24_bs5120_spi_s1buf_sp384_onecycle_improved/ensemble_selection_scene3_ensemble.jsonl`

Metrics:

- `25cm/5deg = 97.78%`
- `10cm/5deg = 89.21%`
- `5cm/5deg = 68.57%`
- `2cm/2deg = 27.62%`
- `1cm/1deg = 4.76%`
- median rotation = `0.6657 deg`
- median translation = `3.3866 cm`

Selection counts:

- `cluster01 = 104`
- `cluster02 = 211`

Comparison to old cluster-local C0 ensemble:

- `acc5` is nearly tied (`68.57` vs `68.89`)
- `acc25` and `acc10` are slightly lower
- median translation is slightly worse (`3.3866` vs `3.0805 cm`)

Conclusion:

- `scene3 cluster-local C1` is now validated end-to-end.
- It is close to the older cluster-local `C0` ensemble, but still slightly
  weaker overall.

### 2.6 scene3 experiment best-of summary

This summary is allowed to mix different runs within the `scene3` experiment
group. For single-model runs it uses all recorded eval-log entries; for the
ensemble result it uses the final saved ensemble summary.

| Metric | Best Value | Source |
| --- | --- | --- |
| `acc25` | `99.05` | `cluster_ensemble_eval` |
| median rotation | `0.6327 deg` | `cluster_ensemble_eval` |
| median translation | `3.0805 cm` | `cluster_ensemble_eval` |
| `acc5` | `69.52` | `scene3_asb60_phase4_fallback_c0_20260423_111442_...`, iter `28` |

## 3. Current Best-Supported Conclusions

1. `scene2a_train`:
   - the gate + repair path is validated and improves downstream performance
   - current best validated result is still the repaired C0 single-memory run

2. `scene3_train`:
   - single-forward should not be forced
   - cluster fallback is necessary
   - cluster-local independent prediction plus post-hoc selection is already
     useful in both `C0` and `C1`
   - current best cluster route is still the older `C0` ensemble, but `C1`
     now runs end-to-end and is close enough to keep as the forward path

3. Single-memory C1:
   - schema and loader/trainer contract are now validated across `scene2a`,
     `scene3`, `scene1`, and `scene4a`
   - the main remaining question is no longer "does C1 run", but "how do we
     recover fine precision while keeping normalized-C1 as the scalable path"
   - the 4090 exploratory aux-ref setting did not recover fine precision on
     `scene2a`; it should not be expanded without a revised auxiliary signal

## 4. Next Empty Slots To Fill

Add the next validated results here when they happen:

- first single-memory `scene2a` C1 training result
- cluster-local C1 versus cluster-local C0 comparison summary for `scene3`
- first shared-model multi-memory training result
- first `scene2a` mechanism run with weaker normalization (`alpha > 1`)
- first revised `scene2a` auxiliary-supervision mechanism that gives real
  fine-precision supervision
