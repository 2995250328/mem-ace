# Reference-Consistent Memory Progress

Last updated: 2026-05-08

This file records the current implementation state of
`EXECUTION_PLAN_reference_consistent_memory.md` so a new conversation can
resume work quickly without reconstructing context from chat history.

## 0. Maintenance Rules

This file is intended to be updated continuously as implementation progresses.

Required maintenance behavior:

1. After every completed task that changes code, diagnostics, experiment
   commands, or implementation status, update this file in the same work turn.
2. Always update `Last updated`.
3. Always update all three of these areas if they changed:
   - "What Has Been Implemented"
   - "What Has Not Been Implemented Yet"
   - "Recommended Next Steps"
4. Add one concise entry to the change log for every meaningful milestone.
5. If a task is only partially done, mark it explicitly as partial instead of
   moving it into the completed bucket.
6. If a previous assumption is invalidated by new experiments, overwrite the
   diagnosis sections instead of only appending contradictory notes.

Environment/command constraint:

- Do not use `conda run` in suggested commands for this project.
- Always prefer:

```bash
conda activate <env>
python ...
```

- Current preferred environments:
  - `mapanything_new` for memory extraction / Q1a diagnostics
  - `ace` for ACE / LMC training and evaluation

Definition of done for future turns touching this project:

- code and/or experiments updated
- this progress file updated
- next-step recommendation updated

## 0.1 Change Log

- 2026-04-21: Added the progress tracking document itself.
- 2026-04-21: Implemented C1 trainer/eval recovery path and metadata
  propagation through memory loading and checkpoints.
- 2026-04-21: Implemented `memory_policy_report.json` Phase 0 reporting.
- 2026-04-21: Fixed `anchor_support` root/reference selection by adding
  `select_anchor_support_reference()`.
- 2026-04-21: Added Q1a diagnostic entry point
  `memory_extraction/reference_swap_diagnostic.py`.
- 2026-04-21: Added explicit maintenance rules so this file must be updated
  whenever implementation progress changes.
- 2026-04-21: Added command constraint: do not use `conda run`; use
  `conda activate <env>` instead.
- 2026-04-21: First Q1a target-scene run confirmed the fixed selector now
  chooses a different selected set headed by `1682`; old root `2355` is no
  longer a valid same-set reference candidate.
- 2026-04-21: Completed Q1a same-selected-set comparison on
  `scene2a_train`; references `1181` and `2850` were materially better than
  the default root `1682`, so the root-selection fix alone is not sufficient.
- 2026-04-21: Implemented a minimal Phase 3 gate in
  `run_memory_extraction.py` and `extract_memory.sh`: Top-M reference
  candidates, light gate, probe gate, `policy_decision.json`, and final batch
  reordering to the selected reference.
- 2026-04-21: First gated extraction attempt failed immediately with
  `NameError: Sequence is not defined`; fixed the missing `typing.Sequence`
  import in `run_memory_extraction.py`.
- 2026-04-21: First successful gated extraction on `scene2a_train` exposed an
  over-strict light gate: all candidates were rejected before probe because
  coverage was treated as reference-dependent and `ref_q90` was too hard.
  Adjusted light gate so coverage stays diagnostic-only and high-connectivity
  candidates can still reach probe.
- 2026-04-21: Second gated extraction on `scene2a_train` succeeded end-to-end.
  `policy_decision.json` selected reference `1181` over default `1682`, and
  final probe/final PoseEval mean translation dropped to about `0.068 m`.
  Coverage, however, still failed the scene-derived limit: mean `0.716 m`,
  max `3.715 m` versus target max `2.618 m`.
- 2026-04-22: Verified downstream ACE-G training on the gated
  `scene2a_train` memory. The Phase 3 run materially outperformed the older
  Phase 0 baseline: `pct5` improved from `56.03` to `72.37`, `pct2` from
  `12.84` to `26.07`, and median translation from `4.57 cm` to `3.41 cm`.
  The active bottleneck is now coverage-tail repair, not reference gating.
- 2026-04-22: Implemented a first fixed-budget post-repair pass for
  `anchor_support`: after the selected set is full, it can swap low-value
  fillers/supports for tail-reducing candidates under topology/reference-risk
  gates. Added CLI + `extract_memory.sh` support, but no new scene2a
  extraction result has been validated yet.
- 2026-04-22: Ran the first post-repair extraction on `scene2a_train` with
  `ASB_POST_REPAIR=true`. The repair pass made no swap and logged
  `Post-repair stopped: no swap improved coverage tail`. Final outputs were
  unchanged relative to the earlier gated baseline: reference stayed `1181`,
  coverage stayed `mean=0.716 m / p95=1.943 m / max=3.715 m`, and probe
  translation stayed `mean=0.068 m / q90=0.102 m / max=0.143 m`.
- 2026-04-22: Strengthened the fixed-budget post-repair search after the first
  failed validation. The repair now prioritizes the owner of the worst tail
  cluster, builds a local candidate shortlist around the worst uncovered
  region, and replaces the previous hard reference-risk reject with a bounded
  slack + penalty scheme. Static validation passed; scene validation is still
  pending for this stronger version.
- 2026-04-22: Validated the stronger post-repair search on `scene2a_train`.
  The repair executed one real swap (`drop=3909 -> add=4294`) and materially
  improved coverage while keeping a strong reference tier. Compared with the
  earlier `rgate4` baseline:
  - coverage mean: `0.716 -> 0.609 m`
  - coverage p95: `1.943 -> 1.529 m`
  - coverage max: `3.715 -> 2.338 m`
  - probe translation mean: `0.0680 -> 0.0641 m`
  - probe translation q90: `0.1025 -> 0.0984 m`
  - selected reference moved to `2850`, which remains in the previously
    validated good-reference tier (`1181/2850`).
- 2026-04-22: Upgraded post-repair again to balance top-K uncovered clusters
  instead of only the single worst hole, and wired repair metadata into
  `memory_policy_report.json`. Static validation passed; the next extraction
  will expose swap details, before/after coverage, and graph summaries through
  JSON instead of only log lines.
- 2026-04-23: Reviewed completed downstream ACE-G training for the repaired
  Phase 4 memory and the full Phase 0 rerun on `scene2a`. The repaired memory
  is the current best run (`pct5=76.65`, median translation `3.01 cm`) and
  outperforms the Phase 0 rerun (`pct5=73.15`, median translation `3.37 cm`).
  The trained repaired memory is still the pre-top-K-report extraction
  `20260422_122243`, so its `memory_policy_report.json` does not yet contain
  the newer structured `repair` section.
- 2026-04-23: Validated the top-K/report repaired extraction on
  `scene2a_train` at `20260423_103144`. The run reproduced the useful
  `drop=3909 -> add=4294` swap, kept selected reference `2850`, preserved the
  repaired coverage/probe stats, and wrote structured `repair` details into
  `memory_policy_report.json` including coverage before/after and graph
  before/after.
- 2026-04-23: Ran cross-scene top-K/report extraction on `scene3_train` at
  `20260423_105122`. The policy correctly predicted `cluster_branch`: all four
  candidate references failed the probe gate, repair made no swap, coverage
  stayed at `mean=0.745 m / p95=2.449 m / max=5.633 m`, and the best fallback
  reference `613` still had probe translation `mean=0.195 m / q90=0.363 m /
  max=0.517 m`. This is the first concrete scene where the current
  single-forward path is not supported by the gate.
- 2026-04-23: Ran the `scene3_train` larger-view diagnostic with 60 views,
  Top-M=6, and max repair swaps=12 at `20260423_111442`. The extra views
  improved the best fallback reference to `3136` and reduced probe translation
  to `mean=0.148 m / q90=0.254 m / max=0.440 m`, but no candidate passed the
  probe gate and policy still selected `cluster_branch`. This points to a true
  cluster-fallback case rather than just a 40-view budget issue.
- 2026-04-23: Decided that training the `scene3_train` 60-view fallback memory
  is acceptable only as a negative-control diagnostic. It must not be recorded
  as a successful single-forward policy result unless downstream performance
  unexpectedly contradicts the gate.
- 2026-04-24: Completed the diagnostic ACE-G training run on the `scene3_train`
  60-view fallback memory from `20260423_111442`. The run directory is
  `04_evaluation/train_compare/memory_pooled_vs_asb/indoor6_ace/scene3/dino_ace_lmc_ace_g/scene3_asb60_phase4_fallback_c0_20260423_111442_aceg_fS2cie_global_res518_buf2.6M_F7.7M_K64_it28_ep24_bs5120_spi_s1buf_sp384_onecycle_improved`.
  Best checkpoint selection used iter 28 with `5cm/5deg=69.52%`, `10cm/5deg=88.57%`,
  `25cm/5deg=95.87%`, median `0.638 deg / 3.390 cm`. The automatic
  post-train re-evaluation produced `5cm/5deg=67.94%`, `10cm/5deg=88.25%`,
  `25cm/5deg=97.14%`, median `0.650 deg / 3.429 cm`. This is a diagnostic
  fallback result, not a gate-approved single-forward policy success.
- 2026-04-24: Implemented the first offline Phase 4 cluster-fallback builder.
  It is disabled by default and is activated with `ENABLE_CLUSTER_FALLBACK=true`
  / `--enable_cluster_fallback`. When the reference policy predicts
  `cluster_branch`, extraction now plans bounded recursive k=2 clusters over
  selected camera centers, writes `cluster_fallback_plan.json`, builds
  cluster-local memories through the same MapAnything infer +
  `two_pass_processing()` constructor, and saves a sidecar
  `memory_bse.clustered.pt` package plus per-cluster `.pt` files. The main
  single-memory output remains unchanged for backward-compatible training.
- 2026-04-25: Validated the offline Phase 4 path on
  `scene3/60v_v0.05_bilinear_bse_ut0.02_noaug_asb_adaptive_pool1.0_rgate6_prepair12_cfb2_sor_gm_l2/20260425_130557`.
  Extraction produced `cluster_fallback_plan.json`,
  `memory_bse.clustered.pt`, `memory_bse.cluster_01.pt`, and
  `memory_bse.cluster_02.pt` as intended. The policy still predicts
  `cluster_branch`; both clusters are currently marked `low_confidence=true`
  with `disconnected_covis_graph`, so the code path is working but scene3 is
  not yet cleanly decomposed. Added a small helper
  `memory_extraction/emit_cluster_train_commands.py` to emit one training
  command per cluster-local memory, and updated the loader to raise a clear
  error if `memory_bse.clustered.pt` is passed directly into train/eval.
- 2026-04-30: The later `C1` milestones are now recorded in
  `KEY_RESULTS_reference_consistent_memory.md`: `scene2a` single-memory `C1`
  completed end-to-end but still trails repaired `C0`; `scene3` single-memory
  `C1` remains diagnostic; `scene1` and `scene4a` same-recipe `C1` runs were
  validated cross-scene. Future large-scale multi-scene training ideas are now
  tracked separately in repository-root memo
  `C1_NORMALIZED_PRETRAINING_MEMO.md` so they do not mix with the current
  memory-extraction mainline.
- 2026-05-08: Checked the existing 4090 exploratory paired re-evaluation for
  `scene2a` C1 aux-ref (`--c1_aux_ref_loss_weight 0.1`) versus control. The
  saved 5-seed eval summaries already exist. Aux-ref does not pass the
  expansion check: mean `acc5=71.98` versus control `73.54`, with median
  translation essentially tied/slightly worse (`3.4068 cm` versus
  `3.4038 cm`). Do not expand this aux-ref setting to all scenes.
- 2026-05-08: Added a 4090 full-baseline scheduler at
  `memory_extraction/run_indoor6_full_baselines_4090.sh` for the 6-scene
  matrix `c0_original`, `c0_p4`, and `c1_p4`. Training commands use
  `buffer_on_cpu=True`, `buffer_on_cpu_final=True`, `training_buffer_size=2.56M`,
  and `buffer_size_final=7.68M`. Trainer code now also has a
  `--buffer_on_cpu_final` safeguard so final S2 buffers can be forced onto CPU
  even if regular buffers are kept on GPU.
- 2026-05-08: Added all-mode valid-depth patch sampling for buffer/sample-based
  S1. New flags `--buffer_sample_valid_coords` and
  `--buffer_valid_coord_sample_ratio` make C0/C1 and original/P4 variants
  prefer patch-level scene coordinates generated from ACE depth or WAI aux
  depth, falling back to image-mask random sampling when no valid depth exists.
  The 4090 full-baseline scheduler enables this for all variants and passes
  each scene's WAI `gt_depth` root into training.

## 1. Resume Checklist

When resuming in a new conversation, read these files first:

1. `memory_extraction/03_implementation/EXECUTION_PLAN_reference_consistent_memory.md`
2. `memory_extraction/03_implementation/PROGRESS_reference_consistent_memory.md`
3. `memory_extraction/run_memory_extraction.py`
4. `trainer_dinov2_lmc.py`
5. `test_ace_dinov2_lmc.py`
6. `../utils_lmc.py`
7. `memory_extraction/reference_swap_diagnostic.py`

Current high-priority open thread:

- `anchor_support` selection is only partially upgraded.
- A critical root/reference weakness was found on `indoor6/scene2a_train`.
- Q1a has now completed on the fixed 40-view selected set.
- The result shows strong same-set reference sensitivity:
  - default root `1682` is not the best reference for this selected set
  - `1181` and `2850` are much better and are also mutually consistent
  - the new minimal Phase 3 gate is now validated end-to-end on
    `scene2a_train` and downstream training also benefits
- The highest-priority open problem is now selector/report validation:
  - gated extraction fixed reference choice
  - repaired extraction reduced the worst coverage tail and gave the best
    downstream result
  - the newer top-K repair/report path still needs a fresh extraction before
    more training

## 2. What Has Been Implemented

### 2.1 C1 training/eval recovery path

Implemented:

- Trainer now reads `contract_mode`, `conditioning_reference`, and
  `normalization_ref` from loaded memory metadata.
- Trainer supports interpreting head outputs as:
  - `points_world` for C0
  - `points_ref` for C1 without ref normalization
  - `points_ref_norm` for C1 with ref normalization
- Before reprojection loss, trainer recovers predictions:

```text
P_ref_norm -> P_ref -> P_world -> normalized training world space (if coord_sigma active)
```

- Eval now performs the same recovery before DSAC / PnP.
- Checkpoints now store the reference recovery metadata inside `lmc_config`.

Main code locations:

- `trainer_dinov2_lmc.py`
- `test_ace_dinov2_lmc.py`
- `../utils_lmc.py`

Status:

- Implemented.
- Syntax-level validation done.
- Full end-to-end C1 training/eval experiment still pending.

### 2.2 Memory metadata propagation

Implemented:

- `load_memory_features()` now returns:
  - `mode`
  - `contract_mode`
  - `reference_index`
  - `points_ref`
  - `points_ref_norm`
  - `conditioning_reference`
  - `normalization_ref`
  - `selection`

Reason:

- Before this change, trainer/eval could not actually see the C1 contract
  metadata even if extraction had written it.

Status:

- Implemented.

### 2.3 Phase 0 report instrumentation

Implemented:

- `memory_policy_report.json` generation is present.
- Report includes selection, coverage, topology, reference-risk, probe, stop,
  and config information.

Status:

- Implemented.
- Good enough for offline comparison of FPS / ASB / adaptive ASB.

### 2.4 Anchor-support root/reference fix

Problem found:

- `anchor_support` originally used the frame nearest the global camera-center
  mean as the reference/root.
- On `indoor6/scene2a_train`, this produced a weak root:
  - selected root was `2355`
  - local positive covisibility support was much weaker than alternative
    center-near candidates
  - in one observed selected set, the root had zero positive covis edges to the
    final selected 28-view batch

Implemented fix:

- Added `select_anchor_support_reference()` to score center-near candidates by
  local/global covisibility support.
- `anchor_support` now uses this scored reference instead of blindly using the
  scene-center nearest frame.

Status:

- Implemented.
- Rerun and downstream validation completed on `scene2a_train`.
- The root fix alone was not sufficient, but it provided the candidate pool
  needed for the Phase 3 gate to pick a much better reference.

### 2.5 Q1a reference-swap diagnostic

Implemented:

- New script:

```text
memory_extraction/reference_swap_diagnostic.py
```

- This script:
  1. builds one selected set,
  2. chooses multiple candidate references from that same set,
  3. reorders the batch so each candidate becomes view 0,
  4. reruns MapAnything `infer()` for each reference,
  5. runs PoseEval for each run,
  6. writes a JSON report.

Report contents include:

- selected set metadata
- candidate reference list
- per-reference PoseEval summaries
- per-view errors
- pairwise drift between references

Status:

- Implemented.
- CLI help tested successfully in the `mapanything_new` environment.
- First target-scene run confirmed that the new root selector changes the
  selected set and promotes `1682` to the selected root.
- Full same-selected-set multi-reference report completed for:
  - `memory_extraction/04_evaluation/reference_swap/scene2a_train/reference_swap_report_20260421_151731.json`

Key result from that report:

- selected batch size reached the requested 40 views
- tested references: `1682`, `2850`, `1181`, `3419`
- mean translation error:
  - `1181`: `0.0949 m`
  - `2850`: `0.1031 m`
  - `3419`: `0.1389 m`
  - `1682`: `0.1519 m`
- pairwise drift between `1181` and `2850` is very small:
  - translation abs diff mean: `0.0098 m`
  - rotation abs diff mean: `0.126 deg`
- therefore the fixed root `1682` is not catastrophic, but it is still a poor
  single-choice default relative to stronger candidates inside the same batch

### 2.6 Phase 3 minimal gate

Implemented:

- Added minimal same-selected-set Phase 3 gating to
  `memory_extraction/run_memory_extraction.py`.
- Added shell/env integration to `memory_extraction/extract_memory.sh`.
- New behavior when enabled:
  1. generate Top-M reference candidates from the fixed selected set,
  2. run a cheap light gate from graph/coverage/reference-risk stats,
  3. run bounded MapAnything probe gate on light-pass candidates,
  4. write `policy_decision.json`,
  5. reorder the final extraction batch to the selected reference.

Current design choice:

- The gate is opt-in, not default-on.
- If the gate predicts `cluster_branch` before Phase 4 exists, extraction still
  continues with the best probe candidate as a single-forward debug fallback,
  and this is recorded in `policy_decision.json`.

New knobs:

- `--enable_reference_policy_gate`
- `--reference_policy_top_m`
- `--reference_policy_light_min_selected_links`
- `--reference_policy_probe_q90_m`
- `--reference_policy_probe_max_m`

Status:

- Implemented.
- Syntax validation done.
- End-to-end extraction validation completed on `scene2a_train`.
- Downstream ACE-G training validation completed on `scene2a`.

### 2.7 Phase 3 gated training validation on scene2a

Implemented/validated:

- Confirmed that the `scene2a` Phase 3 training run actually used the gated
  memory:
  - memory path:
    `memory_extract/.../40v_v0.05_bilinear_bse_ut0.02_noaug_asb_adaptive_pool1.0_rgate4_sor_gm_l2/20260421_154005/memory_bse.pt`
  - stable S1 settings:
    - `s1_use_buffer=True`
    - `s1_loss_mode=sample_per_image`
    - `s1_batch_size=16`
    - `buffer_batch_size=1`
    - `buffer_on_cpu=True`
- Compared the new Phase 3 run against the older Phase 0 baseline:
  - Phase 0 best:
    - `best_iter=2`
    - `pct5=56.0311`
    - `pct2=12.8405`
    - `pct1=3.8911`
    - `median_tErr=4.5737 cm`
    - `median_rErr=0.4758 deg`
  - Phase 3 best:
    - `best_iter=27`
    - `pct5=72.3735`
    - `pct2=26.0700`
    - `pct1=6.6148`
    - `median_tErr=3.4089 cm`
    - `median_rErr=0.3652 deg`

Interpretation:

- The new gated memory is not a no-op.
- The main reference-choice issue is sufficiently validated for this scene.
- The next quality ceiling is now dominated by coverage holes rather than by
  the previous weak-root failure mode.

Status:

- Implemented and validated.

### 2.8 Fixed-budget post-repair for coverage tail

Implemented:

- Added a new post-selection repair helper in
  `memory_extraction/run_memory_extraction.py`.
- The repair runs only after the selected batch size is already fixed.
- It does not add views; it only swaps selected views.
- Current policy:
  - prefer dropping fillers first
  - then supports
  - anchors last
  - reject swaps that violate basic topology or reference-risk limits
  - accept swaps only when they materially reduce coverage tail
- Added CLI/config knobs:
  - `--covis_post_repair_asb`
  - `--covis_post_repair_max_swaps`
  - `--covis_post_repair_tail_percentile`
  - `--covis_post_repair_min_tail_improvement_m`
- Added shell/env forwarding in `memory_extraction/extract_memory.sh`:
  - `ASB_POST_REPAIR`
  - `ASB_POST_REPAIR_MAX_SWAPS`
  - `ASB_POST_REPAIR_TAIL_PERCENTILE`
  - `ASB_POST_REPAIR_MIN_TAIL_IMPROVEMENT_M`
- Added a second-stage improvement after the first successful validation:
  - repair now balances top-K uncovered owner-clusters instead of optimizing
    only the single worst uncovered region
  - repair metadata now flows into `memory_policy_report.json`
  - new knob:
    - `ASB_POST_REPAIR_CLUSTER_TOP_K`

Validation:

- `python -m py_compile memory_extraction/run_memory_extraction.py`
- `bash -n memory_extraction/extract_memory.sh`
- First scene validation run completed on `scene2a_train`.
- After the stronger repair patch, `python -m py_compile
  memory_extraction/run_memory_extraction.py` still passes.
- Stronger repair extraction validated on:
  - `memory_extraction/04_evaluation/memory_extract/scene2a/40v_v0.05_bilinear_bse_ut0.02_noaug_asb_adaptive_pool1.0_rgate4_prepair8_sor_gm_l2/20260422_122243`
- Downstream ACE-G training from that repaired extraction completed on
  `scene2a`:
  - run:
    `04_evaluation/train_compare/memory_pooled_vs_asb/indoor6_ace/scene2a/dino_ace_lmc_ace_g/scene2a_asb40_phase4_c0_20260422_122243_aceg_fS2cie_global_res518_buf2.6M_F7.7M_K64_it28_ep24_bs5120_spi_s1buf_sp384_onecycle_improved`
  - best checkpoint:
    - `best_iter=27`
    - `pct5=76.6537`
    - `pct2=26.0700`
    - `pct1=5.0584`
    - `median_tErr=3.0150 cm`
    - `median_rErr=0.3266 deg`
  - post-train eval:
    - `pct5=74.7082`
    - `pct2=23.7354`
    - `pct1=6.2257`
    - `median_tErr=3.0989 cm`
    - `median_rErr=0.3319 deg`
- After the top-K/report patch:
  - `python -m py_compile memory_extraction/run_memory_extraction.py`
  - `python -m py_compile memory_extraction/reference_swap_diagnostic.py`
  - `bash -n memory_extraction/extract_memory.sh`
- Top-K/report extraction validation completed on:
  - `memory_extraction/04_evaluation/memory_extract/scene2a/40v_v0.05_bilinear_bse_ut0.02_noaug_asb_adaptive_pool1.0_rgate4_prepair8_sor_gm_l2/20260423_103144`
  - structured `repair` report:
    - `enabled=true`
    - `applied=true`
    - `swap_count=1`
    - `cluster_top_k=3`
    - `drop_idx=3909`
    - `add_idx=4294`
    - coverage mean `0.7163 -> 0.6089 m`
    - coverage p95 `1.9432 -> 1.5291 m`
    - coverage max `3.7155 -> 2.3385 m`
    - graph components stayed `4 -> 4`
    - isolated count stayed `0 -> 0`
  - final policy/probe:
    - selected reference `2850`
    - probe translation mean `0.0641 m`
    - probe translation q90 `0.0984 m`
    - probe translation max `0.1385 m`
- Cross-scene top-K/report extraction validation completed on:
  - `memory_extraction/04_evaluation/memory_extract/scene3/40v_v0.05_bilinear_bse_ut0.02_noaug_asb_adaptive_pool1.0_rgate4_prepair8_sor_gm_l2/20260423_105122`
  - policy decision:
    - `decision=cluster_branch`
    - `reason_codes=["no_candidate_passed_probe_gate", "best_probe_fallback_saved_for_debug"]`
    - selected fallback reference `613`
  - repair report:
    - `enabled=true`
    - `applied=false`
    - `swap_count=0`
    - coverage unchanged at mean `0.7451 m`, p95 `2.4492 m`, max `5.6328 m`
    - graph unchanged at `2` components and `0` isolated views
  - candidate probe q90/max translation:
    - `2672`: `0.4556 / 0.5683 m`
    - `613`: `0.3631 / 0.5169 m`
    - `2176`: `0.5130 / 0.6583 m`
    - `4051`: `0.4655 / 0.5545 m`
- `scene3_train` larger-view diagnostic completed on:
  - `memory_extraction/04_evaluation/memory_extract/scene3/60v_v0.05_bilinear_bse_ut0.02_noaug_asb_adaptive_pool1.0_rgate6_prepair12_sor_gm_l2/20260423_111442`
  - policy decision:
    - `decision=cluster_branch`
    - `reason_codes=["no_candidate_passed_probe_gate", "best_probe_fallback_saved_for_debug"]`
    - selected fallback reference `3136`
  - repair report:
    - `enabled=true`
    - `applied=false`
    - `swap_count=0`
    - coverage mean improved from the 40-view run (`0.7451 -> 0.6987 m`)
    - coverage p95/max stayed at `2.4492 / 5.6328 m`
    - graph stayed at `2` components and `0` isolated views
  - best fallback probe improved but still failed:
    - 40-view best `613`: mean/q90/max `0.1954 / 0.3631 / 0.5169 m`
    - 60-view best `3136`: mean/q90/max `0.1483 / 0.2537 / 0.4395 m`
    - q90 and max remain above thresholds `0.18 / 0.25 m`

Status:

- Implemented.
- First validation showed the original swap heuristic was too weak.
- The stronger second-pass heuristic is now validated on `scene2a_train`.
- Downstream training from the stronger repaired extraction is validated and is
  the current best `scene2a` result.
- Top-K balancing/report instrumentation is validated on `scene2a_train`; it
  preserved the already-trained repaired selection and now emits the intended
  structured `repair` report.
- Cross-scene validation on `scene3_train` shows the current policy can produce
  a negative decision (`cluster_branch`) and should not be forced into another
  single-forward training run without additional evidence.
- The 60-view diagnostic reduces error but still fails the probe gate, so
  `scene3_train` should be treated as a likely true Phase 4 cluster fallback
  target.

## 3. What Has Not Been Implemented Yet

### 3.1 Coverage-tail repair for anchor_support

Missing:

- Phase 4 cluster fallback implementation. `scene3_train` now provides a
  concrete target case where 40 and 60 view single-forward policies both fail
  the probe gate.
- Calibration of top-K repair behavior and report outputs beyond scenes that
  are already single-domain feasible.
- Downstream training only if a future extraction changes the selected set,
  reference, or coverage/probe stats relative to the already-trained repaired
  memory.

Current evidence motivating this work:

- The repaired `20260422_122243` extraction selects a good reference (`2850`)
  from the validated `1181/2850` tier.
- Its probe/final PoseEval mean translation is about `0.064 m`.
- Its coverage tail improves materially over the earlier gated extraction:
  - mean `0.716 -> 0.609 m`
  - p95 `1.943 -> 1.529 m`
  - max `3.715 -> 2.338 m`
- The downstream ACE-G run from repaired memory is now the current best
  `scene2a` result.
- The `20260423_103144` top-K/report extraction reproduces the same selected
  set/reference/coverage/probe behavior and adds structured repair diagnostics,
  so it does not by itself require another same-scene training run.
- The `scene3_train` `20260423_105122` extraction is a negative control:
  repair is no-op, all candidate references fail probe, and policy predicts
  `cluster_branch`.
- The 60-view diagnostic `20260423_111442` improves but does not rescue
  single-forward feasibility, so the next engineering work should move to
  cluster fallback rather than more single-forward training.

### 3.2 Q1b minimal predictor C0 vs C1

Missing:

- Formal closeout of the `scene2a` same-selected-set `C0 vs C1` comparison in
  the progress / handoff documents.
- No final `contract_mode_default` decision has been recorded in this file.

Current evidence already available elsewhere:

- `scene2a` repaired `C0` remains the best validated single-memory baseline.
- `scene2a` single-memory `C1` now completes extraction + train + eval
  end-to-end.
- The current same-recipe `C1` result does not beat repaired `C0` on `acc5`
  or median translation.

Status:

- Partial. The experiment evidence exists, but this file has not yet been
  brought up to date with the final Q1b conclusion.

### 3.3 Phase 2 selector completion

Partially implemented only.

Implemented pieces:

- adaptive ASB
- reference-risk-aware heuristics
- auto-stop behavior
- policy report output
- better root/reference selection

Still missing relative to the plan:

- explicit stop metadata around the full score decomposition
- formal repair pipeline
- selector behavior fully structured as the plan describes

Status:

- Partial.

### 3.4 Phase 3 two-level feasibility gate

Missing:

- formal/lightweight gate calibration beyond the current minimal thresholds
- decision-quality validation on more than one scene
- stricter handling for `cluster_branch` once Phase 4 exists

Status:

- Partial.

### 3.5 Phase 4 cluster fallback

Missing:

- cluster planner
- clustered package schema creation
- low-confidence cluster handling

Status:

- Not implemented.

### 3.6 Phase 5 query-time relocalization interface

Missing:

- clustered query-time relocalization
- cluster result ranking / selection
- query-time ensemble interface

Status:

- Not implemented.

### 3.7 Future multi-scene normalized-C1 training line

Deferred, not part of the current memory-extraction closeout:

- weaker normalization / scaled normalized target
- metric auxiliary loss on recovered `points_ref` or `points_world`
- large-scale multi-scene shared pretraining experiments

Tracking location:

- `/home/xwh/project/ace_depth/ace_dinov2_lmc/C1_NORMALIZED_PRETRAINING_MEMO.md`

Status:

- Design memo only. No implementation or runnable CLI flag exists yet.

## 4. Current Diagnosis About scene2a_train

Observed from gated extraction and training results:

- The old weak-root diagnosis was real, but it is no longer the dominant
  blocker on `scene2a_train`.
- The minimal Phase 3 gate selected reference `1181` instead of default `1682`.
- Gated extraction reduced probe/final PoseEval mean translation to about
  `0.068 m`.
- The repaired extraction selected reference `2850`, executed a real coverage
  repair swap, and reduced probe/final PoseEval mean translation to about
  `0.064 m`.
- The full Phase 0 rerun invalidates the earlier "old Phase 0 was far worse"
  comparison: Phase 0 rerun reaches `pct5=73.15`, so the older `pct5=56.03`
  result should be treated as under-trained.
- The repaired Phase 4 run is still the current best downstream result:
  `pct5=76.65`, `pct2=26.07`, median translation `3.01 cm`.
- The top-K/report extraction `20260423_103144` reproduces the same useful
  repair and now writes the intended structured report.

Important conclusion:

- The result should not be interpreted as "C0 is bad".
- The selector is now good enough to beat a full Phase 0 rerun on this scene,
  but the evidence is strongest for the combined reference gate + repair path,
  not for reference gating alone.
- The most defensible current finding is now:

```text
reference gating solved the main root-choice failure on scene2a_train, and the
first repaired selection gives the best downstream result; the newer top-K
repair/report variant is now validated on this scene, so the remaining work is
transfer beyond one scene
```

Evidence:

- Same-set reference sensitivity was confirmed by Q1a, and the gate responded
  correctly by choosing `1181` in Phase 3 and `2850` after repair.
- The full Phase 0 rerun reached:
  - `best_iter=28`
  - `pct5=73.1518`
  - `pct2=19.0661`
  - `pct1=4.2802`
  - median translation `3.3740 cm`
- The repaired Phase 4 run reached:
  - `best_iter=27`
  - `pct5=76.6537`
  - `pct2=26.0700`
  - `pct1=5.0584`
  - median translation `3.0150 cm`
- The repaired extraction improved coverage from the earlier gated tail:
  - mean `0.716 -> 0.609 m`
  - p95 `1.943 -> 1.529 m`
  - max `3.715 -> 2.338 m`
- The structured repair report records graph health as stable:
  - component count `4 -> 4`
  - isolated count `0 -> 0`

Operational conclusion:

- Do not spend more time questioning whether Phase 3 gating is useful on this
  scene; the remaining question is how much benefit comes from gating alone
  versus gating plus repair.
- Do not keep rerunning the old under-trained Phase 0 baseline; use the
  completed Phase 0 rerun as the comparison point.
- The immediate implementation target is cross-scene extraction validation, not
  launching another unchanged training run on the same repaired `scene2a`
  selected set.

## 4.1 Current Diagnosis About scene3_train

Observed from the top-K/report extraction:

- Run:
  `memory_extraction/04_evaluation/memory_extract/scene3/40v_v0.05_bilinear_bse_ut0.02_noaug_asb_adaptive_pool1.0_rgate4_prepair8_sor_gm_l2/20260423_105122`
- The selected set reached 40 views, but coverage still failed:
  - mean `0.745 m`
  - p95 `2.449 m`
  - max `5.633 m`
- Repair was enabled but made no swap:
  - `applied=false`
  - `swap_count=0`
  - graph stayed healthy enough: `2` components, `0` isolated views
- The reference policy selected fallback reference `613`, but the policy
  decision was `cluster_branch`.
- All candidate references failed probe:
  - best fallback `613`: translation mean `0.195 m`, q90 `0.363 m`, max
    `0.517 m`
  - no candidate was close to the current `q90=0.18 m`, `max=0.25 m`
    thresholds
- Larger-view diagnostic:
  `memory_extraction/04_evaluation/memory_extract/scene3/60v_v0.05_bilinear_bse_ut0.02_noaug_asb_adaptive_pool1.0_rgate6_prepair12_sor_gm_l2/20260423_111442`
- Increasing to 60 views helped but did not make the single-forward path
  feasible:
  - selected fallback reference changed to `3136`
  - coverage mean improved `0.745 -> 0.699 m`
  - coverage p95/max stayed `2.449 / 5.633 m`
  - best probe translation improved to mean `0.148 m`, q90 `0.254 m`, max
    `0.440 m`
  - q90/max still fail the `0.18 / 0.25 m` thresholds

Important conclusion:

```text
scene3_train is the first concrete negative policy example: the current
single-forward selector/report path works diagnostically, but the scene remains
single-forward infeasible even after the 60-view budget diagnostic
```

Operational conclusion:

- Do not train the saved `scene3` fallback memory as if it were a valid Phase 4
  success.
- Prioritize Phase 4 cluster fallback implementation over more single-forward
  training on this scene.
- The `20260423_111442` 60-view fallback training has now been run as a
  diagnostic. Its best-selection eval reached `5cm/5deg=69.52%`; its automatic
  post-train re-eval reached `5cm/5deg=67.94%`. Keep this label as diagnostic
  fallback and compare it against the gate prediction.

## 5. Commands To Resume Immediately

### 5.1 Q1a result on scene2a_train

Completed report:

```bash
/home/xwh/project/ace_depth/ace_dinov2_lmc/memory_extraction/04_evaluation/reference_swap/scene2a_train/reference_swap_report_20260421_151731.json
```

Result summary:

- best reference among tested candidates: `1181`
- close second and highly consistent with best: `2850`
- current default root `1682` is significantly worse on translation
- current scene therefore needs candidate gating, not just one root heuristic

### 5.2 Best validated repaired extraction artifacts

Current best repaired extraction:

```bash
/home/xwh/project/ace_depth/ace_dinov2_lmc/memory_extraction/04_evaluation/memory_extract/scene2a/40v_v0.05_bilinear_bse_ut0.02_noaug_asb_adaptive_pool1.0_rgate4_prepair8_sor_gm_l2/20260423_103144
```

Key files:

- `policy_decision.json`
  - selected reference: `2850`
  - decision: `single_forward`
  - candidate probe mean translations:
    - `2850`: `0.0641 m`
    - `1572`: `0.0740 m`
    - `2637`: `0.0904 m`
    - `1682`: `0.1167 m`
- `extraction_log.txt`
  - post-repair swap: `drop=3909 -> add=4294`
  - coverage max: `3.7155 -> 2.3385 m`
  - coverage p95: `1.9432 -> 1.5291 m`
  - coverage mean: `0.7163 -> 0.6089 m`
- `memory_policy_report.json`
  - probe mean translation: about `0.064 m`
  - structured `repair` section is present
  - graph health stayed stable through repair:
    - component count `4 -> 4`
    - isolated count `0 -> 0`
- `selection_coverage_report.json`
  - long-tail max distance: about `2.338 m`

Previously trained repaired extraction:

```bash
/home/xwh/project/ace_depth/ace_dinov2_lmc/memory_extraction/04_evaluation/memory_extract/scene2a/40v_v0.05_bilinear_bse_ut0.02_noaug_asb_adaptive_pool1.0_rgate4_prepair8_sor_gm_l2/20260422_122243
```

It has the same selected set/reference/coverage/probe behavior as
`20260423_103144`, but lacks the structured top-K `repair` report.

### 5.3 Best validated training baseline

Current best downstream training run using repaired memory:

```bash
/home/xwh/project/ace_depth/ace_dinov2_lmc/04_evaluation/train_compare/memory_pooled_vs_asb/indoor6_ace/scene2a/dino_ace_lmc_ace_g/scene2a_asb40_phase4_c0_20260422_122243_aceg_fS2cie_global_res518_buf2.6M_F7.7M_K64_it28_ep24_bs5120_spi_s1buf_sp384_onecycle_improved
```

Best metrics from `best_checkpoint_meta.json`:

- `best_iter=27`
- `pct5=76.6537`
- `pct2=26.0700`
- `pct1=5.0584`
- `median_tErr=3.0150 cm`
- `median_rErr=0.3266 deg`

Full Phase 0 rerun comparison:

```bash
/home/xwh/project/ace_depth/ace_dinov2_lmc/04_evaluation/train_compare/memory_pooled_vs_asb/indoor6_ace/scene2a/dino_ace_lmc_ace_g/scene2a_asb40_phase0_c0_rerun_20260422_aceg_fS2cie_global_res518_buf2.6M_F7.7M_K64_it28_ep24_bs5120_spi_s1buf_sp384_onecycle_improved
```

Best metrics:

- `best_iter=28`
- `pct5=73.1518`
- `pct2=19.0661`
- `pct1=4.2802`
- `median_tErr=3.3740 cm`
- `median_rErr=0.3534 deg`

## 6. Recommended Next Steps

Ordered next steps:

1. Treat the repaired Phase 4 `scene2a` run as the current best single-memory
   `C0` baseline, and treat the completed `scene2a` single-memory `C1` run as
   evidence that `C1` works end-to-end but does not yet beat repaired `C0`.
2. Treat the existing 4090 `scene2a` aux-ref exploratory pair as negative for
   expansion: aux-ref improved `acc25`/`acc10` but lost on `acc5`, so do not
   launch an all-scene aux-ref sweep from this setting.
3. For the requested 4090 full baseline, use
   `memory_extraction/run_indoor6_full_baselines_4090.sh`. Start with
   `DRY_RUN=true` to confirm the matrix, then run the default extraction +
   training matrix on GPU 0/1.
4. Formally close out `Q1b` in the handoff / progress layer: keep default
   single-memory policy on repaired `C0`; do not promote `C1` to default yet.
5. Do not launch another unchanged single-memory run on `scene2a`, `scene3`,
   or additional cross-scene same-recipe `C1` scenes. The current missing
   evidence is no longer “does single-memory `C1` run?”.
6. `scene3 cluster-local C1` is now closed end-to-end:
   - `C1` cluster-fallback extraction validated on `20260430_222156`
   - `cluster_01` and `cluster_02` both trained successfully
   - ensemble eval reached `acc25=97.78`, `acc5=68.57`,
     median `0.6657 deg / 3.3866 cm`
   - this is close to the older cluster-local `C0` ensemble but still
     slightly weaker overall
7. Do not spend more GPU time on more same-shape `scene3` single-memory or
   repeated cluster-local `C1` reruns unless a deterministic-eval discrepancy
   needs to be resolved.
8. Keep the future multi-scene training ideas separate from the current
   memory-extraction mainline. Track weaker normalization (`alpha > 1`) and
   metric auxiliary loss only in
   `/home/xwh/project/ace_depth/ace_dinov2_lmc/C1_NORMALIZED_PRETRAINING_MEMO.md`
   as a separate research line.
9. After `scene3 cluster-local C1`, the next optional mechanism study
   should be a strictly single-scene `scene2a` matrix, not a new cross-scene
   sweep. Keep it minimal:
   - `C1 points_ref_norm(alpha=2)`
   - a revised auxiliary signal only if it provides real supervision on
     `indoor6_ace`; the existing aux-ref exploratory pair did not justify
     expansion
   - one third run only if needed (`alpha=4`, tuned aux weight, or the
     combined variant)
10. Treat these post-mainline mechanism studies as diagnosis of normalized-C1
   fine-precision behavior. Evaluate them primarily by:
   - `acc5`
   - median translation
   while requiring `acc25` to remain close to the current `C1` baseline.

## 7. Validation Performed So Far

Completed:

- AST parsing succeeded for:
  - `trainer_dinov2_lmc.py`
  - `test_ace_dinov2_lmc.py`
  - `../utils_lmc.py`
  - `memory_extraction/run_memory_extraction.py`
  - `memory_extraction/reference_swap_diagnostic.py`

- `reference_swap_diagnostic.py --help` succeeded under:
 - `reference_swap_diagnostic.py --help` succeeded in the
   `mapanything_new` environment.

```bash
conda activate mapanything_new
python -m ace_dinov2_lmc.memory_extraction.reference_swap_diagnostic --help
```

Not yet completed:

- deterministic eval should remain available as a comparison tool, but it does
  not need to be forced as the always-on default; some random DSAC variation is
  acceptable as long as key judgments are rechecked when needed
- for future cheap evals, key checkpoints should by default be evaluated
  `5` times and recorded as best-of over the saved `eval_summary_*.txt` files
- existing 4090 `scene2a` aux-ref/control pair has 5 saved deterministic eval
  summaries each; aux-ref did not beat control on `acc5`
- the first `scene2a` weaker-normalization (`alpha > 1`) mechanism run
- shared-model multi-memory training path
- clustered package as a first-class training input interface

## 8. Notes For Future Conversations

If a future conversation resumes this work, the first actionable question
should be:

```text
Has Phase 4 cluster fallback construction been implemented and validated on
scene3_train?
```

If the answer is yes, the next actionable question should be:

```text
If yes, what are the per-cluster references, coverage/probe stats, and any
low-confidence cluster reasons?
```
