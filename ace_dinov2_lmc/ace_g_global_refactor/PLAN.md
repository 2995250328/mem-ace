# ACE-G Global Refactor Plan

Scope: `--lmc_flow ace_g` with effective `lmc_mode=global`.

Maintenance rule: `PLAN.md` is the source-of-truth plan. Any structural
change to this file must be mirrored in `PLAN_CN.md` as the synchronized
Chinese translation. Do not maintain a separate `ROADMAP_CN.md`.

This file is the high-level tracker, priority map, and onboarding summary for
new agents. It should be detailed enough to understand the problem and intended
solution without immediately reading source code. Concrete implementation notes
still live in one file per step under `steps/`.

## Status Legend

- `[todo]`: not started.
- `[designing]`: goal accepted, exact implementation still being discussed.
- `[ready]`: concrete implementation measure has been agreed.
- `[done]`: code/docs verified for this step.
- `[deferred]`: moved out of the near-term ACE-G global track.

## Current Problem Map

### P0 - Experiment Semantics Were Not Reliable

Symptoms:

- Old commands could say `--lmc_mode global` while the runtime silently switched
  to `local`.
- Global FPS used a random starting point, so the same memory/checkpoint could
  produce different latent coordinates.
- Some structure-affecting config was implicit or not saved in checkpoint
  metadata.

Risk:

- A result can be mislabelled as global while actually being effective-local.
- S1/S2/test can see different compressed memory distributions.
- Layer and geometry ablations become impossible to interpret.

Status:

- Mostly fixed. See steps 01, 02, 03, and 04.

### P1 - Loss Contracts Were Partly Duplicated

Symptoms:

- S1 sampled, S1 full-map, S2, and S2-G have overlapping reprojection/invalid
  loss logic.
- Boundary behavior can diverge: dyntanh step, invalid clamp, masks, and
  sampled/full-map handling.

Risk:

- We may think S1/S2 supervision is equivalent when it is not.
- Later architectural changes can be blamed incorrectly when the true cause is
  a loss-path difference.

Status:

- Fixed for the current compatibility-preserving path. See step 07.

### P2 - Compressor Geometry Is Hard To Interpret

Symptoms:

- Key feature selection was implicit; it is now explicit, but not yet ablated.
- Current default is effectively slice 2 / layer 12.
- The current multi-layer memory contract is conservative: key uses one selected
  layer, while value keeps all-layer concatenation. It is not yet decided
  whether a later LMC should use a DPT-style fused memory feature or explicit
  multi-level compression/fusion.
- PE and distance bias use raw scene coordinates, so geometry frequency/scale
  differs across scenes.
- Distance bias currently uses a scalar `log(dist_sq)` style signal.

Risk:

- Key layer choice may be a hidden bottleneck.
- Geometry features may mean different things in different scene scales.
- Compressor attention may overfit to raw coordinate scale rather than useful
  relative geometry.

Status:

- Contract partially fixed; ablations and scene-scale contract are not done.
  See `steps/06_compressor_geometry_contract.md`.

### P3 - Fusion Geometry Does Not Affect Matching Logits

Symptoms:

- Current fusion uses geometry only in the value path:
  `k = k_proj(memory_z)`, `v = v_proj(memory_z + PE(memory_p))`.
- Query-to-token matching logits do not directly see `memory_p`.

Risk:

- The model can only use geometry after a token has already been selected.
- Global tokens may be selected by feature similarity alone even when geometry
  should disambiguate.

Status:

- Implemented as Progressive Geometry Injection / GeoKey v0. The first fusion
  change is a falsifiable memory-side geometry probe, not full GeoMatch. A0/A1/A2
  experiments are pending. See step 05.

### P4 - Token Usage Needed Observability

Symptoms:

- We do not know whether global K=64 tokens are used broadly, collapse to a few
  tokens, or remain too diffuse.
- We do not log attention entropy, effective token count, avg max attention, or
  raw/fused feature norm changes.

Risk:

- Usage loss or MoE-style routing could be added for the wrong reason.
- Fusion improvements cannot be explained beyond final pose metrics.

Status:

- Diagnostic-only runtime observability is implemented behind an explicit flag.
  See step 10.

### P5 - Future Multi-Scene LMC Needs A New System Contract

Symptoms:

- Current training is still single-scene, single-memory, FPS-based ACE-G.
- Future goal is a truly reusable LMC trained jointly across scenes.
- Current memory features come from a MapAnything/reference-frame feature path,
  while point coordinates are still primarily handled as world coordinates.
- Large scenes may require sub-memory construction and routing.
- Memory files already contain ray / Plucker-style geometry that is mostly
  unused by the current compressor/fusion/head.

Risk:

- Multi-scene training can be dominated by large-scale scenes unless coordinate
  normalization is explicit.
- Reference-frame features and world-frame points can create an avoidable
  coordinate-system gap.
- Sub-memory routing can fail whole queries if the wrong memory shard is chosen.
- MoE heads or geometry/ray features can add complexity before the baseline is
  observable enough to justify them.

Status:

- Long-term system design item. Do not mix into current true-global baseline
  changes. See `LONG_TERM_RESEARCH_PLAN.md`.

## Priority Order

## Classification Snapshot

This section classifies the current design ideas into completed work,
near-term ACE-G architecture work, and long-term system work. It is meant to
avoid mixing a baseline cleanup with larger research changes.

### Already Done / Current Baseline Contract

- Deterministic FPS is implemented and logged through `lmc_fps_start_policy`.
- Requested/effective LMC mode is explicit, and requested mode is authoritative
  by default unless visibility fallback is explicitly enabled.
- The old key-layer ambiguity is fixed through `lmc_key_slice_idx`, while the
  compatibility default remains slice 2 / layer 12 for five-layer memory.
- Checkpoints now record the structure-affecting LMC metadata needed to
  reconstruct the compressor/fusion path.
- ACE-G S2 compressor freezing is explicit at the parameter trainability
  boundary.
- Sampled S1 ReproLoss time-axis behavior is explicit through
  `s1_loss_step_mode`.

### Near-Term LMC Architecture Work

These items are relevant after the current true-global baseline comparison,
but should be added as controlled ablations with diagnostic logs:

- Diagnostic-only fusion/token observability is already implemented and remains
  the required explanation layer for new fusion ablations.
- Progressive Geometry Injection / GeoKey v0 is implemented behind explicit
  fusion flags: first test scene-scale-normalized value geometry, then add
  memory geometry into fusion keys with a zero-init scalar gate. This does not
  use query 3D coordinates and should not be treated as full GeoMatch.
- Detached token-routing prior is now preferred before any coordinate-based
  coarse GeoMatch. It predicts a distribution over memory tokens and can remain
  multi-modal, instead of forcing a pre-fusion raw feature into a single 3D
  coordinate.
- Compressor key-layer ablations, starting from the current layer 12 baseline.
- Memory feature hierarchy ablations after the basic key-layer result: current
  selected-layer key/all-layer value, key scalar mix, DPT-style fused memory
  feature, or explicit multi-level compressor/fusion.
- Compressor/fusion geometry scale diagnostics and a shared scene-scale
  encoding contract for PE and distance bias.
- Shared reprojection/invalid-loss helper is done for the current
  compatibility-preserving path.

### Long-Term System Work

These items are important, but they change broader contracts than the current
single-scene ACE-G baseline and should not be mixed into the first architecture
ablations:

- Multi-scene joint LMC training.
- Reference-frame point/memory coordinate contract.
- Full coordinate normalization for multi-scene loss balancing.
- Sub-memory construction, routing, and candidate reranking for large scenes.
- Training-free MapAnything/VGGT memory-extraction acceleration.
- Ray / Plucker geometry usage.
- Anchor-assisted coordinate heads and MoE / expert heads.
- DSD-style local/deformable compressors and larger head replacements.

### Priority 0 - Completed Baseline Stabilization

1. `[done]` Deterministic FPS.
   - Problem: random FPS start changed latent coordinates.
   - Fix: deterministic start policy, default `farthest_from_center`.
   - Detail: `steps/01_deterministic_fps.md`

2. `[done]` Explicit low-risk LMC contract.
   - Problem: key layer, config, and S2 compressor freezing were implicit.
   - Fix: explicit key slice, config metadata, S2 trainability boundary.
   - Detail: `steps/02_low_risk_contract.md`

3. `[done]` Explicit S1 sampled loss time-axis.
   - Problem: sampled S1 effectively used hidden fixed-zero dyntanh behavior.
   - Fix: `fixed_zero / per_iter / global_monotonic` mode is explicit.
   - Detail: `steps/03_s1_sampled_loss_step_mode.md`

4. `[done]` Requested LMC mode is authoritative by default.
   - Problem: visibility fallback silently changed global to local.
   - Fix: `lmc_auto_mode_by_visibility=False` by default.
   - Detail: `steps/04_lmc_mode_authority.md`

### Priority 1 - Finish Current True-Global Comparisons

5. `[ready]` Compare true-global baselines before more architecture changes.
   - Required runs: `global + fixed_zero`, `global + per_iter`.
   - Compare against old requested-global/effective-local baseline.
   - Success metric: post-train multi-seed pose metrics plus best iteration and
     training stability logs.
   - Detail: `steps/05_geomatch_near_term.md`

### Priority 2 - Clean Shared Supervision Contract

6. `[done]` Unify reprojection and invalid-loss behavior.
   - Problem: duplicated loss implementations can hide behavior differences.
   - Scope: S1 sampled, S1 full-map, S2, and S2-G.
   - Constraint: preserve current behavior first; only refactor into a shared
     helper with tests/smoke checks.
   - Priority reason: it reduces risk before changing fusion/compressor
     architecture.
   - Detail: `steps/07_loss_contract.md`

7. `[done]` Non-architecture hygiene and fail-fast checks.
   - Problem: a few engineering hazards remained before architecture ablation:
     hidden head-grid trimming, delayed memory feature-dim errors, and preset
     choices drifting from preset definitions.
   - Scope: no loss, fusion, compressor, or head-architecture behavior change.
   - Detail: `steps/08_non_arch_hygiene.md`

8. `[done]` Module train/eval mode restoration contract.
   - Problem: helper functions temporarily switched modules to eval/train mode
     and could restore a broader state than the caller originally had.
   - Scope: compressor memory compression, fused-buffer construction, ACE-G raw
     buffer construction, and exact regressor/encoder/heads mode restoration.
   - Detail: `steps/09_module_mode_contract.md`

### Priority 3 - Add Observability Before New Losses

9. `[done]` Add diagnostic-only compressor/fusion observability and runtime semantics.
   - Problem: token usage and fusion behavior are invisible.
   - Track: attention entropy, effective token count, avg max attention, token
     usage, raw/fused feature norms, compressor geometry stats, and experiment
     semantics in summaries.
   - No new loss and no behavior change when disabled.
   - Detail: `steps/10_runtime_observability_and_semantics.md`

10. `[designing]` Add compressor geometry contract and ablations.
   - Problem: key layer and geometry scale may be hidden bottlenecks.
   - First ablations: layer 12 baseline, layer 18, layer 6, final, learned
     scalar mix.
   - Later ablations should decide whether memory layers stay as selected-key /
     all-layer-value, collapse through a DPT-style feature neck, or remain
     explicit through multi-level compression/fusion.
   - Scene-scale contract comes before PE/distance-bias redesign.
   - Detail: `steps/06_compressor_geometry_contract.md`

### Priority 4 - Controlled Architecture Ablations

11. `[done]` Progressive Geometry Injection / GeoKey v0.
   - Problem: geometry does not affect query-memory matching logits.
   - Implemented ablation modes: A0 current value-only raw PE, A1 value-only
     normalized PE, A2 normalized PE plus memory-side GeoKey.
   - Treat A1/A2 as probes. No result here can by itself prove or disprove full
     query-memory GeoMatch.
   - Hard constraints: keep A0 exactly compatible, initialize the GeoKey scalar
     gate at zero, fix `lmc_key_slice_idx=2` for the first run, and record the
     scene-scale source/value in logs and checkpoint metadata.
   - Detail: `steps/05_geomatch_near_term.md`

12. `[todo]` Usage regularization only if diagnostics justify it.
    - Problem: possible token collapse.
    - Add only after measuring token usage.
    - Start with S1-only `lambda_usage` ablations.
    - Detail: `steps/05_geomatch_near_term.md`

13. `[todo]` Fusion residual gate as a lower-priority ablation.
    - Problem: `LayerNorm(query_feats + attention_out)` may allow fusion to
      heavily reparameterize raw features.
    - Candidate: `query_feats + residual_gate * attention_out`.
    - Lower priority than key/value geometry and diagnostics.
    - Detail: `steps/05_geomatch_near_term.md`

### Priority 5 - Medium/Long-Term Research

14. `[designing]` Detached token-routing prior before coordinate GeoMatch.
    - Keep ACE head as primary output.
    - Prefer predicting a detached memory-token distribution
      `r(token | query)` and adding `g * log(r + eps)` as a weak residual logit
      prior before adding any coarse-coordinate distance bias.
    - This is safer than a single coarse 3D point because it can keep ambiguous
      or repeated-texture patches multi-modal.
    - Only attempt after A0/A1/A2 and attention diagnostics are understood.
    - Detail: `steps/05_geomatch_near_term.md`

15. `[designing]` Confidence-gated coarse-coordinate GeoMatch as a later weak prior.
    - Do not treat detached coarse coordinates as the default next step after
      GeoKey. A bad coarse coordinate can pull attention into the wrong memory
      region and make refinement unrecoverable.
    - If implemented, it must be a fallback-safe residual: zero/warmup global
      gate, per-query confidence, normalized coordinates, nearby reward rather
      than far-token punishment, and image feature logits remain the main path.
    - Anchor-relative branch remains auxiliary only.
    - Detail: `steps/05_geomatch_near_term.md`

16. `[deferred]` Larger research changes.
    - Full normalized-coordinate target training.
    - Reference-frame memory coordinate contract.
    - DSD/local/deformable compressor.
    - Sub-memory routing.
    - Multi-scene joint LMC training.
    - ACE head replacement / MoE architectures.
    - Ray / Plucker geometry usage.
    - P4Pf / uncalibrated-query solver changes.
    - Detail: `LONG_TERM_RESEARCH_PLAN.md`

## Detailed Task Cards

### 1. Deterministic FPS

Status: `[done]`

Problem:

- Global compressor chooses latent coordinates with FPS.
- The old FPS start point came from randomness.
- The same memory and checkpoint could therefore produce different latent token
  coordinates across S1, S2, and test.

Why this matters:

- Latent coordinates affect compressor query PE.
- They affect global attention distance bias.
- They affect fusion PE from `memory_p`.
- They change the fused feature distribution seen by the ACE head.

Implemented direction:

- Use a deterministic FPS start policy by default.
- Current default is `farthest_from_center`.
- Keep `legacy_random` only as explicit opt-in for old behavior.

Verification:

- Recompressing the same memory should choose the same latent coordinates.
- Checkpoint/log metadata should record `lmc_fps_start_policy`.

Detail:

- `steps/01_deterministic_fps.md`

### 2. Explicit Key Layer And Low-Risk LMC Contract

Status: `[done]`

Problem:

- Memory features contain multiple layer slices:
  `0, 6, 12, 18, final`.
- The old compressor key path looked like it selected layer index `1`, but due
  to helper semantics it effectively selected slice `2`, which corresponds to
  layer `12`.
- This was easy to misread and made layer ablations ambiguous.

Why this matters:

- Key features decide memory attention matching.
- A silent key-layer mismatch changes compressor behavior while command names
  and checkpoint names stay the same.

Implemented direction:

- Add explicit `lmc_key_slice_idx`.
- Default multi-layer behavior remains compatible: slice `2`.
- Save/log `lmc_key_slice_idx`, `key_layer_label`, and `layers_idx`.
- Preserve old behavior until ablations prove a better key layer.

Verification:

- Logs show selected key slice and label.
- Checkpoint config contains selected key metadata.
- Test-time reconstruction uses checkpoint config rather than accidental parser
  defaults.

Detail:

- `steps/02_low_risk_contract.md`

### 3. Save Structure-Affecting LMC Config

Status: `[done]`

Problem:

- Some model-structure or behavior-affecting fields were not fully recorded in
  checkpoints.
- Training and testing could reconstruct different structures if parser
  defaults changed later.

Fields of interest:

- requested/effective LMC mode
- `layers_idx`
- selected key slice / key layer label
- `pe_normalize_input`
- `geo_sigma`
- `num_fine`, `num_coarse`
- `backbone_feature_dim`
- FPS start policy
- S1 sampled loss step mode
- future fusion geometry mode

Implemented direction:

- Save low-risk structure metadata into `lmc_config`.
- Log requested/effective mode and key layer metadata.

Verification:

- Checkpoint inspection should show the same config needed to reconstruct the
  compressor/fusion path.
- Old checkpoints should still load through compatibility defaults.

Detail:

- `steps/02_low_risk_contract.md`

### 4. Explicit S2 Compressor Freeze

Status: `[done]`

Problem:

- ACE-G S2 should not train the compressor.
- Old behavior relied on a combination of `no_grad()` and optimizer membership.
- Future code could accidentally live-forward the compressor in S2 and leak
  gradients.

Implemented direction:

- Add an explicit trainability boundary:
  S1 enables compressor training, S2 disables it.
- Preserve current behavior while making the stage contract visible.

Verification:

- S1 logs indicate compressor trainable.
- S2 logs indicate compressor frozen.
- Optimizer parameter groups should not include compressor parameters in S2.

Detail:

- `steps/02_low_risk_contract.md`

### 5. S1 Sampled Loss Step Mode

Status: `[done]`

Problem:

- Sampled S1 used a hidden `iter_for_loss=0` path in legacy mode.
- This means dyntanh stayed at the most permissive step for sampled S1.
- It was unclear whether this was intended curriculum or historical behavior.

Why this matters:

- `fixed_zero` can be more forgiving and may help early compressor learning.
- `per_iter` is stricter and may align better with normal training schedule.
- Without logging, experiments cannot tell which behavior was used.

Implemented direction:

- Add explicit `s1_loss_step_mode`:
  - `fixed_zero`
  - `per_iter`
  - `global_monotonic`
- Keep compatibility default:
  - legacy profile -> `fixed_zero`
  - `mapany_flow_v1` -> `global_monotonic`

Verification:

- Logs show requested/resolved S1 sampled loss step mode.
- Current comparison runs should isolate `fixed_zero` vs `per_iter`.

Detail:

- `steps/03_s1_sampled_loss_step_mode.md`

### 6. LMC Mode Authority

Status: `[done]`

Problem:

- Old code had visibility-based fallback enabled by default.
- A command with `--lmc_mode global` could silently become effective `local`.
- The fallback statistic only measured whether sampled memory points were in
  front of each camera (`z > 0`), which is too coarse for multi-view scene
  point clouds.

Why this matters:

- Old "global" baselines may actually be effective-local.
- Result folders and command lines can be misleading.

Implemented direction:

- `lmc_auto_mode_by_visibility` defaults to `False`.
- Requested `lmc_mode` is authoritative unless fallback is explicitly enabled.
- Logs/checkpoints record requested and effective mode.

Verification:

- True-global runs must show:
  `requested=global effective=global auto_by_visibility=False`.
- Old fallback runs should be treated as requested-global/effective-local unless
  logs prove otherwise.

Detail:

- `steps/04_lmc_mode_authority.md`

### 7. Current True-Global Baseline Comparison

Status: `[ready]`

Problem:

- The old strong scene2a baseline was not true global; it was requested-global
  but effective-local.
- Before architecture changes, we need actual true-global references.

Required comparison:

- old requested-global/effective-local baseline
- true-global + `fixed_zero`
- true-global + `per_iter`

Why this matters:

- If true global is already better, fusion/compressor changes should be judged
  against true global, not old effective-local.
- If `per_iter` hurts, future work should keep `fixed_zero` as the stable S1
  baseline.

Verification:

- Compare best iteration, best score, post-train multi-seed median, and
  25/10/5/2/1 cm recalls.
- Confirm logs have effective mode and S1 step mode.

Detail:

- `steps/05_geomatch_near_term.md`

### 8. Unified Reprojection / Invalid Loss Contract

Status: `[done]`

Problem:

- S1 sampled, S1 full-map, S2, and S2-G contain overlapping reprojection and
  invalid-loss logic.
- Small differences can exist in dyntanh step handling, invalid clamp, masks,
  and sampled/full-map behavior.

Why this matters:

- Architectural changes may be blamed for metric shifts caused by loss-path
  mismatch.
- It is hard to claim S1/S2 supervision is consistent.

Recommended direction:

- First document the exact current behavior of each path.
- Then extract a shared helper with compatibility-preserving behavior.
- Do not intentionally change the math in the same commit as the refactor.

Verification:

- Synthetic parity checks compare the shared helper against copied old S1 and
  S2 formulas.
- `python -m py_compile trainer_dinov2_lmc.py`

Priority:

- Highest code-cleanup priority after current true-global comparisons.

Detail:

- `steps/07_loss_contract.md`

### 9. Non-Architecture Hygiene And Fail-Fast Checks

Status: `[done]`

Problem:

- Head-grid packing for sampled rows was named as trimming rather than the
  actual fake-grid contract.
- Tail trimming was only warned once and had no cumulative stage statistics.
- LMC memory feature-dim mismatch was a warning even though it leads to later
  shape errors.
- `memory_compare_ace_g_v2` existed in preset defaults but was not accepted by
  `--train_preset`.

Decision:

- Preserve current training math and normal experiment behavior.
- Promote only invalid memory shape contracts to immediate failure.
- Expose the already-defined preset without changing older preset defaults.
- Record this change in `steps/08_non_arch_hygiene.md`; future code/config
  changes must update the refactor docs before being considered complete.

Verification:

- `python -m py_compile options_dinov2_lmc.py trainer_dinov2_lmc.py train_ace_dinov2_lmc.py test_ace_dinov2_lmc.py /home/xwh/project/ace_depth/ace_compressor.py`
- CLI parser accepts `--train_preset memory_compare_ace_g_v2`.
- Synthetic fail-fast check confirms non-divisible memory feature dimensions
  raise `ValueError` with config context.

Detail:

- `steps/08_non_arch_hygiene.md`

### 10. Module Train/Eval Mode Restoration Contract

Status: `[done]`

Problem:

- Some helper functions temporarily switched modules to eval mode for
  deterministic feature extraction, then restored with broad calls such as
  `regressor.train()`.
- That can accidentally turn a previously eval-only child module, especially
  the frozen DINO encoder, back to train mode.
- `_compress_memory()` did not restore the compressor mode through a
  `finally` block if compression failed.

Decision:

- Add small helpers to capture and restore exact module `.training` flags.
- Use them around compressor memory compression, fused-buffer construction, and
  ACE-G raw-buffer construction.
- Preserve output math, optimizer membership, loss contracts, and trainability
  boundaries.

Verification:

- `python -m py_compile trainer_dinov2_lmc.py`
- Synthetic smoke check captures/restores mixed parent/child module modes.
- Existing full sanity run remains the metric-level check; no new baseline
  training is required for this state-restoration-only change.

Detail:

- `steps/09_module_mode_contract.md`

### 11. Runtime Observability And Experiment Semantics

Status: `[done]`

Problem:

- We cannot currently tell whether K=64 global tokens are used well.
- No attention usage, entropy, effective token count, or raw/fused norm metrics
  are logged.

Why this matters:

- Usage loss is risky unless collapse is proven.
- Progressive Geometry Injection improvements should be explainable by
  routing/token behavior, not only pose metrics.

Recommended direction:

- Add optional diagnostics that are off by default.
- Save requested/effective experiment semantics in eval and best-checkpoint
  summaries.
- Add S1/S2-G trainability and optimizer contract guards.
- Do not add loss and do not change outputs when diagnostics are disabled.

Metrics:

- attention entropy
- token usage
- effective token count
- average max attention
- raw feature norm
- attention output norm
- fused feature norm
- geometry gate/scale values when GeoKey is enabled

Verification:

- Diagnostics can be enabled for selected runs.
- Normal runs remain unchanged when diagnostics are disabled.
- Eval summaries include requested/effective mode, key slice, FPS policy, S1
  step mode, and ACE-G fusion mode.

Detail:

- `steps/10_runtime_observability_and_semantics.md`

### 12. Compressor Geometry Contract And Ablations

Status: `[designing]`

Problem:

- Key layer is explicit but not yet ablated.
- Multi-layer memory usage is not yet settled beyond the current
  selected-layer key plus all-layer-concat value contract.
- PE and distance bias still use raw scene coordinates.
- Distance-bias geometry is narrow: mostly scalar distance/log-distance.

Why this matters:

- Layer 12 may not be optimal.
- Geometry encoding can mean different physical scales across scenes.
- Compressor attention may be sensitive to scene size rather than useful
  geometry.

Recommended direction:

- Start with key-layer ablations:
  layer 12, layer 18, layer 6, final, learned scalar mix.
- Then define scene-scale policy.
- After key-layer and scene-scale basics, test memory feature hierarchy
  variants:
  current selected-layer key/all-layer value, key scalar mix, DPT-style fused
  memory feature, and explicit multi-level compressor/fusion.
- Only after scene-scale contract, test PE and distance-bias variants.

Do not mix:

- key-layer ablation
- Progressive Geometry Injection / GeoKey v0
- scene-scale PE changes
- DPT-style or explicit multi-level memory feature fusion

Each should be its own ablation.

Detail:

- `steps/06_compressor_geometry_contract.md`

### 13. Progressive Geometry Injection / GeoKey v0

Status: `[done]`

Problem:

- Current fusion geometry only affects values, not query-memory matching logits.
- Query-side 3D coordinates are not reliable before fusion, so the first
  geometry-logit change must not depend on query geometry.

Recommended direction:

- A0: keep current value-only raw centered PE as the exact baseline.
- A1: value-only with explicit scene-scale-normalized memory geometry.
- A2: A1 plus memory-side GeoKey:
  `k_input = memory_z + key_geo_scale * pe_mem`.
- Fix `lmc_key_slice_idx=2` for the first A0/A1/A2 comparison so key-layer
  ablations do not contaminate fusion geometry results.
- Use scene scale from stable memory metadata or a precomputed memory-point
  statistic, not from per-forward latent coordinates that may later move.

Important compatibility note:

- Current value path is equivalent to `value_geo_scale=1.0`.
- A sigmoid gate initialized at zero gives `0.5`, which is not exactly old
  behavior. Prefer a direct scalar parameter initialized to `0.0` for GeoKey so
  A2 starts exactly as A1.
- A1/A2 are probes. A2 over A1 supports memory-side geometry in keys; A2 flat
  against A1 does not disprove later query-side priors, but it also does not
  justify jumping directly to coordinate-distance bias.
- Implemented CLI:
  `--lmc_fusion_geometry_mode {value_only_raw,value_only_norm,geokey_norm}`,
  `--lmc_fusion_key_geo_init`,
  `--lmc_fusion_scene_scale_source {memory_points_p95,fixed}`, and
  `--lmc_fusion_scene_scale_value`.

Verification:

- Old checkpoints load.
- Logs/checkpoints identify fusion geometry mode, scene-scale source/value, and
  learned key geometry scale.
- A0 reconstructs the current baseline exactly.
- Compare A1/A2 against the true-global `forceglobal_s1_periter` value-only
  baseline and inspect routing diagnostics, not only pose metrics.

Detail:

- `steps/05_geomatch_near_term.md`

### 14. Conditional Usage Regularization

Status: `[todo]`

Problem:

- Token collapse may exist, but it is not yet measured.

Risk:

- Forcing uniform token usage can be wrong if the scene naturally uses some
  tokens more than others.

Recommended direction:

- Add only after diagnostics show collapse.
- Start S1-only.
- Try small weights: `0.001`, `0.003`, `0.01`.

Verification:

- Effective token count improves.
- Pose metrics do not regress.
- Attention maps become more meaningful, not merely uniform.

Detail:

- `steps/05_geomatch_near_term.md`

### 15. Fusion Residual Gate

Status: `[todo]`

Problem:

- Current fusion uses `LayerNorm(query_feats + attention_out)`.
- This can let fusion become a large feature reparameterization rather than a
  controlled memory correction.

Recommended direction:

- Consider `query_feats + residual_gate * attention_out`.
- Lower priority than diagnostics and GeoKey geometry.

Verification:

- Log residual gate value and raw/fused norm ratio.
- Only test after diagnostics exist.

Detail:

- `steps/05_geomatch_near_term.md`

### 16. Detached Token-Routing Prior

Status: `[designing]`

Problem:

- Full query-memory geometry bias wants query-side information, but pre-fusion
  query 3D from raw features is not reliable.
- A single coarse coordinate can be structurally wrong in repeated texture,
  weak texture, or room/region aliasing cases.
- If that wrong coordinate is used as a distance bias, it can reward the wrong
  memory region and suppress recovery by the refined head.

Recommended direction:

- Prefer a detached token-routing prior before coordinate-based GeoMatch.
- Predict a distribution over memory tokens:
  `r_ik = P(memory token k | query i)`.
- Add it to fusion logits only as a weak residual prior:
  `logits = image_logits + g * log(r_ik + eps)`.
- Initialize `g=0` and cap or warm it up until diagnostics show the prior is
  useful.
- Keep the ACE head as the primary output.

Why this comes before coarse coordinates:

- A distribution can remain multi-modal, which is important for ambiguous
  patches.
- It does not require choosing one possibly wrong 3D point before fusion.
- It is closer to a coarse matching prior than to hard coordinate refinement.

Verification:

- Compare against A1/A2 with attention diagnostics enabled.
- Check whether routing entropy, top-token maps, and pose metrics improve
  together.
- Reject or keep diagnostic-only if it merely sharpens wrong attention.

Detail:

- `steps/05_geomatch_near_term.md`

### 17. Confidence-Gated Coarse-Coordinate GeoMatch / Anchor-Assisted Branches

Status: `[designing]`

Problem:

- Coordinate-based GeoMatch needs query-side 3D, but pre-fusion query 3D is not
  reliable.
- Absolute coordinate regression may also be harder than anchor-relative
  residual prediction.

Recommended direction:

- Keep ACE head as the primary output.
- Only consider detached coarse query-coordinate bias after A1/A2 diagnostics
  and the token-routing prior clarify whether memory-side geometry and routing
  are useful.
- Treat coarse-coordinate GeoMatch as a confidence-gated weak prior, not as a
  hard coarse-to-fine refinement assumption.
- Use a zero-init or warmup global gate, detach the coarse coordinate, normalize
  by the same scene-scale contract as A1/A2, and keep image feature logits as
  the dominant matching path.
- Prefer nearby reward, such as `alpha * exp(-d^2 / tau)`, over a global
  negative-distance penalty that suppresses all far tokens.
- Add per-query confidence before applying the bias. Inference-compatible first
  choices are coarse-to-memory nearest distance and/or a predicted uncertainty;
  training-only reprojection validity may be useful for diagnostics but cannot
  be the inference gate.
- Add anchor-relative branch only as auxiliary.
- Start with small loss weights such as `0.05` or `0.1`.

Do not do yet:

- Do not replace ACE head.
- Do not make anchor-only the primary path until auxiliary branch proves useful.
- Do not let coarse coordinates dominate attention logits.
- Do not add a hard distance penalty without a fallback to image logits.

Detail:

- `steps/05_geomatch_near_term.md`
- `LONG_TERM_RESEARCH_PLAN.md`
