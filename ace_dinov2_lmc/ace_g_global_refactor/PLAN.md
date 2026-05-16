# ACE-G Global Refactor Plan

Scope: `--lmc_flow ace_g` with effective `lmc_mode=global`.

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

### P1 - Loss Contracts Are Still Partly Duplicated

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

- Still open. This is the highest-priority cleanup after current global runs.

### P2 - Compressor Geometry Is Hard To Interpret

Symptoms:

- Key feature selection was implicit; it is now explicit, but not yet ablated.
- Current default is effectively slice 2 / layer 12.
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

- Planned as a controlled GeoMatch Fusion v1 ablation. See step 05.

### P4 - Token Usage Is Not Observable

Symptoms:

- We do not know whether global K=64 tokens are used broadly, collapse to a few
  tokens, or remain too diffuse.
- We do not log attention entropy, effective token count, avg max attention, or
  raw/fused feature norm changes.

Risk:

- Usage loss or MoE-style routing could be added for the wrong reason.
- Fusion improvements cannot be explained beyond final pose metrics.

Status:

- Planned before usage regularization. See step 05.

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

- Diagnostic-only fusion/token observability.
- GeoMatch Fusion v1: geometry into fusion key logits while preserving the
  current value-only path as the exact baseline.
- Compressor key-layer ablations, starting from the current layer 12 baseline.
- Compressor/fusion geometry scale diagnostics and a shared scene-scale
  encoding contract for PE and distance bias.
- Shared reprojection/invalid-loss helper, preserving current behavior first.

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

9. `[todo]` Add diagnostic-only compressor/fusion observability.
   - Problem: token usage and fusion behavior are invisible.
   - Track: attention entropy, effective token count, avg max attention, token
     usage, gate values, raw/fused feature norms, PE scale.
   - No new loss and no behavior change when disabled.
   - Detail: `steps/05_geomatch_near_term.md`

10. `[designing]` Add compressor geometry contract and ablations.
   - Problem: key layer and geometry scale may be hidden bottlenecks.
   - First ablations: layer 12 baseline, layer 18, layer 6, final, learned
     scalar mix.
   - Scene-scale contract comes before PE/distance-bias redesign.
   - Detail: `steps/06_compressor_geometry_contract.md`

### Priority 4 - Controlled Architecture Ablations

11. `[todo]` GeoMatch Fusion v1.
   - Problem: geometry does not affect query-memory matching logits.
   - Ablation: `value_only` vs `gated_key_value`.
   - Preserve old value geometry behavior as closely as possible.
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

14. `[designing]` Anchor-assisted residual branch as auxiliary only.
    - Keep ACE head as primary output.
    - Add anchor-relative branch only after GeoMatch and diagnostics are stable.
    - Detail: `steps/05_geomatch_near_term.md`

15. `[deferred]` Larger research changes.
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

### 11. Diagnostic-Only Observability

Status: `[todo]`

Problem:

- We cannot currently tell whether K=64 global tokens are used well.
- No attention usage, entropy, effective token count, or raw/fused norm metrics
  are logged.

Why this matters:

- Usage loss is risky unless collapse is proven.
- GeoMatch Fusion improvements should be explainable by routing/token behavior,
  not only pose metrics.

Recommended direction:

- Add optional diagnostics that are off by default.
- First implementation should not add loss and should not change outputs when
  disabled.

Metrics:

- attention entropy
- token usage
- effective token count
- average max attention
- raw feature norm
- attention output norm
- fused feature norm
- geometry gate/scale values when GeoMatch is enabled

Verification:

- Diagnostics can be enabled for selected runs.
- Normal runs remain unchanged when diagnostics are disabled.

Detail:

- `steps/05_geomatch_near_term.md`

### 12. Compressor Geometry Contract And Ablations

Status: `[designing]`

Problem:

- Key layer is explicit but not yet ablated.
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
- Only after scene-scale contract, test PE and distance-bias variants.

Do not mix:

- key-layer ablation
- GeoMatch Fusion v1
- scene-scale PE changes

Each should be its own ablation.

Detail:

- `steps/06_compressor_geometry_contract.md`

### 13. GeoMatch Fusion v1

Status: `[todo]`

Problem:

- Current fusion geometry only affects values, not query-memory matching logits.

Recommended direction:

- Keep `value_only` as exact baseline.
- Add `gated_key_value` as controlled ablation.
- Key geometry starts near zero.
- Value geometry should preserve old behavior as closely as possible.

Important compatibility note:

- Current value path is equivalent to `value_geo_scale=1.0`.
- A sigmoid gate initialized at zero gives `0.5`, which is not exactly old
  behavior.

Verification:

- Old checkpoints load.
- Logs/checkpoints identify fusion geometry mode.
- Compare against true-global value-only baseline.

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
- Lower priority than diagnostics and GeoMatch key geometry.

Verification:

- Log residual gate value and raw/fused norm ratio.
- Only test after diagnostics exist.

Detail:

- `steps/05_geomatch_near_term.md`

### 16. Anchor-Assisted Residual Branch

Status: `[designing]`

Problem:

- Absolute coordinate regression may be harder than anchor-relative residual
  prediction.

Recommended direction:

- Keep ACE head as the primary output.
- Add anchor-relative branch only as auxiliary.
- Start with small loss weights such as `0.05` or `0.1`.

Do not do yet:

- Do not replace ACE head.
- Do not make anchor-only the primary path until auxiliary branch proves useful.

Detail:

- `steps/05_geomatch_near_term.md`
- `LONG_TERM_RESEARCH_PLAN.md`
