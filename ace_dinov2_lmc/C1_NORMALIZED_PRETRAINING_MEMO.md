# C1 Normalized Pretraining Memo

Last updated: 2026-04-30

## Purpose

This memo is intentionally independent from the current memory-extraction /
reference-consistent-memory implementation notes.

It records design ideas for future large-scale multi-scene pretraining, where
the goal is to obtain a shared network with stronger cross-scene generalization
without being dominated by raw scene-scale differences.

This file is not a runbook for the current `memory_extraction` mainline.

## Current Observation

From the current single-memory experiments:

- `C1` works end-to-end.
- `acc25` stays close to repaired `C0`.
- `acc5` and median translation are worse than repaired `C0`.
- The degradation is concentrated in high-precision metrics, not in coarse
  geometric feasibility.

Interpretation:

- This pattern is consistent with a normalized-coordinate training target that
  preserves coarse geometry but makes centimeter-level precision harder.
- This does not by itself indicate that the `C1` coordinate transforms are
  wrong.

## Why Keep C1 + Normalization

For future large-scale multi-scene training, directly regressing raw world-frame
coordinates is likely to create optimization instability because:

- different scenes have different coordinate ranges
- different scenes have different scene-center offsets
- one shared head must absorb scene-dependent numeric scale variation

The `C1` route addresses semantic consistency:

- targets are defined in a reference-consistent frame
- recovery back to world remains explicit and geometric

The normalized `points_ref_norm` route addresses numeric consistency:

- target magnitudes become more comparable across scenes
- a shared model can focus more on geometry and less on scene-specific scale

Therefore, `C1 + normalized target` remains the leading candidate foundation
for future shared pretraining.

## Main Hypothesis

The current weakness is not that `C1` is conceptually wrong.

The more likely issue is:

- normalized targets improve shared optimization
- but they suppress high-precision metric calibration
- so fine-grained `acc5` / median-translation performance needs an additional
  metric-aware training signal

## Candidate Direction

Recommended candidate for future large-scale training:

### Option A: C1 normalized pretraining + metric auxiliary loss

Keep:

- `contract_mode = C1`
- primary target space = `points_ref_norm`
- explicit recovery path back to world

Add:

- an auxiliary loss computed after recovering predictions back to
  `points_ref` or `points_world`
- this auxiliary loss should be weaker than the main normalized loss
- it should be used to recover fine metric precision, not replace the main
  normalized objective

Recommended role split:

- main loss: cross-scene stable optimization
- auxiliary metric loss: centimeter-level precision calibration

### Option A0: Weaker normalization before auxiliary loss

Before introducing a second loss, there is a simpler variant worth testing:

- keep `contract_mode = C1`
- keep a normalized reference-frame target
- but reduce the normalization strength so that fine metric errors occupy a
  larger numeric range

Instead of:

- `points_ref_norm = (points_ref - mu_ref) / sigma_ref`

test:

- `points_ref_scaled = alpha * ((points_ref - mu_ref) / sigma_ref)`

with a small scale factor `alpha > 1`, for example:

- `alpha = 2`
- `alpha = 4`

Interpretation:

- larger `alpha` means a larger normalized target range
- the target is still reference-normalized across scenes
- centimeter-level differences become numerically larger in target space
- this may improve `acc5` and median translation without abandoning
  cross-scene normalization

This is conceptually simpler than adding an auxiliary loss immediately, and it
should be treated as the first low-cost ablation inside the normalized-C1
family.

## Why Option A Is Preferred

Compared to switching fully back to raw coordinates:

- it preserves the normalization benefits needed for future multi-scene
  pretraining
- it does not throw away the reference-consistent design
- it introduces the smallest conceptual extension

Compared to switching fully to `points_ref`:

- it is more likely to remain stable across many scenes with different scales
- it keeps the pretraining target numerically comparable across scenes

Compared to adding a separate refinement head immediately:

- it is a smaller first step
- it isolates whether the missing ingredient is simply metric supervision

Compared to immediately introducing a multi-loss setup:

- weaker normalization is easier to reason about
- it changes only the target scale, not the optimization structure
- it can reveal whether the current fine-metric gap is mainly a resolution
  issue in normalized space

## Near-Term Validation Ladder

This memo is for future design, but the design should eventually be validated in
this order:

1. `C1 points_ref_norm` baseline
   - current normalized baseline for comparison
2. `C1 points_ref_norm(alpha=2)` weaker-normalization run
   - first test whether fine precision improves when the normalized target range
     is expanded
3. `C1 points_ref_norm(alpha=4)` weaker-normalization run
   - test whether further expansion helps or begins to erode cross-scene
     normalization benefits
4. `C1 points_ref` control run
   - isolate the effect of reference-frame training without normalization
5. `C1 points_ref_norm + metric auxiliary loss`
   - test whether fine accuracy can be recovered while preserving normalized
     training as the main objective

Expected interpretation:

- if weaker normalization improves `acc5` and median translation while keeping
  `acc25` stable, then the current issue is likely over-compression of the
  target range rather than a conceptual failure of `C1`
- if `C1 points_ref` improves `acc5` materially over standard normalized `C1`,
  then the main issue is fine precision under normalization
- if auxiliary metric loss closes most of that gap, then normalized
  pretraining remains a strong long-term foundation

## What This Memo Does Not Decide Yet

This memo does not yet fix:

- the exact auxiliary loss form
- whether the auxiliary loss should be defined in `points_ref` or `points_world`
- the final loss weight schedule
- whether later stages should add a refinement head or residual branch

Those should be decided only after the first minimal auxiliary-loss ablation.

## Working Conclusion

Current working conclusion:

- do not abandon `C1`
- do not abandon normalized targets
- treat weaker normalization as the first low-cost ablation inside the
  normalized-C1 family
- treat the current `acc5` / median-translation drop as an expected fine-metric
  tradeoff
- pursue `C1 + normalized target + metric auxiliary loss` as the primary
  candidate for future large-scale multi-scene pretraining
