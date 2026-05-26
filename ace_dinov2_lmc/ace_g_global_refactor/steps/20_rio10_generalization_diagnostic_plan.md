# Step 20: RIO10 Generalization Diagnostic Plan

## 1. Purpose

This note defines the diagnostic plan for understanding why the current RIO10 ACE-DINOv2-LMC / ACE-G run generalizes poorly.

The goal is not to add a new architecture. The goal is to locate the failure mode before spending more effort on PointRoPE, CIFA, multi-level fusion, view tokens, or multi-memory routing.

The current priority is:

```text
first diagnose RIO10 generalization
then decide whether architecture changes are meaningful
```

A method that is strong on a single indoor scene but fails on RIO10 should not be treated as validated until the RIO10 failure mode is understood.

## 2. Current Evidence

The current run under:

```text
/home/xwh/project/ace_depth/ace_dinov2_lmc/04_evaluation/train_compare/rio10_conservative_indoor6_migration
```

should be treated as an incomplete diagnostic signal, not a final result.

Observed state:

```text
configured iterations = 28
observed artifacts/logs stop around iter11
trainer-selected best iteration = iter8
best_metric = pct5
best pct5 = 8.31
best median error = 15.30 deg / 45.39 cm
latest observed iter11 pct5 = 7.93
latest observed iter11 median error = 15.50 deg / 45.67 cm
```

The data split does not currently look like the primary problem.

Current evidence indicates:

```text
ace-g official train -> local train
ace-g official val   -> local test
ace-g official test  -> local hidden_test

current ACE-format train scene = /data/xwh/RIO10_ace/scene01_seq01_01
current train frames = 4380
current eval frames = 2069
```

This is consistent with using RIO10 scene01 training sequence for training and the public validation sequence for evaluation.

The current memory/sparse-depth context is:

```text
memory = 40-view ASB single-forward BSE memory
memory points = 10705
memory feature dim = 3840
sparse depth matched = 4356 / 4380
missing sparse depth = 24
c1_aux_ref_loss_weight = 0.0
buffer_sample_valid_coords = True
buffer_valid_coord_sample_ratio = 1.0
```

Recent Indoor6 `scene2a/c0_p4` Danchor controls remain healthy. The best of
the three reported 2026-05-23 runs is `Danchor keybias` with:

```text
median = 0.28 deg / 2.64 cm
25cm/5deg = 99.61
10cm/5deg = 95.33
5cm/5deg = 80.93
2cm/2deg = 35.80
1cm/1deg = 10.12
```

Recorded in:

```text
ace_g_global_refactor/COMPARE_FGPI_Danchor_scene2a_20260523.md
```

This supports the interpretation that the Indoor6 ACE-G path is still healthy,
but it does not explain or solve RIO10. RIO10 should remain focused on the
data/eval contract, resize/memory geometry, domain shift, and sparse-guided
sampling diagnostics below.

The split may still need an integrity audit, but the next investigation should not assume the poor result is simply caused by using the wrong test set.

## 3. Main Hypotheses

### H1: Data / Pose / Calibration Contract Problem

Even if the split is correct, ACE-format conversion could still be wrong.

Risks to audit:

- RGB / pose / calibration files are sorted independently and only count-checked;
- image stems may not be aligned across `rgb`, `poses`, and `calibration`;
- symlinks may point to unexpected source files;
- pose convention may differ from what DSAC evaluation expects;
- calibration may be correct before resizing but wrong after resizing;
- translation scale may not be in meters;
- train/test sequence metadata may be correct but one converted directory may be stale.

If this hypothesis is true, train-set evaluation may already be poor.

### H2: RIO10 Is Hard for the Vanilla DINO ACE Baseline

Before blaming LMC or memory, establish a same-split non-LMC baseline.

If vanilla DINO ACE is also poor on the same RIO10 ACE-format scene, the issue may be:

- domain shift;
- RIO10 pose/calibration/data difficulty;
- DSAC threshold / hypotheses settings;
- insufficient vanilla training recipe;
- a general ACE-DINOv2 limitation on this scene.

### H3: LMC / Memory Fusion Hurts Generalization

If vanilla DINO ACE is reasonable but ACE-G is poor, the failure is likely introduced by the memory path.

Possible causes:

- memory coverage does not represent the eval trajectory;
- 40 ASB views are not enough for RIO10 scene01;
- BSE memory is too sparse or biased;
- memory geometry quality is poor;
- fusion in S2 overfits memory artifacts;
- S1 alignment does not learn useful memory-conditioned features;
- the 3840-dim selected key/value memory contract is too brittle on RIO10.

### H4: Sparse Guided Sampling Is Too Aggressive

The current run uses:

```text
buffer_sample_valid_coords = True
buffer_valid_coord_sample_ratio = 1.0
```

This makes all sampled buffer positions come from sparse-depth-valid coordinates when available.

Given the observed sparse-depth coverage limits, this can reduce sample diversity and overfit training to a narrow subset of patches. Sparse depth may still be useful, but ratio `1.0` is a high-risk setting.

### H5: Training Parameters Are Not Conservative Enough

The current run is not only a dataset migration. It also uses several settings that may be too aggressive for RIO10:

```text
batch_size = 10240
learning_rate_min = learning_rate_max = 1e-4
s2_learning_rate_max = 1e-3
head_lr_multiplier_s2 = 1.5
lmc_profile = legacy
buffer_sampling_replacement = True
ace_g_fusion_in_s2 = True
```

These may interact with RIO10 data size, sparse guided sampling, and memory quality.

## 4. Diagnostic Ladder

Run diagnostics in this order. Do not start with new architecture.

### Step 0: Freeze the Current Result as Partial

Record the current run as incomplete:

```text
best known = iter8
source = best_checkpoint_meta.json + eval_summary_scene01_seq01_01_iter_08.txt
status = incomplete / interrupted around iter11
```

Do not rank it as a completed 28-iteration result.

### Step 1: Data / Pose / Calibration Integrity Audit

Audit the ACE-format scene:

```text
/data/xwh/RIO10_ace/scene01_seq01_01
```

Checks:

1. Verify counts for train/test `rgb`, `poses`, and `calibration`.
2. Verify filename stem alignment after sorting.
3. Verify RGB symlink targets point to expected RIO10/WAI source files.
4. Compare several ACE-format poses against WAI metadata transforms.
5. Compare several ACE-format calibrations against WAI metadata intrinsics.
6. Verify resized intrinsics used by `dataset_dinov2.py` match image resize behavior.
7. Check train/test camera center distributions and translation units.
8. Confirm test sequence corresponds to public validation, not hidden test or train leakage.

Expected interpretation:

```text
mismatch found -> fix data contract before training more models
no mismatch    -> continue to train/eval sanity checks
```

### Step 2: Train-Set Evaluation Sanity Check

Evaluate the current best checkpoint on the training split or a training subset.

Purpose:

```text
separate memorization/fit failure from generalization failure
```

Interpretation:

```text
poor train eval + poor test eval -> data/eval/optimization failure
strong train eval + poor test eval -> generalization/memory/sampling failure
```

This is the highest-value quick diagnostic. If train-set pose accuracy is also bad, do not run more architecture ablations yet.

### Step 3: Vanilla DINO ACE Baseline on the Same Split

Train or locate a same-split vanilla baseline:

```text
use_lmc = False
data_backend = ace
scene = /data/xwh/RIO10_ace/scene01_seq01_01
same image_resolution
same evaluation protocol
```

This baseline is the anchor for interpreting ACE-G.

Interpretation:

```text
vanilla poor -> RIO10 baseline/data/eval/training problem
vanilla good -> LMC/memory/fusion/sampling problem
```

Do not compare ACE-G only against Indoor6. RIO10 needs a local non-LMC reference.

### Step 4: DSAC / Evaluation Sensitivity Sweep

Run a controlled test-time sweep on the same checkpoint.

Parameters to vary:

```text
hypotheses: 64, 256, 512
threshold: 5, 10, 20
maybe inlier alpha / max pixel error if exposed
```

Interpretation:

```text
large improvement with relaxed DSAC -> coordinates may be noisy but useful
no improvement -> coordinate predictions or pose/calibration contract likely wrong
```

This should not be used to change the official metric yet. It is diagnostic only.

### Step 5: Per-Frame and Trajectory Diagnostics

Plot or tabulate per-frame errors against frame index and camera position.

Look for:

- all frames bad;
- only specific trajectory segments bad;
- errors increasing with distance from memory views;
- errors correlated with sparse-depth missing frames;
- errors correlated with camera rotation or view direction;
- repeated pose flips or scale-like translation bias.

Interpretation:

```text
localized failure -> memory coverage / sequence region issue
global failure    -> data/eval/model contract issue
```

### Step 6: Sparse Guided Sampling Ablation

Use the same model contract as Step19 and only vary sampling.

Minimal matrix:

```text
A: buffer_sample_valid_coords = False
B: buffer_sample_valid_coords = True, buffer_valid_coord_sample_ratio = 0.25
C: buffer_sample_valid_coords = True, buffer_valid_coord_sample_ratio = 0.50
D: buffer_sample_valid_coords = True, buffer_valid_coord_sample_ratio = 1.00
```

Keep:

```text
c1_aux_ref_loss_weight = 0.0
same memory
same seed if possible
same evaluation protocol
```

Interpretation:

```text
A >= D      -> sparse guided sampling hurts or sparse masks are biased
B/C > A/D   -> sparse depth helps only as partial prior
D best      -> full sparse-guided sampling is justified
```

Given current evidence, ratio `1.0` should be treated as suspicious until proven otherwise.

### Step 7: LMC / Fusion Ablation

If vanilla is good but ACE-G is poor, isolate the memory path.

Minimal matrix:

```text
L0: vanilla DINO ACE, no LMC
L1: ACE-G, no sparse guided sampling
L2: ACE-G, sparse guided sampling ratio 0.25 or 0.50
L3: ACE-G, ace_g_fusion_in_s2 = False
L4: ACE-G, ace_g_fusion_in_s2 = True
```

Interpretation:

```text
L1 poor vs L0 -> memory compression/fusion path hurts
L3 better than L4 -> S2 fusion overfits or destabilizes
L2 better than L1 -> sparse sampling helps when not too aggressive
```

### Step 8: Training Parameter Ablation

Only after Steps 1-7, test training parameter suspects.

Prioritized changes:

1. Restore Indoor6-style batch size if current run used `10240`:

```text
batch_size = 5120
```

2. Test lower or less boosted S2 learning:

```text
s2_learning_rate_max = 5e-4 or 1e-4
head_lr_multiplier_s2 = 1.0
```

3. Compare legacy profile against a less aggressive profile only if it keeps the main contract interpretable.

4. Reduce replacement-driven buffer bias if possible:

```text
buffer_sampling_replacement = False
```

Interpretation:

```text
lower LR / batch improves -> optimization overfit or instability
replacement off improves  -> repeated sparse/buffer samples caused bias
no change                 -> look back to data/memory coverage
```

### Step 9: Memory Coverage and Density Ablation

If LMC is the likely culprit, inspect memory construction.

Diagnostics:

1. Compare memory view camera centers against train/test trajectory.
2. Compute nearest-memory-view distance for each train/test frame.
3. Compare errors against nearest-memory-view distance.
4. Compare ASB memory against FPS-flat memory.
5. Compare 40 views against larger budgets such as 80 or 120 if feasible.
6. Compare BSE memory against simpler pooled memory if available.
7. Verify `patch_depth_sampling=nearest_valid` for sparse-depth memory extraction.

Interpretation:

```text
more views improve -> memory coverage bottleneck
FPS improves       -> ASB selection too local or biased
simple pooled improves over BSE -> BSE sparsification/density issue
nearest_valid improves -> sparse-depth memory geometry issue
```

## 5. Minimal Experiment Order

The recommended first diagnostic sequence is:

```text
D0: data/pose/calibration audit
D1: current best checkpoint train-set eval
D2: vanilla DINO ACE on same RIO10 ACE-format split
D3: DSAC sensitivity sweep on current best checkpoint
D4: no-guided vs guided-ratio 0.25/0.50/1.00
D5: fusion-in-S2 off vs on
D6: memory coverage 40 ASB vs larger/FPS memory
```

Do not run D4-D6 before D0-D2 unless GPU time is otherwise idle, because D0-D2 determine whether the failure is even in LMC.

## 6. Decision Tree

Use the following decision rules.

```text
train-set eval poor
  -> prioritize data/pose/calibration/eval contract and optimization sanity

train-set eval good, test eval poor
  -> true generalization failure; continue with vanilla baseline and memory coverage

vanilla DINO ACE poor
  -> RIO10 domain/data/eval/training baseline issue; do not blame LMC first

vanilla DINO ACE good, ACE-G poor
  -> LMC/memory/fusion/sparse sampling issue

no-guided >= guided
  -> sparse guided sampling is hurting or too biased

guided 0.25/0.50 > guided 1.0
  -> sparse sampling is useful only as a partial prior; ratio 1.0 collapses diversity

fusion-off > fusion-on
  -> S2 fusion continuation is harmful on RIO10

larger/FPS memory > 40 ASB memory
  -> memory coverage or selection is the bottleneck

DSAC relaxed threshold helps a lot
  -> coordinates are noisy but partially useful; training quality issue

DSAC sweep does not help
  -> coordinate predictions or data/eval contract likely wrong
```

## 7. What Not to Do Yet

Until this diagnostic ladder identifies the failure mode, do not use RIO10 results to justify adding:

- PointRoPE;
- Cascading Internal Fusion Assembly;
- query-side DPT/FPN adapters;
- multi-level compression changes;
- view-token conditioning;
- multi-memory routing;
- sparse-depth auxiliary supervision;
- reference-coordinate auxiliary loss.

These may become useful later, but they would currently add confounders.

## 8. Expected Outcome

The intended output of this diagnostic phase is one of the following conclusions:

```text
A. Data/eval contract was wrong and must be fixed.
B. Vanilla ACE-DINOv2 is already weak on RIO10, so RIO10 needs a new baseline recipe.
C. Vanilla is acceptable but ACE-G memory/fusion hurts generalization.
D. Sparse guided sampling helps only at a partial ratio, not at ratio 1.0.
E. Memory coverage/density is the limiting factor.
F. DSAC/evaluation settings expose noisy but recoverable coordinates.
```

Only after one of these conclusions is supported should the project resume architecture changes.
