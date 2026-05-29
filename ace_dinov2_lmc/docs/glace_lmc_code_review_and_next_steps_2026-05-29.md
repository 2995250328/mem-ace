# GLACE-LMC Memory Fusion Code Review and Next Steps

Date: 2026-05-29
Scope: GLACE-LMC / memory fusion training, configuration, checkpointing, and evaluation paths in `ace_dinov2_lmc`.
Status: Canonical English version. This is a read-only review document; no code changes were made for it.

## 1. Background

The current GLACE-LMC research line tests whether compressed LMC memory tokens can improve GLACE scene coordinate regression by modifying only the local feature stream while bypassing the global feature stream.

Original intended structure:

```text
vanilla GLACE scene head initialization
+ frozen GLACE encoder / global feature source
+ LMC fusion applied only to local features
+ global feature bypass
+ local residual / fixed alpha to control memory delta
+ optional head co-adaptation
```

The core research question is:

```text
Can compressed memory tokens improve GLACE scene coordinate regression through local feature enhancement?
```

Existing experiments indicate that the current direct path is not promising:

```text
LMC compressed memory -> local feature delta -> GLACE head
```

Direct replacement collapses, moderate alpha values degrade or collapse, and very small alpha values produce only weak, unstable fluctuations.

## 2. Implemented Capabilities

### 2.1 Fusion target

The code supports:

```bash
--lmc_fusion_target decoder|local
```

Semantics:

```text
decoder:
  Legacy path. Fusion query is the full decoder feature:
  concat(global, local).

local:
  GLACE-LMC-specific path. Fusion is applied only to the local feature.
  The global feature bypasses memory fusion and is concatenated back before the GLACE head.
```

Relevant files:

```text
options_dinov2_lmc.py
trainer_dinov2_lmc.py
test_ace_dinov2_lmc.py
```

### 2.2 Local residual mode

The code supports:

```bash
--local_residual_mode none|fixed_alpha
--local_residual_alpha <float>
```

Semantics:

```text
none:
  local_out = local_fused

fixed_alpha:
  local_out = local_raw + alpha * (local_fused - local_raw)
```

`local_residual_alpha` is expected to be in `[0, 1]`.

### 2.3 GLACE freeze policy

Relevant arguments:

```bash
--glace_freeze_encoder
--glace_freeze_head
--glace_freeze_base_network
```

Current effective semantics:

```text
glace_freeze_encoder:
  Defaults to True. The GLACE encoder is frozen by default.

glace_freeze_head:
  Defaults to None. When None, it inherits glace_freeze_base_network.

glace_freeze_base_network:
  Defaults to True.
```

Therefore, by default:

```text
GLACE encoder is frozen.
GLACE head is also frozen.
```

### 2.4 Checkpoint and evaluation synchronization

The checkpoint `lmc_config` records the key fusion and freeze fields, including:

```text
lmc_fusion_target
requested_lmc_fusion_target
effective_lmc_fusion_target
local_residual_mode
local_residual_alpha
glace_freeze_encoder
glace_freeze_head
```

The test path reads these fields from the checkpoint and reproduces the corresponding decoder or local-only fusion path.

## 3. Experiment Conclusions To Date

Common evaluation setup:

```text
scene:
  /data/xwh/Wayspots/wayspots_bears

memory:
  /home/xwh/project/ace_depth/ace_dinov2_lmc/04_evaluation/wayspots_lmc/memory/wayspots_bears/32v_v0.05_bilinear_bse_ut0.02_noaug_asb_adaptive_pool1.0_rgate4_prepair8_sor_gm_l2/20260525_223355/memory_bse.pt

GLACE init head:
  /home/xwh/project/ace_depth/ace_dinov2_lmc/04_evaluation/wayspots_baselines/20260525_190055/wayspots_bears/glace/model.pt

eval:
  seeds = 1305,2026,4242
  hypotheses = 256
  aggregation = median
```

### 3.1 Decoder baseline

Representative result:

```text
Median: 1.00 deg / 3.07 cm
25cm/5deg: 95.00%
10cm/5deg: 89.14%
5cm/5deg: 77.59%
2cm/2deg: 14.48%
1cm/1deg: 0.86%
```

The decoder path preserves the GLACE baseline level.

### 3.2 Local-only direct replacement

Configuration:

```text
lmc_fusion_target = local
local_residual_mode = none
```

This means:

```text
local_out = local_fused
```

Representative result:

```text
Median: 29.45 deg / 116.07 cm
25cm/5deg: 0.52%
10cm/5deg: 0.17%
5cm/5deg: 0.00%
```

Conclusion:

```text
Directly replacing the local feature collapses.
This path should not be continued.
```

### 3.3 Fixed-alpha local residual

Configuration:

```text
lmc_fusion_target = local
local_residual_mode = fixed_alpha
```

Observed behavior:

```text
alpha = 0.2:
  Collapses. Main thresholds are essentially zero.

alpha = 0.1:
  Clearly degrades.

alpha = 0.03:
  Still degrades.

alpha = 0.015 / 0.01 / 0.005:
  Close to baseline, but does not stably outperform the decoder baseline.
```

Conclusion:

```text
The usable alpha range is extremely small, roughly 0.005 to 0.01.
Small alpha values only produce weak, local, unstable fluctuations.
Continuing to sweep alpha is not useful.
```

### 3.4 Head co-adaptation

Observed behavior:

```text
head trainable + alpha=0.01:
  No improvement. Overall worse.

head trainable + alpha=0.005:
  No improvement. Overall worse.
```

Conclusion:

```text
Head co-adaptation did not rescue local memory fusion.
```

Caveat:

```text
This conclusion should be annotated with the exact effective freeze policy and optimizer parameter groups.
In particular, confirm that the GLACE head was actually included in the optimizer for those runs.
```

## 4. Code Review Findings

### 4.1 High priority: `glace_freeze_head` default can contaminate experiment interpretation

Relevant files:

```text
options_dinov2_lmc.py
trainer_dinov2_lmc.py
```

Current behavior:

```text
--glace_freeze_head defaults to None.
None inherits --glace_freeze_base_network.
--glace_freeze_base_network defaults to True.
Therefore, the head is frozen by default.
```

Risk:

```text
Old or implicit configs that do not pass --glace_freeze_head False may leave the GLACE head frozen.
This affects the interpretation of "head co-adaptation did not help".
```

Recommended short-term fix:

Do not immediately change the default behavior. First add explicit logs and consistency checks:

```text
Print effective freeze policy.
Print trainable parameter counts.
Print optimizer parameter groups.
Raise an error if the user expects the head to be trainable but no head parameters are in the optimizer.
```

Recommended log content:

```text
[GLACE-LMC config]
  freeze_base_network = ...
  freeze_encoder = ...
  freeze_head = ...
  lmc_fusion_target = ...
  effective_lmc_fusion_target = ...
  local_residual_mode = ...
  local_residual_alpha = ...
  ace_g_fusion_in_s2 = ...

[Trainable params]
  encoder = ...
  head = ...
  compressor = ...
  fusion = ...
  residual_adapter = ...

[Optimizer groups]
  group_name, lr, num_tensors, num_params
```

### 4.2 High priority: `glace_freeze_encoder=False` is currently likely a no-op

Current behavior:

```text
--glace_freeze_encoder False sets encoder parameters to requires_grad=True.
```

But:

```text
S1/S2 optimizers do not include encoder parameters.
S1 online feature extraction uses torch.no_grad().
```

Therefore:

```text
The encoder will not be updated even if requires_grad=True.
```

Risk:

```text
Any encoder-unfreeze experiment may be invalid.
```

Recommended short-term fix:

Reject unsupported encoder unfreeze for the current GLACE-LMC flow:

```text
if model_backend == "glace_lmc" and glace_freeze_encoder is False:
    raise ValueError(
        "glace_freeze_encoder=False is not supported yet: "
        "encoder params are not included in S1/S2 optimizers."
    )
```

Add true encoder training later only with an explicit design:

```text
encoder optimizer group
encoder LR ratio
removal or restructuring of no_grad feature extraction
VRAM and stability policy
```

### 4.3 Medium priority: local-only delta logging may use the wrong dimensional convention

In the local-only path:

```text
base_features_bC is already local-dimensional.
```

Some delta logging may still slice using:

```python
base_features_bC[:, global_dim:]
```

This can make local delta statistics invalid or hard to interpret.

Recommended fix:

Split the logging path explicitly:

```python
if effective_lmc_fusion_target == "decoder":
    base_local = base_features_bC[:, global_dim:]
    head_local = head_features_bC[:, global_dim:]
    delta_stats_scope = "decoder_local_slice"

elif effective_lmc_fusion_target == "local":
    base_local = base_features_bC
    head_local = head_features_bC
    delta_stats_scope = "local_query"
```

Write `delta_stats_scope` to the log.

### 4.4 Medium priority: iteration eval and post-train eval use different settings

Current behavior:

```text
iteration eval:
  hypotheses is fixed to 64.
  seed comes from eval_dsacstar_seed.

post-train eval:
  seeds come from post_train_eval_seeds.
  hypotheses comes from post_train_hypotheses.
```

Risk:

```text
Iteration best checkpoint metrics and post-train multi-seed metrics are not directly comparable.
```

Recommended fix:

Add explicit parameters:

```bash
--iteration_eval_hypotheses
--iteration_eval_seed
```

Or at minimum write these fields into every summary:

```text
eval_type
hypotheses
seeds
deterministic
dsacstar_seed
```

### 4.5 Medium priority: post-train summary lacks complete fusion/freeze semantic fields

Standalone test summary already writes some of these fields, but post-train summary is less complete.

Recommended post-train summary fields:

```text
model_backend
lmc_flow
lmc_fusion_target
requested_lmc_fusion_target
effective_lmc_fusion_target
fusion_query_dim
local_residual_mode
local_residual_alpha
glace_freeze_base_network
glace_freeze_encoder
glace_freeze_head
ace_g_fusion_in_s2
eval_deterministic
post_train_eval_seeds
post_train_hypotheses
```

Goal:

```text
An eval summary should be sufficient to understand the experiment semantics without re-opening the full config.
```

## 5. Directions Not Recommended

Do not continue:

```text
1. Do not keep sweeping alpha=0.006 / 0.0075 / 0.012.
2. Do not continue local_residual_mode=none.
3. Do not simply increase iterations or training steps.
4. Do not evaluate only final pose metrics without feature/pixel diagnostics.
5. Do not strengthen guard loss just to pull fused outputs back to vanilla behavior.
6. Do not immediately build a complex gate network.
7. Do not immediately train the encoder.
```

Reason:

```text
The main unresolved question is not whether training is long enough.
The main unresolved question is whether the memory delta lies on a feature manifold that the GLACE head can interpret.
```

## 6. Recommended Modification Plan

### Step 1: Add freeze / optimizer diagnostics and guards

Goal:

```text
Do not change the training algorithm. Make the experiment semantics visible.
```

Changes:

```text
1. Print effective GLACE-LMC config.
2. Print trainable parameter counts.
3. Print optimizer parameter groups.
4. Reject currently unsupported encoder unfreeze.
5. Check that head trainability matches optimizer membership.
```

Priority files:

```text
trainer_dinov2_lmc.py
options_dinov2_lmc.py
```

### Step 2: Fix local-only delta logging

Goal:

```text
Make feature-level diagnostics trustworthy.
```

Changes:

```text
1. Separate decoder-fusion and local-fusion logging conventions.
2. Do not slice local-only features by global_dim.
3. Log delta_stats_scope.
```

Metrics to record:

```text
local_raw_norm
local_fused_norm
local_delta_norm
delta_local_ratio
delta_cos
head_input_delta_ratio
```

Priority file:

```text
trainer_dinov2_lmc.py
```

### Step 3: Complete post-train summary fields

Goal:

```text
Make post-train eval, iteration eval, and standalone test summaries comparable.
```

Changes:

```text
Add complete semantic fields to the post-train summary in train_ace_dinov2_lmc.py.
```

Priority file:

```text
train_ace_dinov2_lmc.py
```

### Step 4: Add base-vs-fused pixel-level diagnostics

Goal:

Answer the central question:

```text
Does memory delta help only hard pixels, or does it damage most GLACE local features?
```

Compute on the same batch:

```text
base:
  GLACE head(concat(global, local_raw))

fused:
  GLACE head(concat(global, local_raw + alpha * (local_fused - local_raw)))
```

Record:

```text
err_base_px
err_fused_px
err_delta = err_fused_px - err_base_px
```

Bucket by base reprojection error:

```text
easy:
  base_err < 5 px

medium:
  5 px <= base_err < 20 px

hard:
  base_err >= 20 px
```

Output:

```text
mean err_delta
median err_delta
improved ratio
worsened ratio
hard-pixel improved ratio
easy-pixel damaged ratio
```

Interpretation:

```text
If hard pixels improve slightly but easy pixels are widely damaged:
  memory should be used as a gate / uncertainty / hard-pixel selector.

If most pixels get worse:
  abandon direct local feature modification.

If hard pixels improve clearly and easy pixels are not damaged:
  gated memory fusion may be worth pursuing.
```

## 7. Minimal Implementation Order

Recommended first patch:

```text
1. Freeze policy log.
2. Trainable parameter log.
3. Optimizer group log.
4. Encoder-unfreeze no-op guard.
5. Local-only delta logging fix.
6. Post-train summary field completion.
```

Recommended second patch:

```text
1. Pixel-level base-vs-fused diagnostics.
2. Easy / medium / hard bucket statistics.
3. Attention entropy / token usage statistics.
```

Recommended next experiment:

```text
1. Do not train a new model first.
2. Use an existing checkpoint or a small fixed batch.
3. Run diagnostics to identify the memory delta failure mode.
```

## 8. Possible Future Structures

Consider these only if pixel-level diagnostics show that memory helps hard pixels without broadly damaging easy pixels.

### 8.1 Memory predicts gate only

Memory does not directly generate a feature delta. It predicts gate / confidence / uncertainty.

```text
local_out = local_raw + gate * small_delta
```

The gate should be near zero for easy pixels and only open for hard pixels.

### 8.2 Memory only acts on hard pixels

Use base GLACE reprojection error or confidence to identify hard pixels:

```text
if base_err is high:
    allow memory fusion
else:
    keep vanilla local feature
```

### 8.3 Memory as auxiliary loss

Memory does not modify the head input. Instead, it provides auxiliary supervision or contrastive alignment for hard-pixel local features.

### 8.4 Retrieval / context conditioning

Memory provides scene-level or region-level context, but does not directly generate a per-pixel local feature delta.

## 9. Current Recommendation

The immediate priority is not another training sweep. The immediate priority is to make the experiment semantics and failure mode observable:

```text
1. Confirm which modules are actually trainable.
2. Confirm optimizer parameter groups match the intended freeze policy.
3. Fix local-only delta logging.
4. Add base-vs-fused pixel-level diagnostics.
```

Do not expand the experiment matrix until these diagnostics are in place.

Recommended next action:

```text
Implement the diagnostic patch first, then use an existing checkpoint or a small fixed batch to determine whether memory delta helps hard pixels or broadly damages the GLACE feature manifold.
```
