# Step 05 - Progressive Geometry Injection / GeoKey v0

Status: implemented; A0/A1/A2 experiments pending

## Sources

Reviewed local design notes:

- `silu/GeoMatch-ACE-G_技术设计文档.docx`
- `silu/视觉重定位系统设计文档.docx`
- `silu/视觉重定位系统设计文档.md`

Only the near-term, ACE-G-compatible parts are absorbed here. Larger zero-shot
MoE / U-LCA directions are tracked separately in `../LONG_TERM_RESEARCH_PLAN.md`.

## Current Code Fact

`LMCFusionBlock` currently injects memory geometry into the value path only:

```python
k = k_proj(memory_z)
pe = PE(memory_p - scene_center)
v = v_proj(memory_z + pe_proj(pe))
```

This means latent 3D coordinates affect the rendered memory value, but they do
not directly affect query-memory matching logits.

## Critical Assumption

The next question is not whether to build full GeoMatch immediately. The
testable near-term question is narrower:

```text
Can memory-side geometry improve fusion token selection before the model has a
reliable query-side 3D coordinate?
```

Do not introduce query-memory distance bias as the first step. Query-side 3D is
predicted by the same system that fusion is meant to improve, so direct
`B_geo(q_geo, m_geo)` creates a circular dependency unless the query coordinate
is produced by a detached coarse path. Even then, a detached coarse coordinate
is not automatically safe: if the coarse point is outside the correct basin of
attraction, a geometry bias can reward the wrong memory region and make the
refined head less able to recover. Coordinate-based GeoMatch therefore belongs
behind token-routing diagnostics and must be a confidence-gated weak prior with
fallback, not a hard coarse-to-fine constraint.

## Near-Term Sequence

1. Finish and compare true-global baselines.
   - `global + fixed_zero`
   - `global + per_iter`
   - Do not merge architectural changes until these runs clarify the baseline.

2. Use the already implemented runtime diagnostics.
   - Attention entropy, token usage, effective token count, max attention, and
     raw/fused feature norms should be available when explicitly enabled.
   - These diagnostics explain routing changes; they are not a new loss.

3. Run Progressive Geometry Injection as A0/A1/A2.
   - A0: current value-only raw centered PE.
   - A1: value-only with explicit scene-scale-normalized memory geometry.
   - A2: A1 plus memory-side GeoKey with a zero-init scalar gate.
   - Implementation is available through `--lmc_fusion_geometry_mode`.

4. Consider usage regularization only if diagnostics justify it.
   - Start S1-only.
   - Try small weights only, such as `0.001`, `0.003`, `0.01`.
   - Do not add load balancing loss before observing collapse.

5. Consider a detached token-routing prior before coordinate GeoMatch.
   - Predict a memory-token distribution from query features.
   - Detach or otherwise protect the prior so it cannot dominate the main
     image-feature matching path.
   - Add it to fusion logits as a weak residual prior, e.g.
     `logits = image_logits + g * log(r + eps)`.

6. Consider coordinate-based coarse GeoMatch only after token routing.
   - Coarse head predicts query coordinate, but the coordinate is detached and
     normalized before any geometry use.
   - Add only a confidence-gated nearby reward to refined fusion logits.
   - Keep the feature-matching logits as the dominant fallback path.

## A0/A1/A2 Design Notes

### A0 - Exact Current Baseline

```python
norm_p = memory_p - scene_center
pe_mem = pe_proj(coord_encoder(norm_p))
k_input = memory_z
v_input = memory_z + pe_mem
```

Requirements:

- Must reconstruct old value-only fusion exactly.
- Must load old checkpoints through compatibility defaults.
- This is the reference for the current `forceglobal_s1_periter` baseline.

### A1 - Normalized Value-Only Geometry

```python
p_norm = (memory_p - scene_center) / scene_scale
pe_mem = pe_proj(coord_encoder(p_norm))
k_input = memory_z
v_input = memory_z + pe_mem
```

Requirements:

- `scene_scale` should come from stable memory metadata or a precomputed
  statistic over memory points, such as P95 radius around `scene_center`.
- Do not compute scale from per-forward latent coordinates if those coordinates
  may later become trainable or deformable.
- Record `scene_scale_source` and `scene_scale_value` in logs/checkpoint
  metadata.

Interpretation:

- A1 > A0 suggests PE scale/frequency matters.
- A1 > A0 does not prove geometry matching, because key logits are unchanged.
- A1 <= A0 may mean the scale contract or PE frequency changed poorly; it is
  not enough to reject geometry-aware fusion.

### A2 - Normalized GeoKey

```python
p_norm = (memory_p - scene_center) / scene_scale
pe_mem = pe_proj(coord_encoder(p_norm))
k_input = memory_z + key_geo_scale * pe_mem
v_input = memory_z + pe_mem
```

Requirements:

- `key_geo_scale` should be a direct scalar parameter initialized to `0.0`.
- Avoid a sigmoid gate initialized at zero, because it outputs `0.5` and is not
  baseline-equivalent.
- Keep the old value path equivalent to `value_geo_scale = 1.0`.
- Fix `lmc_key_slice_idx=2` for the first A0/A1/A2 comparison so key-layer
  ablations do not contaminate fusion geometry results.

Interpretation:

- A2 > A1 supports memory-side geometry entering fusion keys.
- A2 ~= A1 does not disprove full GeoMatch; it only says memory-only GeoKey was
  insufficient.
- A2 worse than A1 should trigger diagnostics before escalation: check learned
  gate value, attention entropy, effective token count, norm ratios, and
  nonfinite behavior.

## Candidate Config Fields

Keep naming explicit and backward compatible:

```text
--lmc_fusion_geometry_mode value_only_raw
--lmc_fusion_geometry_mode value_only_norm
--lmc_fusion_geometry_mode geokey_norm
--lmc_fusion_key_geo_init 0.0
--lmc_fusion_scene_scale_source memory_points_p95
--lmc_fusion_scene_scale_source fixed
--lmc_fusion_scene_scale_value <positive_float>
```

Compatibility default is `value_only_raw`, so old training and test checkpoints
remain on the old value-only fusion path unless the new flag is set.

## Non-Goals For This Step

- No memory extraction changes.
- No compressor redesign.
- No ACE head replacement.
- No full coordinate target normalization.
- No reference-frame point-cloud conversion.
- No multi-scene joint training.
- No sub-memory routing.
- No ray / Plucker feature injection.
- No PnP / RANSAC solver change.
- No P4Pf or uncalibrated-query assumptions.
- No learned relative geometry bias before token-routing and any hand-designed
  weak geometry prior are understood.
- No usage/load-balancing loss before diagnostics.

Related but deferred ideas:

- Multi-scene universal LMC, reference-frame memory coordinates, full
  coordinate normalization, sub-memory routing, MapAnything/VGGT acceleration,
  ray / Plucker geometry, and MoE heads are tracked in
  `../LONG_TERM_RESEARCH_PLAN.md`.
- They are important for the eventual system, but they change data contracts,
  output spaces, or memory construction. They should not be mixed into the
  first attention-diagnostics or GeoKey v0 ablations.

## Success Criteria

- A0 is exactly compatible with current value-only fusion.
- A1 and A2 can be run as one-variable ablations against A0.
- Logs/checkpoints identify fusion mode and scene-scale contract.
- Old checkpoints load through a deliberate compatibility path.
- Runtime diagnostics remain off by default and do not change outputs.

## Detailed Execution Plan

### Phase A - Lock True-Global Baselines

Before changing architecture, finish the controlled baseline comparison:

- `global + fixed_zero`
- `global + per_iter`

Required log contracts:

```text
[LMC] Mode contract: requested=global effective=global auto_by_visibility=False
[S1] sampled loss step mode: requested=fixed_zero resolved=fixed_zero
```

or:

```text
[LMC] Mode contract: requested=global effective=global auto_by_visibility=False
[S1] sampled loss step mode: requested=per_iter resolved=per_iter
```

Do not treat old `--lmc_mode global` runs as true global unless logs prove
`effective=global`. The old indoor6 4090 baseline is mostly
requested-global/effective-local because visibility fallback was enabled.

Baseline comparison should report:

- best iteration
- best score
- post-train multi-seed median rotation / translation
- 25cm, 10cm, 5cm, 2cm, 1cm recall
- S1 stability signals
- S2 invalid/nonfinite behavior
- cross-iter eval drop if available

### Phase B - Diagnostic-Only Attention Observability

Add observability before changing fusion behavior.

The first implementation should not add any new loss and should not change
outputs when disabled.

Candidate config fields:

```text
--lmc_log_attention_stats False
--lmc_attention_stats_interval 100
--lmc_attention_stats_max_pixels 4096
```

Metrics to log:

- `attn_entropy_mean`
- `attn_entropy_p10`, `attn_entropy_p50`, `attn_entropy_p90`
- `token_usage_min`, `token_usage_max`
- `token_usage_top5`
- `effective_token_count = exp(entropy(token_usage))`
- `avg_max_attention = mean(max(attn over K))`
- `raw_feature_norm`
- `fused_feature_norm`
- `attention_out_norm`
- learned geometry gate/scale values, if GeoKey is enabled
- optional top-1 token spatial map for a small fixed image subset

Suggested log format:

```text
[LMC-Attn] iter=... phase=S1 entropy_mean=... effective_tokens=...
[LMC-Attn] usage_top5=[...] usage_min=... usage_max=... avg_max=...
[LMC-Fusion] raw_norm=... attn_out_norm=... fused_norm=...
```

Interpretation:

- Very low effective token count suggests token collapse.
- Very high entropy suggests routing is too soft or uninformative.
- Healthy usage does not need to be perfectly uniform. A scene may naturally use
  some tokens more than others.

Checkpoint metadata should save whether diagnostics were enabled and the
diagnostic sampling settings. Do not save large attention tensors in normal
checkpoints.

### Phase C - Progressive Geometry Injection / GeoKey v0

Goal: test whether memory-side geometry should influence query-memory matching
logits before using query-side 3D coordinates.

Run three cells:

```text
A0: value_only_raw
A1: value_only_norm
A2: geokey_norm
```

Use this progression:

- A0 preserves current behavior.
- A1 changes only the memory geometry scale/frequency entering the value path.
- A2 adds memory geometry to key input with zero-init scalar gate.

Compatibility requirement:

- Current value path is equivalent to `value_geo_scale = 1.0`.
- Do not accidentally change it to `0.5` via a sigmoid gate initialized at
  zero unless that is an intentional ablation.
- Prefer a direct scalar gate initialized at exactly zero for GeoKey.
- Fix `lmc_key_slice_idx=2` for the first A0/A1/A2 comparison.

Metadata to save:

- fusion geometry mode
- scene scale source
- scene scale value
- key geometry initial scale
- learned key geometry scale
- whether attention diagnostics are enabled
- `lmc_key_slice_idx`
- requested/effective LMC mode
- S1 sampled loss step mode

Decision rules:

- A1 > A0 suggests PE scale/frequency matters.
- A2 > A1 supports memory-side geometry entering fusion keys.
- A2 ~= A1 does not disprove full GeoMatch; it only says memory-only GeoKey was
  insufficient.
- A1/A2 both fail means inspect PE scale, gate learning, fusion/head dominance,
  and memory token quality before escalating.

Rollback criteria:

- NaNs or nonfinite ratio increases.
- S1/S2 cross-iter drop gets worse without metric gain.
- Metrics drop beyond normal seed variance.
- Learned key geometry scale behaves pathologically or diagnostics show
  degraded routing.

### Phase D - Usage Regularization

Only add usage regularization if Phase B shows token collapse.

Candidate loss:

```python
usage = mean(attn over batch, heads, query_pixels)
L_usage = KL(usage || Uniform(K))
```

First settings:

```text
lambda_usage in {0.001, 0.003, 0.01}
phase = S1 only
```

Why S1-only:

- Compressor/fusion are mainly trained in S1.
- S2 freezes compressor, so usage regularization in S2 can create confusing
  head/fusion-only pressure.

Stop if usage becomes artificially uniform while pose metrics get worse.

### Phase D2 - Fusion Residual Gate

This is a lower-priority ablation than GeoKey geometry.

Current residual form:

```python
x = LayerNorm(query_feats + attention_out)
```

Possible future form:

```python
x = LayerNorm(query_feats + residual_gate * attention_out)
```

Why it might help:

- It can prevent fusion from becoming a large, uncontrolled feature
  reparameterization.
- It gives diagnostics a clear scalar for how much the memory path is modifying
  raw query features.

Why it is not first:

- It does not solve the main issue that geometry is absent from key logits.
- It can reduce useful memory influence if initialized or regularized poorly.

Only consider this after attention diagnostics and A1/A2 are understood.

### Phase E - Detached Token-Routing Prior

Only consider query-side priors after A1/A2 and diagnostics are understood. The
first query-side prior should be token-routing, not coordinate-distance bias.

Motivation:

- Pre-fusion coarse coordinates are predicted from raw features, which may be
  weaker than the final fused features.
- In repeated or weak-texture regions, a coarse coordinate can be structurally
  wrong rather than just noisy.
- A hard distance bias around that wrong coordinate can suppress the correct
  memory tokens before the refined head gets a chance to recover.

Safer first form:

```text
raw/query feature
  -> routing head predicts r(token=k | query_i)
  -> detach or stop-gradient the prior path for the first ablation
  -> fusion logits += g * log(r + eps)
  -> ACE head remains primary output
```

Initial constraints:

- `g` starts at `0.0`.
- Cap or warm up `g` until the routing prior is proven useful.
- Keep image feature logits as the main matching path.
- Do not require the distribution to be one-hot; multi-modal uncertainty is a
  feature, not a bug.
- Log routing entropy, top-token maps, attention entropy, effective token
  count, and pose metrics together.

Success criteria:

- Pose metrics improve or stay stable while routing diagnostics become more
  interpretable.
- Improvements correlate with attention/token behavior rather than only final
  metric noise.
- If the routing prior only sharpens wrong attention, keep it diagnostic-only
  and do not progress to coordinate bias.

### Phase F - Confidence-Gated Coarse GeoMatch / Anchor-Assisted Branches

Only after token-routing diagnostics should coordinate-based GeoMatch be
considered. This phase also absorbs the useful EANR-MoE anchor-relative idea
without replacing the working ACE head.

Future detached coarse GeoMatch form:

```text
raw feature
  -> coarse ACE head predicts q_coord
  -> detach + normalize q_coord
  -> confidence c_i and nearby reward B(q_coord_norm, memory_p_norm)
  -> fusion logits += g * c_i * B
  -> refined ACE head predicts final coord
```

Constraints:

- No GT coordinate leakage.
- Coarse coordinate must be detached before entering the geometry bias.
- Distance bias gate starts at zero or uses an explicit warmup.
- Geometry scale must reuse the same scene-scale contract as A1/A2.
- Feature matching remains the main path; geometry bias is residual only.
- The first geometry bias should be a nearby reward, not a global far-token
  punishment:

```text
B_ik = alpha * exp(-||q_i - m_k||^2 / tau)
logits_ik = image_logits_ik + g * c_i * B_ik
```

- Add per-query confidence before applying the bias. Inference-compatible first
  choices are:
  - coarse-to-memory nearest distance, downweighting coordinates far from all
    memory tokens,
  - predicted uncertainty from the coarse head.
- Training-only reprojection validity/error can be logged for diagnostics, but
  it cannot be the inference-time confidence gate.
- Do not use a negative-distance penalty that suppresses all far tokens until a
  fallback-safe reward form has been shown to help.

Near-term form:

```text
ACE head remains primary output.
Anchor branch is auxiliary only.
```

Possible auxiliary path:

```python
routing_logits = route_head(fused_feats)
delta = delta_head(fused_feats)
P_anchor = weighted_sum(memory_p_k + delta_k)
L_total = L_ace_reproj + lambda_anchor * L_anchor_reproj
```

Recommended initial settings:

```text
top_m = 3
lambda_anchor = 0.05 or 0.1
temperature starts soft, then sharpens only if stable
```

Do not implement this before:

- true-global baseline comparison is understood,
- attention diagnostics exist,
- A1/A2 are either stable or rejected with evidence,
- token-routing prior has either helped or been rejected with diagnostics.

## Non-Goals Expanded

This near-term track should not include:

- memory extraction changes
- compressor redesign
- ACE head replacement
- normalized coordinate target training
- DSAC / PnP solver changes
- P4Pf or uncalibrated-query assumptions
- DSD / deformable token movement
- sub-memory routing
- learned relative geometry bias before token-routing and weak geometry priors
  are understood
- load balancing loss before diagnostics

## Experiment Naming

Use output directories that encode the single changed variable:

```text
indoor6_full_baselines_4090_forceglobal_fixedzero
indoor6_full_baselines_4090_forceglobal_s1_periter
indoor6_full_baselines_4090_forceglobal_s1_periter_a1_valnorm
indoor6_full_baselines_4090_forceglobal_s1_periter_a2_geokeynorm
indoor6_full_baselines_4090_forceglobal_s1_periter_a2_geokeynorm_attndiag
```

Do not mix old effective-local runs and true-global runs in the same folder.

## Minimal Verification Before Training

For code changes in this step:

```bash
conda run -n mapanything python -m py_compile \
  options_dinov2_lmc.py \
  trainer_dinov2_lmc.py \
  train_ace_dinov2_lmc.py \
  test_ace_dinov2_lmc.py \
  /home/xwh/project/ace_depth/ace_fusion.py \
  /home/xwh/project/ace_depth/ace_compressor.py
```

Behavior checks:

- Baseline config reconstructs old value-only fusion.
- A2 starts numerically equivalent to A1 when `key_geo_scale=0`.
- Old checkpoints still load.
- Logs include requested/effective LMC mode.
- Logs include fusion geometry mode and scene-scale source/value.
- Runtime diagnostics are off by default.
