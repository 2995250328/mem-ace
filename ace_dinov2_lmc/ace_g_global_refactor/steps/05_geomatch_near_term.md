# Step 05 - GeoMatch Near-Term Absorption

Status: designing

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

## Near-Term Sequence

1. Finish and compare true-global baselines.
   - `global + fixed_zero`
   - `global + per_iter`
   - Do not merge architectural changes until these runs clarify the baseline.

2. Add attention and token diagnostics first.
   - Return or record fusion attention without changing training behavior.
   - Track token usage histogram, attention entropy, effective token count, and
     optionally top-1 token maps.
   - Use these diagnostics to decide whether token collapse is real.

3. Add GeoMatch Fusion v1 as a controlled ablation.
   - Preserve current value-path geometry behavior as closely as possible.
   - Add a gated geometry path into key projection / matching logits.
   - Initialize key geometry weakly so the old feature matching path remains the
     default at the start of training.
   - Keep ACE head, compressor, memory extraction, and loss contract unchanged.

4. Consider usage regularization only if diagnostics justify it.
   - Start S1-only.
   - Try small weights only, such as `0.001`, `0.003`, `0.01`.
   - Do not add load balancing loss before observing collapse.

5. Add anchor-assisted residual branch only after GeoMatch is stable.
   - Keep ACE head as the primary prediction.
   - Add anchor-relative coordinate prediction as an auxiliary branch with small
     loss weight.
   - Do not replace the ACE head in this near-term track.

## GeoMatch Fusion v1 Design Notes

The proposed direction is:

```python
pe = coord_encoder(memory_p - scene_center)
pe_mem = pe_proj(pe)

k_input = memory_z + key_geo_scale * pe_mem
v_input = memory_z + value_geo_scale * pe_mem

k = k_proj(k_input)
v = v_proj(v_input)
```

Compatibility detail:

- The existing value path is equivalent to `value_geo_scale = 1.0`.
- If using `sigmoid(gate)` parameters, initialize the value gate close to `1.0`,
  not `0.5`, unless intentionally changing the old baseline.
- Initialize key geometry weakly, for example with a scale near zero.

## Non-Goals For This Step

- No memory extraction changes.
- No compressor redesign.
- No ACE head replacement.
- No coordinate normalization contract change.
- No reference-frame point-cloud conversion.
- No multi-scene joint training.
- No sub-memory routing.
- No ray / Plucker feature injection.
- No PnP / RANSAC solver change.
- No P4Pf or uncalibrated-query assumptions.
- No usage/load-balancing loss before diagnostics.

Related but deferred ideas:

- Multi-scene universal LMC, reference-frame memory coordinates, full
  coordinate normalization, sub-memory routing, MapAnything/VGGT acceleration,
  ray / Plucker geometry, and MoE heads are tracked in
  `../LONG_TERM_RESEARCH_PLAN.md`.
- They are important for the eventual system, but they change data contracts,
  output spaces, or memory construction. They should not be mixed into the
  first attention-diagnostics or GeoMatch Fusion v1 ablations.

## Success Criteria

- Logs/checkpoints identify whether GeoMatch Fusion is enabled.
- Attention diagnostics are available for true-global runs.
- GeoMatch v1 can be run as a clean ablation against the current value-only
  fusion baseline.
- If it fails, it can be disabled without invalidating the baseline path.

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
- learned geometry gate/scale values, if GeoMatch Fusion is enabled
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

### Phase C - GeoMatch Fusion v1

Goal: let memory geometry influence query-memory matching logits, not only the
value path.

Current baseline:

```python
k_input = memory_z
v_input = memory_z + pe_mem
```

GeoMatch v1 direction:

```python
pe = coord_encoder(memory_p - scene_center)
pe_mem = pe_proj(pe)

k_input = memory_z + key_geo_scale * pe_mem
v_input = memory_z + value_geo_scale * pe_mem

k = k_proj(k_input)
v = v_proj(v_input)
```

Compatibility requirement:

- Current value path is equivalent to `value_geo_scale = 1.0`.
- Do not accidentally change it to `0.5` via a sigmoid gate initialized at
  zero unless that is an intentional ablation.
- Key geometry should start near zero so old feature matching remains dominant
  at initialization.

Recommended first ablation:

```text
fusion_geometry_mode=value_only
fusion_geometry_mode=gated_key_value
key_geo_init_scale=0.0 or very small
value_geo_init_scale=1.0
```

Candidate config fields:

```text
--lmc_fusion_geometry_mode value_only
--lmc_fusion_key_geo_init 0.0
--lmc_fusion_value_geo_init 1.0
--lmc_fusion_return_attn False
```

Metadata to save:

- fusion geometry mode
- key/value geometry scale initial values
- learned key/value geometry scale values, if trainable
- whether attention diagnostics are enabled

Success criteria:

- `value_only` exactly preserves the current baseline.
- `gated_key_value` is a clean one-variable ablation.
- Old checkpoints can still load through a deliberate compatibility path.
- Logs identify the fusion mode and geometry scale/gate values.

Rollback criteria:

- NaNs or nonfinite ratio increases.
- S1/S2 cross-iter drop gets worse without metric gain.
- Metrics drop beyond normal seed variance.
- Learned key geometry stays near zero and diagnostics show no routing change.

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

This is a lower-priority ablation than GeoMatch key/value geometry.

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

Only consider this after attention diagnostics and GeoMatch v1 are understood.

### Phase E - Anchor-Assisted Residual Branch

This absorbs the useful EANR-MoE anchor-relative idea without replacing the
working ACE head.

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
- GeoMatch Fusion v1 is either stable or rejected with evidence.

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
- load balancing loss before diagnostics

## Experiment Naming

Use output directories that encode the single changed variable:

```text
indoor6_full_baselines_4090_forceglobal_fixedzero
indoor6_full_baselines_4090_forceglobal_s1_periter
indoor6_full_baselines_4090_forceglobal_attndiag
indoor6_full_baselines_4090_forceglobal_geomatch_v1
indoor6_full_baselines_4090_forceglobal_geomatch_usage001
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
- Old checkpoints still load.
- Logs include requested/effective LMC mode.
- Logs include fusion geometry mode when enabled.
- Attention diagnostics are off by default.
