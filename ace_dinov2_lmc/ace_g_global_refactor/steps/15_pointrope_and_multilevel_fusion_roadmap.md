# Step 15: Next-round ACE-G Architecture Roadmap

## 1. Purpose

This document consolidates the post-Step12/13/14 decision state for the next ACE-G architecture iteration. It is not a single mechanism proposal. It is a cross-direction roadmap after:

- completed B3-lite multi-level compression experiments,
- completed Step12/14 geometry encoding and distance-bias experiments,
- external web-model review of eight candidate schemes,
- code-level reassessment of feasibility, optimization risk, and expected gain.

The goal is to decide what should be implemented or tested next without rerunning already-negative directions unchanged.

## 2. Current reference and experiment context

The accepted reference remains FGPI-4090 / `4090_forceglobal_s1_periter`:

```text
true global
lmc_profile=legacy
s1_loss_step_mode=per_iter
lmc_key_slice_idx=2
lmc_feature_hierarchy_mode=selected_key_concat_value
lmc_fusion_geometry_mode=value_only_raw
pct5=83.66
pct2=34.24
med_t=2.6233 cm
med_r=0.2831 deg
```

Unless explicitly testing a new variable, future experiments should preserve:

```bash
--lmc_profile legacy
--s1_loss_step_mode per_iter
--lmc_key_slice_idx 2
--lmc_feature_hierarchy_mode selected_key_concat_value
--lmc_fusion_geometry_mode value_only_raw
```

## 3. Completed results that constrain this roadmap

### 3.1 Multi-level compression results

Current B3-lite / `levelwise_latent_merge` is negative:

| Run | pct5 | pct2 | med_t | med_r | Judgment |
|---|---:|---:|---:|---:|---|
| B3L-main | 77.43 | 24.90 | 2.96 cm | 0.31 deg | reject unchanged |
| B3L-diag | 80.16 | 28.02 | 2.84 cm | 0.30 deg | better but still below reference |

Interpretation: this does not falsify multi-level features. It says the current compression-side levelwise latent merge is too disruptive or in the wrong location.

### 3.2 Geometry encoding and distance-bias results

Post-train aggregated median over seeds `1305,2026,4242,7777,9001`, hypotheses=256:

| Run | Main change | pct5 | pct2 | med_t | med_r | Judgment |
|---|---|---:|---:|---:|---:|---|
| FGPI-4090 ref | current clean reference | 83.66 | 34.24 | 2.623 cm | 0.283 deg | baseline |
| GBR-main | residual multi-scale RBF distance bias | 78.99 | 30.35 | 2.90 cm | 0.30 deg | negative |
| FPE-main | Fourier PE v2 residual | 80.93 | 30.35 | 2.82 cm | 0.29 deg | below baseline |
| PointRoPE-main | axis-wise 3D RoPE on q/k | 81.71 | 33.46 | 2.73 cm | 0.28 deg | best geometry candidate |
| CRPB-main | MLP continuous relative position bias | 79.38 | 28.79 | 3.02 cm | 0.29 deg | negative |

Interpretation: PointRoPE is the only geometry branch close enough to justify immediate refinement. Current GBR and CRPB should not be rerun unchanged.

## 4. External proposal summary

The external review proposed eight schemes:

1. Zero-gated residual multi-level compression with a Layer12 anchor.
2. Layer12-guided cross-attention feature merging.
3. Zero-gated late-stage multi-scale query residual injection.
4. ViT-Adapter-style bottleneck skip connections.
5. Adaptive PointRoPE radius/base frequency.
6. Hybrid PointRoPE plus deterministic sinusoidal PE seed.
7. Decoupled 3D relative position bias.
8. Tanh-bounded logit bias with temperature scaling.

The broad direction is useful, but several engineering corrections are required before implementation.

## 5. Critical corrections to the external proposal

### 5.1 Multi-level query fusion is more complex than it looks

The proposal correctly identifies query-side multi-level fusion as promising, but underestimates implementation complexity.

`ace_fusion.py` currently receives only the already-extracted query feature tensor:

```text
query_feats: (B, N_q, C)
compressor_out: compressed memory tokens
scene_center: (B, 3)
```

Relevant code:

- `ace_fusion.py:109-157`: `LMCFusionBlock.forward()`.
- `ace_fusion.py:187-266`: `LMCFeatureFusion.forward()`.
- `trainer_dinov2_lmc.py:1511-1520`: `LMCFeatureFusion` construction.
- `trainer_dinov2_lmc.py:2658+`: `_fuse_features()`.
- `trainer_dinov2_lmc.py:4756-4763`: ACE-G S2 fused feature path.

A DPT/FPN-like query adapter needs access to intermediate DINOv2/query image features. These are not currently passed into `LMCFeatureFusion`. Therefore the real task includes:

1. finding or adding intermediate DINOv2 feature extraction,
2. deciding whether multi-level query features are stored in the buffer or recomputed online,
3. making S1/S2/train/test paths consistent,
4. saving all structure-affecting flags in `lmc_config`,
5. reconstructing the adapter in `test_ace_dinov2_lmc.py`.

So the expected optimization risk is low if zero-gated, but code complexity is medium-to-high.

### 5.2 PointRoPE is the best immediate geometry candidate

PointRoPE-main is below the reference but close:

```text
PointRoPE-main: pct5=81.71, pct2=33.46, med_t=2.73 cm, med_r=0.28 deg
FGPI reference: pct5=83.66, pct2=34.24, med_t=2.623 cm, med_r=0.283 deg
```

This makes PointRoPE a better near-term target than FPE-v2, GBR, or CRPB.

Relevant code:

- `ace_compressor.py:341-369`: `_apply_point_rope()`.
- `ace_compressor.py:380-389`: q/k PointRoPE application in `DecoupledCrossAttention.forward()`.
- `ace_compressor.py:648-666`: `GeoLMC` PointRoPE config validation.
- `options_dinov2_lmc.py:723-794`: CLI flags for `lmc_pos_encoding_mode`, Fourier-v2, and PointRoPE.

The most conservative next test is fixed-radius sensitivity before adding learnable parameters.

### 5.3 Current distance-bias failures do not prove the bias was too large

The external proposal hypothesizes GBR/CRPB failed because logit bias may have been too large and collapsed softmax. This is possible but not established.

Current implementation is often conservative at initialization:

- `ResidualRBFDistanceBias.alpha` can start at 0.0: `ace_compressor.py:235-260`.
- CRPB output layer can be zero-initialized: `ace_compressor.py:331-333`.

Therefore the failure may also be:

- bias stayed too small,
- gradients into bias were weak,
- compressor-side bias is the wrong location,
- scalar logit bias is less useful than relative key/value encoding,
- radius/normalization is wrong.

Before more distance-bias runs, add diagnostics:

```text
bias_abs_mean
bias_abs_max
qk_logit_abs_mean
bias_to_qk_ratio
attention_entropy_before_after_bias
final_rbf_alpha
per_head_bias_stats
```

### 5.4 Layer12 must remain an explicit anchor

For compression-side multi-level changes, avoid replacing the selected Layer12 reference path. The safer pattern is:

```text
z_out = z_l12 + gamma * adapter(z_multilevel)
```

with `gamma=0` at initialization. This is different from B3-lite, which merged per-level latents through a global softmax gate.

Relevant code:

- `ace_compressor.py:616-638`: feature hierarchy mode checks.
- `ace_compressor.py:693-710`: levelwise projection/gate initialization.
- `ace_compressor.py:999-1082`: `_forward_levelwise_latent_merge()`.
- `ace_compressor.py:1086+`: default `GeoLMC.forward()` path.

## 6. Final priority ranking

| Priority | Scheme | Reason |
|---:|---|---|
| 1 | PointRoPE radius/scale refinement | Closest geometry result; smallest code and optimization risk. |
| 2 | Cascading Internal Fusion Assembly | Safer fusion-side multi-level refinement; no backbone multi-layer extraction or buffer growth; zero-gated over first fusion block. |
| 3 | Layer12-anchored residual multi-level compression | Safe replacement for failed B3-lite; preserves baseline latent path. |
| 4 | Backbone multi-layer query adapter | Dense-prediction aligned but deferred because it requires intermediate DINOv2 features and buffer/recompute policy. |
| 5 | Deterministic sin/cos PE seed with PointRoPE | Addresses random Fourier seed, but should follow radius sensitivity. |
| 6 | Axial/decoupled 3D relative position bias | More structured than current CRPB, but requires diagnostics/constraints. |
| 7 | Layer12-guided cross-attention merging | Plausible but must be zero-gated; higher perturbation risk than 1.1. |
| 8 | ViT-Adapter-style skip | Mostly a variant of query-side residual adapter, not a separate first-priority direction. |
| 9 | Tanh-bounded logit bias | Should follow diagnostics; current failures are not proven to be bias-overflow failures. |

## 7. Next implementation candidates

### Candidate A: PointRoPE fixed-radius sensitivity

Purpose: test whether PointRoPE underperformed due to scene-scale mismatch.

Change only:

```bash
--lmc_pos_encoding_mode point_rope
--lmc_point_rope_radius 2.0   # or 3.0 in a second run
--lmc_geo_bias_mode legacy
```

Keep all other FGPI contract flags unchanged.

Expected value:

- very low code complexity,
- no new architecture code if current flags exist,
- directly tests whether PointRoPE can close the small gap to FGPI.

Decision rule:

- promote if it beats PointRoPE-main and approaches/exceeds FGPI on both `pct5` and `pct2`.
- if radius 2.0/3.0 are worse, do not add learnable radius immediately.

### Candidate B: Scene-derived PointRoPE radius

Purpose: replace fixed radius with a memory-derived scale such as:

```text
radius = percentile(||pooled_points - scene_center||, 90 or 95)
```

Code anchor:

- `ace_compressor.py:341-369`: apply radius in `_apply_point_rope()`.
- `ace_compressor.py:548-692`: store/resolve radius policy in `GeoLMC`.
- `options_dinov2_lmc.py:762-794`: add radius policy flag if needed.
- `trainer_dinov2_lmc.py:1461-1508`: pass the policy into `GeoLMC`.
- `test_ace_dinov2_lmc.py:307-390`: reconstruct the same policy for eval.

Safer design:

```text
point_rope_radius_policy = fixed | memory_p95
```

Default must remain `fixed` with radius 4.0.

### Candidate C: Fusion-side multi-level refinement

Purpose: introduce multi-level refinement without paying the storage and latency cost of extracting or buffering multiple DINOv2 backbone layers.

#### Candidate C1: Cascading Internal Fusion Assembly (preferred first version)

The safer near-term version moves the meaning of "multi-level" from the DINOv2 backbone to the fusion module itself. Query-side input remains the same single spatial feature tensor used by the current FGPI contract. Multi-level features are the internal states produced by a cascade of lightweight `LMCFusionBlock`s.

Desired computation:

```text
H0 = query_feats                         # existing single-layer query tokens
H1 = FusionBlock_1(H0, compressed_memory) # baseline anchor
H2 = FusionBlock_2(H1, compressed_memory)
H3 = FusionBlock_3(H2, compressed_memory)
H4 = FusionBlock_4(H3, compressed_memory)

residual = AssemblyMLP(concat(H2, H3, H4))
out = H1 + gamma * residual
```

with `gamma=0` at initialization.

This preserves the current buffer/backbone contract:

- no intermediate DINOv2 feature extraction,
- no multi-layer query feature buffer,
- no S1/S2 I/O increase,
- no extra memory-file schema requirement,
- train/test can keep the same public fusion forward API if implemented carefully.

Code anchors:

- `ace_fusion.py:109-157`: `LMCFusionBlock.forward()` already maps `(B, N_q, C)` query tokens and compressed memory back to `(B, N_q, C)`.
- `ace_fusion.py:187-266`: `LMCFeatureFusion.forward()` is the main integration site.
- `trainer_dinov2_lmc.py:2675-2714`: `_fuse_features()` reshapes `(B,C,H,W)` to query tokens and calls fusion.
- `test_ace_dinov2_lmc.py:490-528`: test path reconstructs the same fusion module and calls it with cached compressed memory.
- `test_ace_dinov2_lmc_ensemble.py`: ensemble reconstruction must remain compatible if checkpoints use this mode.

Design constraints:

- Preserve the public fusion call signature in the first implementation:
  `forward(query_feats, compressor_out, scene_center, ...)`.
- Save every architecture-affecting field in checkpoint `lmc_config`, e.g.:
  - `lmc_fusion_refinement_mode = single | cascade_internal`,
  - `lmc_fusion_cascade_layers`,
  - `lmc_fusion_assembly_mode`,
  - `lmc_fusion_assembly_gamma_init`.
- Ensure training, single-checkpoint test, and ensemble test reconstruct the exact same module.
- Do not combine C1 with new PointRoPE radius policies, new bias modules, or multi-memory routing in the first run.
- Use a legacy/single-fusion control after adding code to confirm default-path behavior does not drift.

Important correction: CLS-token guidance is not part of C1. Current `_fuse_features()` and test paths only pass spatial query features, not DINOv2 CLS tokens. Adding CLS modulation would require feature-extraction API, trainer, test, ensemble, and checkpoint changes. It should be treated as a later C2/C3 extension, not the first internal-cascade experiment.

#### Candidate C2: Backbone multi-layer query adapter (deferred)

The previous DPT/FPN-style plan remains valid as a longer-term option, but it is no longer the first implementation target because it requires intermediate DINOv2/query features and raises buffer/recompute complexity.

Deferred computation:

```text
fused = LMCFeatureFusion(query_L12, compressed_memory)
residual = Adapter([query_L6, query_L12, query_L18, query_L24])
out = fused + gamma * residual
```

with `gamma=0` at initialization.

C2 requires:

- locating or adding intermediate DINOv2 feature extraction,
- deciding whether multi-level query features are stored in the buffer or recomputed online,
- keeping S1/S2/train/test paths identical,
- saving all adapter config in checkpoint `lmc_config`.

### Candidate D: Layer12-anchored residual multi-level compression

Purpose: retry multi-level compression in a safer form than B3-lite.

Desired computation:

```text
z_l12, p = baseline selected_key_concat_value compression
z_ml = multi-level compression branch
z_out = z_l12 + gamma * Linear(z_ml)
```

with `gamma=0` at initialization.

Code anchors:

- `ace_compressor.py:548+`: `GeoLMC.__init__`.
- `ace_compressor.py:693-710`: multi-level projection/gate modules.
- `ace_compressor.py:999-1082`: existing levelwise branch to reuse carefully.
- `ace_compressor.py:1086+`: default forward path and new anchored residual branch.
- `options_dinov2_lmc.py:646-653`: add a new mode such as `anchored_level_residual`.

Do not reuse `levelwise_latent_merge` unchanged.

### Candidate E: Distance-bias diagnostics before redesign

Purpose: understand why GBR/CRPB failed before proposing another bias module.

Code anchors:

- `ace_compressor.py:211-272`: RBF residual bias.
- `ace_compressor.py:323-333`: CRPB MLP.
- `ace_compressor.py:371-398`: CRPB and bias addition to logits.
- `ace_compressor.py:925-949`: current geo bias runtime stats.

Only after diagnostics should one consider:

- axial decoupled RPB,
- bounded/tanh logit bias,
- relative key/value encoding,
- fusion-side geometric bias.

## 8. Guardrails for all next experiments

1. Change one primary mechanism at a time.
2. Keep `lmc_profile=legacy` unless explicitly testing training dynamics.
3. Keep `s1_loss_step_mode=per_iter`.
4. Preserve `lmc_key_slice_idx=2` unless the experiment is explicitly about key-layer choice.
5. Use zero-gated residual branches for multi-level additions.
6. Save every new architecture-affecting flag in checkpoint `lmc_config`.
7. Ensure `test_ace_dinov2_lmc.py` reconstructs the exact same modules.
8. Do not compare single-seed summaries against 5-seed post-train aggregated medians.
9. Do not rerun B3-lite, GBR, CRPB, or `mapany_flow_v1` unchanged.

## 9. Recommended immediate next step

If GPUs are available now, the cleanest next run is PointRoPE fixed-radius sensitivity:

```bash
--lmc_pos_encoding_mode point_rope
--lmc_point_rope_radius 2.0
--lmc_geo_bias_mode legacy
--lmc_profile legacy
--s1_loss_step_mode per_iter
```

## 10. Update: TP/PointRoPE interrupted sweep and next two-GPU matrix

A later TP/profile sweep partially completed under:

```text
04_evaluation/train_compare/indoor6_full_baselines_TP_mapany_flow_v1
```

This sweep used `--lmc_profile mapany_flow_v1`, which is not the accepted FGPI reference contract. The completed plain TP run is negative:

| Run | Completion | best pct5 | best pct2 | best med_t | best med_r | Judgment |
|---|---:|---:|---:|---:|---:|---|
| TP-mapany-flow-v1 | 28/28 + post-train | 78.60 | 30.35 | 2.902 cm | 0.294 deg | reject unchanged |

The post-train 5-seed median for plain TP is worse:

```text
pct5=74.71
pct2=26.46
med_t=3.0216 cm
med_r=0.3289 deg
```

Therefore `mapany_flow_v1` should remain a negative training-profile direction and should not be expanded unchanged.

### 10.1 Partial PointRoPE signals inside the negative TP profile

Despite the negative profile, two PointRoPE signals are informative:

| Run | Completion | Interesting iteration | pct5 | pct2 | med_t | med_r | Signal |
|---|---:|---:|---:|---:|---:|---:|---|
| TP + PointRoPE r2 fixed + Fourier seed | 19/28 | iter17 | 82.10 | 30.35 | 2.921 cm | 0.304 deg | best threshold signal in TP sweep |
| TP + PointRoPE memory_p95 + Fourier seed | 20/28 | iter18 | 77.82 | 31.13 | 2.620 cm | 0.285 deg | best median-error signal |
| TP + PointRoPE r3 fixed + Fourier seed | 20/28 | iter20 | 80.93 | 32.30 | 2.889 cm | 0.312 deg | secondary diagnostic only |
| TP + PointRoPE r4 fixed + sincos seed | 19/28 | iter19 | 78.99 | 28.02 | 2.875 cm | 0.309 deg | negative |

Interpretation:

- r2 fixed radius may improve threshold recall (`pct5`) but not median translation/rotation.
- memory_p95 radius may improve central tendency (`med_t`, `med_r`) but not the threshold tail.
- r3 is not stronger than the two signals above.
- deterministic sin/cos seed is not supported in this TP combination.

### 10.2 Immediate use of four GPUs

If four GPUs are available and two are already allocated to complete TP r2 and TP memory_p95, use the remaining two GPUs to migrate the useful signals back to the accepted FGPI/legacy contract instead of running more TP-profile variants.

Recommended remaining two-GPU matrix:

| GPU | Experiment | Purpose |
|---|---|---|
| free GPU A | FGPI-contract PointRoPE r2 fixed + Fourier seed | Test whether r2 threshold signal survives without negative TP profile. |
| free GPU B | FGPI-contract PointRoPE memory_p95 + Fourier seed | Test whether memory_p95 median-error signal survives without negative TP profile. |

Keep the accepted reference contract:

```bash
--lmc_profile legacy
--s1_loss_step_mode per_iter
--lmc_key_slice_idx 2
--lmc_feature_hierarchy_mode selected_key_concat_value
--lmc_fusion_geometry_mode value_only_raw
--lmc_geo_bias_mode legacy
```

Only change the PointRoPE radius policy:

```bash
# r2 fixed
--lmc_pos_encoding_mode point_rope
--lmc_point_rope_radius_policy fixed
--lmc_point_rope_radius 2.0
--lmc_point_rope_seed_pe fourier_legacy
--lmc_point_rope_base 10000
--lmc_point_rope_axes xyz_split
--lmc_point_rope_apply_to qk

# memory_p95
--lmc_pos_encoding_mode point_rope
--lmc_point_rope_radius_policy memory_p95
--lmc_point_rope_radius 4.0
--lmc_point_rope_seed_pe fourier_legacy
--lmc_point_rope_base 10000
--lmc_point_rope_axes xyz_split
--lmc_point_rope_apply_to qk
```

### 10.3 Decision rules

Promote only if a 5-seed post-train evaluation shows one of:

1. r2 fixed improves over PointRoPE-main and closes the FGPI gap on both `pct5` and `pct2`.
2. memory_p95 matches or improves FGPI-level `med_t`/`med_r` without a large drop in `pct5`/`pct2`.
3. Either setting beats FGPI on the primary metric (`pct5`) without worse `pct2` and median errors.

Reject or demote if:

- improvements appear only inside `mapany_flow_v1`,
- median error improves but `pct5`/`pct2` remain far below FGPI,
- a setting improves one metric by trading off too much tail accuracy.

### 10.4 Architectural implication

If memory_p95 repeatedly improves median error but hurts thresholds, treat it as a scale-policy signal rather than a final setting. The next architecture question becomes how to combine:

```text
fixed small radius for threshold/tail recall
memory-derived radius for central median accuracy
```

Possible later designs, after the FGPI-contract check:

- per-head mixed radii: some heads use r2 fixed, some use memory_p95;
- zero-gated residual memory_p95 branch over r2 baseline;
- learned bounded interpolation between fixed and memory-derived radius.

Do not implement these until the two FGPI-contract validation runs finish.
