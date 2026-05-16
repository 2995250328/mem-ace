# ACE-G Global Code Audit and Refactor TODO

Date: 2026-05-03

Scope: only the real training and evaluation chain for `--lmc_flow ace_g` with effective `lmc_mode=global`. This note intentionally does not review local, hierarchical, learned, or the memory construction pipeline except where their shared code leaks into the global path.

## Current Contract

ACE-G global is a two-stage outer loop in `trainer_dinov2_lmc.py`.

1. S1 trains `GeoLMC + LMCFeatureFusion + Head`.
2. S2-G freezes `GeoLMC`, caches one compressed memory output, then trains `Head` or `Head + LMCFeatureFusion`.
3. The DINOv2 encoder is frozen and only produces raw feature maps.
4. ACE-G S2 buffers store raw backbone feature rows, not fused feature rows.
5. The supervision remains ACE-style scene-coordinate reprojection loss, not a feature-level objective.

This contract is mostly implemented, but several parts are rough enough to deserve cleanup before more experiments are trusted as final.

## Critical Review of GeoMatch-ACE-G Design Doc

Source:

- `GeoMatch-ACE-G_技术设计文档.docx`

The historical GeoMatch document is useful, but it mixes near-term fixes with long-term architecture speculation. It should not be copied into the implementation roadmap wholesale. The parts worth keeping are the ones that satisfy three constraints:

1. They preserve the current ACE-G S1/S2 contract.
2. They can be toggled and ablated independently.
3. They do not replace the ACE head, reprojection supervision, or S2 compressor-freeze rule.

Keep for the near-term roadmap:

- GeoMatch Fusion v1: inject memory coordinate PE into fusion `key` as well as `value`, controlled by learnable gates.
- Attention diagnostics: return/log fusion attention so token usage, entropy, effective token count, and gate values are inspectable.
- Conditional usage loss: only consider a small token-usage regularizer after diagnostics show token collapse.
- Scene-scale contract: use a stable `scene_scale` for PE and distance-bias normalization before attempting full normalized-coordinate training.

Defer to a separate research track:

- Full EANR-MoE / U-LCA-MoE replacement of the ACE head.
- Anchor residual branch as a primary output path.
- DSD-style deformable/local compressor.
- Sub-memory routing for very large scenes.
- Full normalized-coordinate target training.

Why defer these:

- They change too many variables at once: fusion, head, target coordinate space, loss, and sometimes routing/solver behavior.
- They make it hard to tell whether gains come from LMC token quality, geometry injection, head changes, or changed target scaling.
- The current best-performing path is still ACE-G global with the old ACE head; near-term work should improve observability and local fusion/compressor choices first.

## High-Priority TODOs

### 1. Make global FPS deterministic or checkpoint the selected latent set

Status: done. Fixed in code on 2026-05-07 and tracked in `ace_g_global_refactor/steps/01_deterministic_fps.md`.

Implemented first change:

- `farthest_point_sampling()` now has an explicit `start_policy`.
- Default policy is deterministic `farthest_from_center`.
- `legacy_random` is retained as an opt-in rollback path via `--lmc_fps_start_policy legacy_random`.
- Training stores `lmc_fps_start_policy` in `lmc_config`.
- Evaluation rebuilds `GeoLMC` from the saved `lmc_config`, so train/eval use the same FPS start policy.

Residual note:

- Old checkpoints do not contain the selected latent coordinates and cannot exactly recover their original random FPS start. They will use the deterministic default unless evaluated with a manually patched config or retrained with the new setting.

Location:

- `/home/xwh/project/ace_depth/ace_compressor.py:43`
- `/home/xwh/project/ace_depth/ace_compressor.py:49`
- `trainer_dinov2_lmc.py:2753`
- `test_ace_dinov2_lmc.py:330`

`farthest_point_sampling()` chooses its initial point with `torch.randint()`. In global mode this decides `latent_coords`, so it is not just harmless sampling noise. The selected latent coordinates affect compressor queries, geometric attention bias, fusion value positional encoding, and therefore the head's learned input distribution.

Why this is rough:

- S1 calls `self.compressor(self.memory_dict)` repeatedly, with a fresh FPS start each forward.
- S2 calls `_compress_memory()` once per iteration and trains against that cached latent set.
- Test-time compression calls the compressor again and may select a different latent set than the one S2 adapted to.
- Checkpoints save compressor and fusion weights, but not the actual compressed latent coordinates or cached S2 compressor output.

Modification options:

- Minimum fix: add deterministic FPS start, for example first point is the point farthest from `scene_center`, or the lowest/highest stable index after a deterministic score.
- Reproducibility fix: add an optional `torch.Generator` or `fps_seed` parameter and save it in `lmc_config`.
- Strongest consistency fix: checkpoint the final compressed latent output used by the best model, or checkpoint `latent_coords` and recompute only `latent_z` if needed.

Recommended first change: deterministic FPS start. It is simple, keeps checkpoint format smaller, and removes the largest train/eval randomness source.

### 2. Make key-layer selection explicit and ablate it

Location:

- `/home/xwh/project/ace_depth/ace_compressor.py:308`
- `/home/xwh/project/ace_depth/ace_compressor.py:348`
- `memory_extraction/run_memory_extraction.py:1005`
- `memory_extraction/run_memory_extraction.py:1218`
- `/home/xwh/project/map-anything/mapanything/models/mapanything/model.py:1881`

`GeoLMC.forward()` calls `_get_layer_slice(pooled_features, 1)`. The helper then computes:

```python
start = (layer_idx + 1) * C
end = (layer_idx + 2) * C
```

So `layer_idx=1` actually selects the third slice under zero-based indexing. For current MapAnything memory, the saved layer order is:

```text
slice 0: intermediate[0]
slice 1: intermediate[6]
slice 2: intermediate[12]
slice 3: intermediate[18]
slice 4: final
```

Actual `memory_pooled.pt` and `memory_bse.pt` artifacts inspected for `scene2a` store:

```text
layers_idx = [0, 6, 12, 18, 'final']
feature_dim = 3840 = 5 * 768
```

MapAnything storage mode calls `info_sharing(..., return_input_as_first_intermediate=True)`, so `intermediate[0]` may be the input to the info-sharing attention stack rather than a post-attention block output. This makes the helper's implicit `+1` skip plausibly intentional: it may be trying to avoid using that raw/pre-attention slice as the key. The rough part is that the caller passes `1`, so the current global compressor does not select the first post-skip candidate `layer 6`; it selects `layer 12`.

Because `k_proj` uses this single slice while `v_proj` uses the full concatenated feature, this key-layer choice is central to global compressor behavior.

Why this is rough:

- It looks like an off-by-one bug even if it is an intentional "skip raw input" convention.
- The actual current key is `slice 2 / layer 12`, but this is not logged or saved in config.
- If the intended policy is "skip raw input and use the first attention-processed intermediate", the current call should select `slice 1 / layer 6`.
- If the intended policy is "use a mid-layer key", then `layer 12` may be reasonable, but it must be explicit and ablated.
- The checkpoint does not record enough metadata to reconstruct the semantic key-layer choice during later analysis.

Layer-choice recommendation:

- Do not use `slice 0` as the default key. It is likely too close to raw MapAnything input and may lack info-sharing context.
- Do not assume `final` is best. It may be semantically strong but less locally discriminative as an attention key.
- Treat `layer 12` and `layer 18` as the most plausible single-layer key candidates.
- Keep value as full multi-layer concat, because it should preserve all memory content.
- Add a `learned scalar mix` key ablation after the single-layer candidates are measured:

```text
key_feat = sum_l softmax(w_l) * feat_l
value    = concat(feat_0, feat_6, feat_12, feat_18, feat_final)
```

This keeps the key projection input dimension at one layer width while letting the model choose among intermediate/final features. Start with `slice 0` excluded or initialized with a low weight.

Recommended first change: replace the implicit helper with explicit config such as `lmc_key_slice_idx` or `lmc_key_layer_label`, defaulting to the current actual behavior (`slice 2 / layer 12`) for backward compatibility. Log `layers_idx`, selected slice index, selected layer label, and slice width at compressor construction.

### 3. Save all structure-affecting LMC config fields

Locations:

- `trainer_dinov2_lmc.py:498`
- `trainer_dinov2_lmc.py:533`
- `test_ace_dinov2_lmc.py:275`

Training passes `pe_normalize_input`, `geo_sigma`, `num_fine`, and `num_coarse` to `GeoLMC`, but the checkpoint config does not persist all of them. Test-time reconstruction currently uses defaults for omitted fields.

Why this is rough:

- If `--pe_normalize_input True` is used, eval silently rebuilds a different compressor.
- `geo_sigma` is irrelevant to global mode's current attention layer, but it is still a structural constructor argument for shared modes and should not be implicit.
- Missing config fields make old experiment directories harder to audit.

Recommended first change: add `pe_normalize_input`, `geo_sigma`, `num_fine`, `num_coarse`, and `backbone_feature_dim` to `lmc_config`, and make test-time reconstruction consume them.

### 4. Unify reprojection and invalid-loss code paths

Locations:

- `trainer_dinov2_lmc.py:1212`
- `trainer_dinov2_lmc.py:1313`
- `trainer_dinov2_lmc.py:2911`
- `trainer_dinov2_lmc.py:3061`

The same reprojection pipeline is duplicated across full-map S1, sampled S1, standard LMC S2, and ACE-G S2. The duplicated paths are already behaviorally different.

Examples:

- S2 clamps `loss_invalid` through `loss_invalid_max_delta`; S1 sampled/full-map does not consistently share that behavior.
- S1 sampled mode uses `iter_for_loss = 0` outside `mapany_flow_v1`, while full-map S1 and S2 have their own time-axis logic.
- Mask shape handling and non-finite handling are copied with small differences.

Why this is rough:

- Fixes to numeric stability can easily land in only one stage.
- It is hard to reason about whether S1 and S2 supervise the same contract.
- ACE-G S2 duplicates most of standard S2 only to insert fusion before head.

Recommended first change: factor out one helper that accepts either `pred_scene_coords_b3HW` or `features_bCHW`, plus `target_px`, poses, intrinsics, `step_eff`, and stage metadata. ACE-G should only be responsible for producing fused features; loss computation should be shared.

### 5. Make S1 sampled-feature ReproLoss time explicit

Location:

- `trainer_dinov2_lmc.py:1357`
- `trainer_dinov2_lmc.py:1689`
- `trainer_dinov2_lmc.py:1997`

In `_s1_compute_loss_from_features()`, non-`mapany_flow_v1` sampled S1 uses `iter_for_loss = 0`. That keeps `dyntanh` at its easiest setting for all sampled S1 updates, even after many outer iterations.

Why this is rough:

- The behavior is not obvious from the option name `s1_loss_mode=sample_per_image`.
- It diverges from full-map S1 and S2 loss scheduling.
- It makes S1 loss numbers across modes less comparable.

Recommended first change: pass `loss_step` into `_s1_compute_loss_from_features()` from both S1-buffer and online sampled paths, and document the intended S1 schedule.

### 6. Prevent accidental effective-mode changes during global experiments

Location:

- `trainer_dinov2_lmc.py:417`
- `options_dinov2_lmc.py:498`

`lmc_auto_mode_by_visibility=True` can silently change requested `global` into `local` or `hierarchical`. That may be useful generally, but it is dangerous for ACE-G global comparisons. The default is now `False`, so the requested `lmc_mode` is honored unless fallback is explicitly enabled.

Why this is rough:

- A command line that says `--lmc_mode global` may produce a non-global checkpoint.
- Experiment paths and summaries can be misread if only the requested mode is inspected.

Status: addressed for the default path. `lmc_auto_mode_by_visibility` defaults to `False`, and checkpoints/logs record both `requested_lmc_mode` and `effective_lmc_mode`. Fallback remains available only as an explicit opt-in.

### 7. Restore module training modes instead of forcing them

Locations:

- `trainer_dinov2_lmc.py:1021`
- `trainer_dinov2_lmc.py:1130`

`_compress_memory()` always ends with `self.compressor.train()`. `create_training_buffer_ace_g()` always ends with `self.regressor.train()`. These helpers should restore prior state instead of assuming what the next stage wants.

Why this is rough:

- It relies on caller order to repair module states.
- It makes debugging eval-in-train and cross-iteration eval more fragile.

Recommended first change: save `was_training` for compressor, fusion, regressor, encoder, and head where relevant, then restore exact previous states in `finally`.

### 8. Enforce S2 compressor freezing at the parameter level

Locations:

- `trainer_dinov2_lmc.py:2175`
- `trainer_dinov2_lmc.py:2753`
- `trainer_dinov2_lmc.py:3093`

S2 currently freezes compressor by no-grad cached output and by excluding compressor from `optimizer_head`. That is enough for the current code, but it is an implicit contract.

Why this is rough:

- Future edits could accidentally route a live compressor forward through S2 and create gradients.
- The code says "compressor frozen" but `requires_grad` remains true.

Recommended first change: add small helpers such as `_set_compressor_trainable(True/False)` and call them at S1/S2 boundaries.

### 9. Reduce the fake `1 x C x 16 x W` head-packing leakage

Status: addressed for current non-architecture hygiene. The sampled-row packer
is now named `_pack_feature_rows_for_head(...)`, the old helper remains as a
compatibility wrapper, and per-stage trim counts are accumulated.

Location:

- `trainer_dinov2_lmc.py:674`
- `trainer_dinov2_lmc.py:1330`
- `trainer_dinov2_lmc.py:1686`
- `trainer_dinov2_lmc.py:3080`

The head is made of 1x1 convolutions, so sampled feature rows are packed into a fake spatial grid only to reuse the old head API. The fixed height `16` is a magic number and trimming is silent after the first warning per stage.

Why this is rough:

- It hides sample dropping.
- It couples S1/S2 sample counts to a fake spatial shape.
- It makes code harder to read because the tensor shape suggests an image even when it is just sampled rows.

Recommended first change: add a clearly named helper, for example `_pack_feature_rows_for_head(features_bC, aligned_tensors, grid_h=16)`, and log trim counts in summary metrics.

### 10. Promote feature-dimension mismatch from warning to error

Status: addressed. Non-divisible `pooled_features_dim / num_layers` now raises
`ValueError` immediately with `pooled_features_dim`, `num_layers`,
`layers_idx`, and `memory_path`.

Location:

- `trainer_dinov2_lmc.py:451`
- `/home/xwh/project/ace_depth/ace_compressor.py:261`

If `pooled_features_dim` is not divisible by `num_layers`, the trainer only logs a warning and then constructs `GeoLMC` using truncated `feature_dim`. `v_proj` still expects `feature_dim * num_layers`, so this can become a late shape error.

Recommended first change: raise `ValueError` immediately with `pooled_features_dim`, `num_layers`, `layers_idx`, and `memory_path`.

### 11. Clean up preset/config drift

Status: addressed conservatively. `memory_compare_ace_g_v2` is now accepted by
`--train_preset`; existing preset defaults were not changed.

Locations:

- `train_ace_dinov2_lmc.py:60`
- `options_dinov2_lmc.py:321`

`TRAIN_PRESET_DEFAULTS` contains `memory_compare_ace_g_v2`, but `--train_preset` choices only expose `memory_compare_ace_g_v1`.

Recommended first change: either expose v2 in `choices` or remove the dead preset until it is ready.

### 12. Add GeoMatch Fusion v1 as a controlled ablation, not a replacement architecture

Locations:

- `/home/xwh/project/ace_depth/ace_fusion.py:11`
- `/home/xwh/project/ace_depth/ace_fusion.py:47`
- `trainer_dinov2_lmc.py:1033`
- `trainer_dinov2_lmc.py:3061`
- `test_ace_dinov2_lmc.py:380`

The GeoMatch design document correctly identifies a local structural weakness: current fusion uses memory geometry only in the value path, so 3D coordinates enrich what is read but do not help choose which memory token to read. This is worth testing before larger head or compressor redesigns.

The near-term version should be conservative:

```text
pe = PE(memory_p - scene_center)
pe_mem = pe_proj(pe)

gate_k = sigmoid(key_geo_gate)
gate_v = sigmoid(value_geo_gate)

q = q_proj(query_feats)
k = k_proj(memory_z + gate_k * pe_mem)
v = v_proj(memory_z + gate_v * pe_mem)
```

Recommended initialization:

```text
key_geo_gate   = -4.0   # small sigmoid value; starts close to current feature-only key matching
value_geo_gate =  0.0   # moderate sigmoid value; keeps value-geometry behavior active
```

Why this is rough in the current code:

- Geometry cannot directly affect fusion matching logits.
- Attention is not returned or logged, so token collapse cannot be diagnosed.
- Any future usage loss would be blind unless attention statistics are available first.

Recommended first change:

- Add config-gated GeoMatch Fusion v1 while keeping the current value-only fusion as the default baseline.
- Add optional `return_attn` support or a low-overhead diagnostics path for fusion attention.
- Log token usage, attention entropy, effective token count, and gate values before adding any new loss.
- Only enable token usage regularization if diagnostics show actual collapse.

## Global Compressor Design Notes

Implementation:

- `/home/xwh/project/ace_depth/ace_compressor.py:19`
- `/home/xwh/project/ace_depth/ace_compressor.py:133`
- `/home/xwh/project/ace_depth/ace_compressor.py:229`
- `/home/xwh/project/ace_depth/ace_compressor.py:331`

### Positional encoding

`FourierPositionEncoding` uses a random Gaussian projection buffer `B_gauss`, followed by sin/cos and a two-layer MLP. The Gaussian matrix is not trainable, but it is randomly initialized when the module is constructed and then saved in the checkpoint state dict.

In global mode, compressor PE is applied to:

```text
latent_coords - scene_center
```

Fusion has its own separate PE instance and applies it to:

```text
memory_p - scene_center
```

Rough points:

- There is no explicit coordinate scale contract near the PE. `pe_normalize_input=False` by default, and when enabled the normalization divides by batch std along dim 1. For a single scene memory this may be stable, but it is still a hidden per-forward rescaling.
- Compressor PE and fusion PE are independent. That may be fine, but it means the same 3D coordinate is embedded in two unrelated random Fourier bases.
- The random Gaussian basis has weak semantics. A run may work, but the frequency scale is not tied to the physical scene scale or memory density.
- `num_frequencies=10` is a small, opaque basis for 3D scene coordinates. It is hard to know whether failures come from compression, fusion, or an under/over-scaled PE.
- The only PE diagnostic logs min/max once. It does not log coordinate std, scene scale, or whether `pe_normalize_input` is active.

Modification ideas:

- Persist and report `pe_normalize_input` in `lmc_config`.
- Define a stable coordinate normalization contract:

```text
coords_norm = (coords - scene_center) / scene_scale
```

`scene_scale` should be logged and can be based on bbox radius, coordinate std, or percentile radius. This same scale should feed compressor PE, fusion PE, and global distance bias.

- Add a short compressor/fusion PE diagnostic block to training logs: coordinate range, std, scene scale, `B_gauss` sigma, output mean/std.
- Add a low-risk PE ablation using fixed multi-scale Fourier frequencies instead of a random Gaussian matrix.
- Consider appending raw normalized coordinates and radius to the Fourier output:

```text
[fourier(coords_norm), coords_norm, ||coords_norm||]
```

- Consider one shared coordinate encoder or at least a shared frequency basis only after the scale-normalized baseline is stable. Do not refactor this blindly.

### Key/value asymmetry

Global compressor builds:

```text
k_base = k_proj(_get_layer_slice(pooled_features, 1))
v      = v_proj(pooled_features)
```

This is an intentional-looking asymmetric design: key is a single-layer retrieval view; value is full multi-layer memory content.

Rough points:

- The layer chosen for key is hard-coded.
- The indexing helper is ambiguous: it skips the first stored slice and currently selects `slice 2 / layer 12` for MapAnything memory.
- The checkpoint does not record which memory layer is used as key.
- The design question "which layer is best for key" is currently hidden inside code shape, not exposed as a controlled experiment.

Modification ideas:

- Add `lmc_key_slice_idx` with explicit zero-based semantics against the saved `layers_idx` list.
- Save `layers_idx`, `num_layers`, `key_slice_idx`, and `key_layer_label` in `lmc_config`.
- Add an assertion that the chosen key slice has exactly `input_dim`.
- Run key ablations in this order: `layer 12` current baseline, `layer 18`, `layer 6`, `final`, then learned scalar mix.
- Keep "all-layer concat key" as a later experiment, not the first fix. It increases parameters and makes key/value semantics less clean.

### Latent coordinate selection

Global mode selects `K` coordinates with FPS if memory has more than `K` points. Query tokens are initialized only from positional encoding of those coordinates, not from memory features.

Rough points:

- Random FPS start is the largest reproducibility issue in the global path.
- If `N <= K`, the code returns fewer than `K` tokens (`pooled_points[:, :K_curr, :]`). That is probably acceptable, but consumers and logs still talk as if K is fixed.
- There is no visibility, density, or feature-aware latent selection. This may be acceptable because global is currently the best-performing mode, but it should be documented as a deliberate baseline.

Modification ideas:

- Log actual `K_eff`.
- Add deterministic FPS.
- Only consider feature-aware or visibility-aware token selection after deterministic FPS is established as the baseline.

### Global attention and geometry bias

`GlobalSoftAttention` computes normal attention logits and adds a learned MLP bias from:

```text
log(dist_sq + 1e-6)
```

This is stronger than plain global attention because every latent query can attend to all memory points while still receiving a learned distance-dependent bias.

Rough points:

- Distance scale is raw scene-coordinate scale. If scene coordinates are normalized in some runs and raw-world in others, the geometry bias MLP sees different distributions.
- `log(dist_sq + 1e-6)` can span a large range. There is no clamp or normalization before `geo_bias_mlp`.
- No debug metric reports the relative magnitude of content logits versus geometry bias.
- The bias only sees scalar distance. It cannot model direction, anisotropic scene structure, token coverage radius, or memory density.
- Content attention and geometry bias are added with no learned gate. Early in training, a poorly scaled bias can dominate or vanish.
- The same distance feature is reused for every head except for the final MLP output. There is no explicit multi-scale distance basis.

Modification ideas:

- Log mean/std/min/max of `log(dist_sq + 1e-6)` for the first compressor forward.
- Add optional clamping or normalization only after inspecting those logs.
- Add a debug-only hook to compare `q @ k` logits scale and geo bias scale.
- Replace single `log(dist_sq)` input with scale-normalized distance features:

```text
dist_norm = sqrt(dist_sq) / scene_scale
geo_feat = [
  log1p(dist_norm),
  dist_norm,
  exp(-dist_norm^2 / sigma_1^2),
  exp(-dist_norm^2 / sigma_2^2),
  exp(-dist_norm^2 / sigma_3^2),
]
```

- Add a learnable per-head geometry gate:

```text
attn_logits = content_logits + alpha_head * geo_bias
```

Initialize `alpha_head` small so the model starts close to feature attention and learns how much geometry to use.

- Consider direction features only after the normalized distance-basis version is measured:

```text
delta = memory_point - latent_coord
direction = delta / (||delta|| + eps)
```

Direction may help, but it is a larger design change than distance normalization and gating.

### Scale token injection

`all_scale_tokens` are averaged over views, passed through `scale_mlp`, then added to every latent token.

Rough points:

- This is a global shift, not token-specific conditioning.
- It assumes mean-over-views is sufficient and that invalid/missing view tokens are already handled upstream.
- It is injected after attention blocks, so it cannot affect compressor attention routing in global mode.

Modification ideas:

- Document this as "global latent bias" rather than spatial/token alignment.
- Log whether scale tokens are present and their input/output norm.
- Do not make token-specific scale injection until an ablation justifies it.

## Fusion Design Notes

Implementation:

- `/home/xwh/project/ace_depth/ace_fusion.py:11`
- `/home/xwh/project/ace_depth/ace_fusion.py:47`
- `trainer_dinov2_lmc.py:1033`
- `trainer_dinov2_lmc.py:3061`

### Cross-attention structure

Fusion does:

```text
q = q_proj(query_feats)
k = k_proj(memory_z)
pe = PE(memory_p - scene_center)
v = v_proj(memory_z + pe_proj(pe))
```

Attention logits are based on image feature query and memory latent feature key. Geometry enters the value branch, not the matching score.

This is a coherent design: image features decide which latent content to read, while latent coordinates enrich what is read.

Rough points:

- If coordinates should help choose which memory token to attend to, the current fusion cannot express that directly.
- There is no attention mask, no temperature control, and no logging of attention entropy or token utilization.
- Dropout defaults to 0.1 and is active in S1 and ACE-G R2 S2. That is fine, but the stochasticity is not separated from FPS stochasticity, making ablations noisy.
- Query image features do not carry an explicit 2D coordinate or camera prior here, so fusion relies entirely on feature matching to choose memory tokens.
- Geometry is injected only into `v`. This means the model can use coordinates in the read content, but not in the read address.

Modification ideas:

- Add debug metrics for attention entropy and average max attention per batch.
- Add GeoMatch Fusion v1 after deterministic FPS and key-layer logging are in place:

```text
pe_mem = pe_proj(pe)
gate_k = sigmoid(key_geo_gate)
gate_v = sigmoid(value_geo_gate)
k = k_proj(memory_z + gate_k * pe_mem)
v = v_proj(memory_z + gate_v * pe_mem)
```

Use either separate PE projections for key/value or one shared projection in the first implementation. The important ablation is whether 3D location should affect memory-token matching, not only the value content.

Recommended gate initialization:

```text
key_geo_gate   = -4.0
value_geo_gate =  0.0
```

This starts close to current feature-only key matching while preserving a meaningful value-geometry path.

- Consider a small learnable attention temperature or logit scale for fusion if attention entropy shows collapse or uniform reads.
- Add `fusion_dropout` to `lmc_config` if it becomes configurable.

### Attention diagnostics and token usage

The GeoMatch design document's diagnostics are worth adopting before adding any regularization. Fusion should optionally expose enough attention statistics to answer whether memory tokens are being used as a broad latent set or whether a few tokens dominate most query rows.

Suggested diagnostics:

- `attention_entropy`: entropy over memory tokens, averaged over batch, heads, and query rows.
- `token_usage`: mean attention per memory token over batch, heads, and query rows.
- `effective_token_count`: `exp(entropy(token_usage))`.
- `avg_max_attention`: average top-1 attention weight.
- `gate_k` and `gate_v`: current geometry gate values.

Do not add a usage loss first. If diagnostics show collapse, test:

```text
L_usage = KL(token_usage || Uniform)
lambda_usage in {0.001, 0.003, 0.01}
```

Use this only in S1 initially. S2-G should not introduce new compressor-dependent objectives because the compressor output is cached and the compressor is frozen.

### Residual and normalization

Fusion uses:

```text
x = LayerNorm(residual + attention_out)
x = LayerNorm(x + FFN(x))
```

Rough points:

- This normalizes DINOv2 feature rows before the old ACE head sees them. That is a strong distribution change relative to vanilla DINO-ACE.
- S2 R2 continues changing this distribution while compressor output is cached. This is intended, but it raises the importance of `per_iter` ReproLoss scheduling.
- There is no direct diagnostic comparing raw feature norm and fused feature norm.
- Fusion strength is not explicitly controlled. `LayerNorm(residual + attention_out)` can turn fusion into a full feature reparameterization rather than a small memory-conditioned correction.

Modification ideas:

- Log raw/fused feature mean/std/norm in S1 and S2-G at low frequency.
- Track whether fusion is behaving as a small correction or a full feature reparameterization.
- Add a residual gate ablation:

```text
fused = norm(query_feats + gate * attention_out)
```

Initialize `gate` small or at 1.0 depending on whether the goal is stable fine-tuning or preserving current behavior. This is especially relevant for ACE-G R2, where fusion keeps changing during S2 while compressor output is cached.

### Batch expansion of single-scene memory

`_fuse_features()` expands a single memory output to the image batch size with `.expand()`.

Rough points:

- `.expand()` is memory efficient, but any future in-place mutation on expanded tensors would be dangerous.
- The pattern is duplicated in test-time inference.

Modification ideas:

- Add a small helper for expanding compressor outputs.
- Keep it read-only; do not clone unless a future operation requires writes.

## Deferred GeoMatch Research Tracks

These ideas from the GeoMatch design document are not discarded, but they should not be mixed into the first ACE-G global cleanup pass.

### Anchor-assisted residual branch

The idea:

```text
fused query feature + latent anchors
-> route to Top-M anchors
-> predict anchor-relative residual
-> recover world coordinate from anchor + residual
```

Why it is deferred:

- It adds new heads, routing logits, Top-M selection, delta regression, and another reprojection loss.
- It can interfere with the existing ACE head unless carefully weighted or detached.
- It changes the output contract more than fusion-key geometry does.

A later safe version should be auxiliary-only:

- Keep ACE head as the official output.
- Add anchor branch with small `lambda_anchor`, for example `0.05` or `0.1`.
- Compare `ACE-only`, `auxiliary`, `anchor-only`, `Top1`, and `Top3` before promoting it.

### Full normalized-coordinate training

The useful near-term piece is `scene_scale` for PE and geometry bias. Full target normalization is a separate system contract:

```text
P_norm = (P_world - mu_scene) / sigma_scene
P_world = P_norm * sigma_scene + mu_scene
```

Why it is deferred:

- It changes the target space and all world/reprojection recovery assumptions.
- It interacts with invalid-depth handling, camera-space conversion, depth thresholds, and checkpoint evaluation.
- It should first be proven equivalent on a single scene before multi-scene training.

### DSD-style local/deformable compressor

The idea is to move beyond FPS coverage tokens toward task-driven local/deformable tokens:

```text
FPS latent coords -> KNN local neighborhood -> local attention -> bounded offset refinement
```

Why it is deferred:

- It changes compressor output distribution and latent coordinates.
- S2-G caches compressor output and does not train compressor, so S1/S2 distribution drift becomes more dangerous.
- Deterministic FPS, key-layer ablations, and fusion diagnostics should be established first.

### Sub-memory routing

The idea is useful for large scenes:

```text
query descriptor + sub-memory descriptors -> top-1/top-2 memory selection -> ACE-G inference -> pose-level rerank
```

Why it is deferred:

- It solves a large-scene scaling problem, not the current global-mode module-design uncertainty.
- It needs separate evaluation for boundary queries and wrong-route failures.
- It should be introduced only after single-memory ACE-G global is stable and well-instrumented.

## Suggested Refactor Order

1. Deterministic FPS and config persistence.
2. Explicit key-layer config/logging, defaulting to current `slice 2 / layer 12`.
3. Key-layer ablations: `layer 12`, `layer 18`, `layer 6`, `final`, then learned scalar mix.
4. Shared coordinate scale contract for compressor PE, fusion PE, and global distance bias.
5. Shared reprojection/invalid-loss helper.
6. Explicit S1 sampled loss time-axis.
7. Module state restoration and explicit compressor `requires_grad` toggling.
8. Baseline diagnostics: PE scale, geometry-bias scale, fusion feature norms, attention entropy.
9. GeoMatch Fusion v1: gated geometry into fusion key/value, with current value-only fusion retained as the baseline.
10. Fusion attention diagnostics: token usage, effective token count, average max attention, gate curves.
11. Conditional usage-loss ablation only if diagnostics show token collapse.
12. Optional compressor/fusion ablations: fixed-frequency PE, normalized multi-distance geometry bias with a learnable gate, fusion residual gate, deterministic latent selection alternatives, token-specific scale injection.
13. Deferred research tracks: anchor auxiliary branch, full normalized-coordinate training, DSD-style local/deformable compressor, sub-memory routing.

The first four items are needed before global-mode experiments are easy to interpret. Items five through seven are engineering hygiene. Items eight through eleven are the near-term GeoMatch-informed research track. Items twelve and thirteen should be guarded by ablation logs and not folded into the baseline blindly.
