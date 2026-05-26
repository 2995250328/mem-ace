# Step 17: Conservative Plan for Using View Tokens Stored in Memory

## 1. Purpose

This note records a conservative plan for using intermediate view-level tokens already stored inside ACE-DINOv2-LMC memory files.

In the current codebase, the relevant field is not an explicit `camera_token` key but `all_scale_tokens`, which is currently the closest equivalent to per-view global tokens. These tokens are stored per memory view and originate from a CLS-style global token produced during memory extraction.

The key question is not whether these tokens exist, but how they should be used without violating the accepted ACE-G contract, destabilizing the compressor/fusion stack, or mixing weak global semantics with dense metric geometry.

This note therefore separates:

1. what exists today,
2. what is already used,
3. what is safe to try next,
4. what should not be done yet.

## 2. Current state in code

The current token-like memory field is:

```text
all_scale_tokens: [M, D]
```

where:

- `M` = number of memory views,
- `D` = token dimension.

The current system behavior is conservative:

- memory extraction saves `all_scale_tokens`,
- memory loading preserves it,
- trainer/test pass it into `memory_dict`,
- GeoLMC can consume it through a lightweight scale-token injection path,
- fusion does not consume raw view tokens directly,
- query-side CLS tokens are not passed through the current training/test fusion API.

The existing use is weak and summary-based:

```text
all_scale_tokens [M, D]
-> mean over views
-> small MLP / scale-token injection
-> added to latent features
```

This means the codebase already supports a minimal memory-side use of view tokens, but not a full token-aware fusion design.

## 3. Interpretation of these tokens

These stored tokens should currently be treated as **view-level global descriptors**, not as dense coordinate evidence and not as reliable metric pose tokens.

They may contain:

- global appearance cues,
- scene-level semantic context,
- coarse view/layout information,
- weak viewpoint or scale information,
- reference-conditioned context inherited from the MapAnything forward.

They should not be assumed to contain:

- exact camera pose,
- direct metric 3D constraints,
- stable cross-reference-frame invariance,
- a reliable replacement for geometric memory points.

Therefore their best role is as a **side-channel signal** for:

- memory-side conditioning,
- token pooling,
- latent modulation,
- memory routing,
- optional calibration or gating.

## 4. Main recommendation

Do not insert raw `all_scale_tokens` directly into dense fusion or query features as a first experiment.

Instead, use a staged plan:

1. diagnostics first,
2. low-risk memory-side summary use,
3. optional latent modulation,
4. later multi-memory routing,
5. defer query-side CLS/view-token conditioning.

The baseline ACE-G contract remains unchanged:

- preserve the accepted C0/FGPI-style path,
- keep current training/test/checkpoint compatibility,
- require all token-related additions to be default-off,
- require full fallback if a memory file does not contain `all_scale_tokens`.

## 5. Stage 0: diagnostics before training changes

Before introducing any new learned use, inspect whether the stored view tokens actually contain stable information.

Recommended diagnostics:

```text
shape / dtype / device
per-dimension mean and std
token norm distribution
pairwise cosine similarity distribution
PCA / t-SNE / UMAP projections
token similarity vs camera-center distance
token similarity vs viewing-direction difference
token similarity vs covisibility / overlap if available
token clustering vs spatial scene region
```

The goal is to answer:

```text
Do nearby or overlapping views have token similarity structure that the model could exploit?
```

If the answer is weak or inconsistent, the tokens should remain auxiliary metadata only.

## 6. Stage 1: safer replacement for current mean pooling

The lowest-risk next use is to improve the current summary operation:

```text
summary = mean(all_scale_tokens)
```

Replace or augment it with optional view-token pooling:

```text
summary = ViewTokenPool(all_scale_tokens)
```

Recommended first variant:

```text
learnable query
-> cross-attention over all_scale_tokens
-> pooled summary
-> residual blend with mean summary
```

Example conceptual form:

```text
summary_mean = mean(all_scale_tokens)
summary_attn = CrossAttention(q_learned, all_scale_tokens)
summary = summary_mean + gamma * proj(summary_attn)
```

with `gamma=0` or near-zero initialization.

This is attractive because it:

- stays entirely on the memory side,
- preserves the current feature buffer and query path,
- modifies only the current token-summary logic,
- remains easy to disable and compare.

## 7. Stage 2: latent modulation inside GeoLMC

A stronger but still conservative option is to use the pooled view-token summary to modulate latent features.

Conceptually:

```text
summary = ViewTokenPool(all_scale_tokens)
scale, shift = MLP(summary)
latent = latent * (1 + gamma * scale) + gamma * shift
```

or:

```text
latent = latent + gamma * latent_bias(summary)
```

This is preferable to raw token injection into fusion because it:

- remains inside the compressor side,
- does not require changing the public fusion forward signature,
- avoids mixing view tokens with dense query patch tokens,
- is compatible with the existing notion that `all_scale_tokens` is a memory-side side signal.

Design constraints:

- initialize to identity or zero residual,
- keep the feature dimension unchanged,
- require explicit config flags in `lmc_config`,
- fall back cleanly when `all_scale_tokens` is absent.

## 8. Stage 3: multi-memory routing descriptor

If the system later moves toward multiple local memories or large-scene memory routing, `all_scale_tokens` can become a memory-level descriptor.

For each memory bundle, compute:

```text
memory_descriptor = pool(all_scale_tokens)
```

Potential uses:

- memory clustering,
- memory similarity analysis,
- top-k memory preselection,
- multi-memory routing scores,
- scene-chunk retrieval.

Important limitation:

Current training/test paths do not pass query-side CLS tokens. Therefore the first routing use should not claim true query-adaptive token matching yet.

Safer early uses are:

- memory-vs-memory organization,
- local-map grouping,
- routing analysis with offline oracle support,
- later integration once a stable query global descriptor exists in the test path.

## 9. Optional later extension: camera-pose-conditioned view tokens

A more ambitious direction is to combine each stored view token with explicit camera metadata.

For each memory view:

```text
view_token_i
+ pose_encoding(camera_center_i, rotation_i, viewing_direction_i)
+ reference-frame metadata
```

This could be used to create a richer per-view descriptor for:

- geometry-aware token pooling,
- routing,
- memory-view scoring,
- attention bias.

However this should remain a later-stage option because it raises several risks:

- frame convention mistakes,
- accidental coupling to reference-conditioned coordinates,
- overclaiming metric meaning from weak global descriptors,
- larger metadata contracts.

If attempted, it should first appear only as an attention bias or side descriptor, not as a replacement for geometric memory points.

## 10. What not to do yet

The following should be avoided in the first iteration:

1. **Do not feed raw `all_scale_tokens` directly into dense fusion.**
   They are view-level global tokens, not patch-wise query features.

2. **Do not alter the accepted C0/FGPI baseline by default.**
   All new token logic must be opt-in.

3. **Do not assume they are exact camera pose tokens.**
   They are at best weak camera/view-conditioned global descriptors.

4. **Do not add query CLS guidance yet.**
   Current trainer/test/ensemble paths do not pass query CLS tokens into fusion.

5. **Do not use them as direct metric geometry supervision.**
   They should modulate or summarize, not replace point-based geometry.

6. **Do not combine this direction immediately with PointRoPE, multi-level fusion, or multi-memory routing changes.**
   Isolate the mechanism first.

## 11. Experiment plan

### Stage A: offline analysis only

Goal:

- determine whether `all_scale_tokens` contains stable and scene-relevant structure.

Deliverables:

- token statistics report,
- token-to-pose correlation report,
- PCA/t-SNE plots,
- recommendation whether to proceed.

### Stage B: dry-run forward compatibility

Goal:

- verify optional pooling/modulation modules run without changing training semantics.

Checks:

- shape,
- dtype,
- NaN/Inf,
- device placement,
- latency,
- checkpoint save/load compatibility.

### Stage C: pooled-summary ablation

Compare:

```text
none
mean
attention_pool
```

Metrics:

- training stability,
- S1 alignment loss,
- S2 reprojection loss,
- median translation/rotation,
- `pct5`,
- runtime cost.

### Stage D: latent modulation ablation

Compare:

```text
baseline
latent_bias
latent_film
```

Again keep all runs isolated from unrelated geometry/fusion changes.

### Stage E: later routing use

Only after the above is stable:

- use pooled token descriptors for memory grouping/routing,
- compare against memory routing baselines,
- treat query-adaptive token routing as a future extension.

## 12. Documentation and configuration guidance

This direction should remain outside the current Step15 mainline, which is already focused on PointRoPE, fusion-side internal refinement, and compression-side residual multi-level changes.

It also should not replace the main Step16 large-scene multi-memory plan.

Therefore the correct place is a separate note like this one.

If implementation starts later, expected config fields should be explicit, for example:

```text
use_view_tokens
view_token_mode = none | mean | attn_pool | latent_bias | latent_film
view_token_gamma_init
view_token_pose_bias = false | true
```

All of them should be checkpointed in `lmc_config` and default to the current no-change behavior.

## 13. Final recommendation

Treat `all_scale_tokens` as a promising but currently underused memory-side side channel.

The correct first move is not to route them into dense fusion, but to:

1. measure what information they contain,
2. replace the current mean pooling with a safer optional learned pooling,
3. optionally modulate GeoLMC latent features,
4. later reuse the same summary for multi-memory routing,
5. defer query-side CLS/view-token guidance until the training/test API supports it cleanly.

This keeps the ACE-G baseline intact while still opening a low-risk path for exploiting stored global view tokens.
