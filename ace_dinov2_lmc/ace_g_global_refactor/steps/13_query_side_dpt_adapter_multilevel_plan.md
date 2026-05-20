# Step 13 — Query-side DPT Adapter Multi-level Feature Plan

Date: 2026-05-19

Purpose: preserve and hand off the revised multi-level feature design after current B3-lite Levelwise Latent Merge failed. The central conclusion is that B3-lite does not falsify multi-level DINOv2 features for dense coordinate prediction; it more likely used hierarchy in the wrong location. Future multi-level work should move closer to DPT/FPN-style dense query feature fusion rather than replacing the memory compressor output.

This is a planning/design document only. Do not implement or launch training from this document without a separate implementation review.

---

## 1. Current evidence

Accepted reference remains FGPI-4090:

- Run: `4090_forceglobal_s1_periter`
- Contract: true global, no auto fallback, `s1_loss_step_mode=per_iter`, `lmc_key_slice_idx=2`, `lmc_fusion_geometry_mode=value_only_raw`, legacy profile
- Protocol: scene2a/c0_p4, memory `20260508_103423/memory_bse.pt`, 5-seed post-train median, hypotheses=256
- Metrics: pct5=83.66, pct2=34.24, med_t=2.6233, med_r=0.2831

Recent relevant negatives:

- B1 scalar_mix: negative; learned weights collapsed to layer12.
- Selected key-layer ablations key1/layer6 and key3/layer18: negative; layer12 remains the accepted key anchor.
- CPE-A1 / scene-scale PE: negative or mixed; do not rerun unchanged.
- B3-lite Levelwise Latent Merge:
  - main: pct5=77.43, pct2=24.90, med_r=0.31deg, med_t=2.96cm
  - diag: pct5=80.16, pct2=28.02, med_r=0.30deg, med_t=2.84cm
  - verdict: reject current B3-lite form; do not continue unchanged repeats or small B3 gate/shared-attention variants without a specific diagnostic bug.
- TP-mapany-flow-v1: pct5=74.71, pct2=26.46; reject and do not combine by default.

---

## 2. Main interpretation

B3-lite failure should not be interpreted as "multi-level DINOv2 features are useless." Dense prediction literature such as DPT-style depth heads and FPN-style decoders uses intermediate transformer/CNN features because dense regression requires both local spatial detail and global semantic/contextual stability.

The more likely conclusion is:

```text
B3-lite used hierarchy at the wrong point in the ACE-G pipeline.
```

B3-lite placed multi-level processing on the memory-compressor side:

```text
multi-layer memory features
  -> per-layer compressor into K latent tokens
  -> global level gate
  -> final compressed memory
```

DPT-style dense prediction instead uses multi-layer features on the dense image/query grid before the prediction head:

```text
query image multi-layer features
  -> per-layer projection
  -> spatial dense fusion / top-down refinement
  -> dense prediction head
```

For ACE-G, the prediction target is dense scene coordinates for query pixels/patches. Therefore the query-side dense representation may be a more natural place to exploit DINOv2 hierarchy than the memory latent bottleneck.

---

## 3. Why B3-lite may have failed

### 3.1 Multi-level information was compressed too early

B3-lite reduced each level into the same K=64 latent bottleneck before it influenced dense query prediction. Fine local cues from lower/intermediate layers may have been collapsed before they could help coordinate regression.

### 3.2 Memory-side hierarchy is not query-side hierarchy

DPT-like benefits usually come from multi-layer features of the current image/query. B3-lite mostly altered memory-bank representation. If the dense query feature remains too single-layer/semantic, memory-side hierarchy may not recover lost local precision.

### 3.3 Global level gate is too coarse

A global softmax over levels cannot express that some spatial regions/channels need low-level detail while others need high-level context. DPT/FPN fusion is spatial and channel-wise, not a single scene-level scalar mix.

### 3.4 It disturbed a stable layer12 anchor

The current stable contract uses selected layer12 as the key anchor and all-layer value concat as value. B3-lite replaced too much of that path. Future designs should preserve the layer12 anchor and add multi-level information as a residual or side adapter.

---

## 4. Design principles for next multi-level work

1. Preserve the accepted ACE-G memory contract by default:
   - true global
   - `s1_loss_step_mode=per_iter`
   - `lmc_key_slice_idx=2`
   - `lmc_fusion_geometry_mode=value_only_raw`
   - selected key + all-layer value concat
   - legacy profile

2. Move multi-level feature fusion closer to dense query prediction.

3. Add, do not replace. Use zero-gated residuals so the initial network is equivalent or near-equivalent to baseline.

4. Do not combine with Step12 GBR/FPE, TP-mapany-flow-v1, value_only_norm, geokey_norm, CPE scene-scale, or aux-ref expansion in the first test.

5. Keep the first implementation narrow enough to attribute results to one mechanism.

---

## 5. Recommended direction A: DPT-Q — Query-side DPT Adapter

### 5.1 Hypothesis

Multi-layer DINOv2 features are useful for ACE-G dense coordinate regression when fused on the query image grid before the ACE head/fusion path. This mirrors DPT/FPN usage more closely than B3-lite memory-latent merging.

### 5.2 Architecture

Conceptual flow:

```text
DINOv2 query intermediate layers, e.g. L6/L12/L18/L24
  -> per-layer projection to d_q
  -> reshape to patch grid
  -> DPT/FPN-style fusion block
  -> projected residual with output dim 1024
  -> query_feature_out = query_feature_layer12 + gate * dpt_residual
  -> existing LMCFeatureFusion / ACE head
```

The baseline query feature remains the anchor. The adapter is a residual branch initialized to zero or near zero:

```text
gate init = 0.0
```

### 5.3 First-version constraints

- Use fixed layer set first: `6 12 18 24` or the repository's existing layer-index convention equivalent.
- Use a small adapter width, e.g. 256.
- Output must be 1024-dim to preserve downstream interfaces.
- Do not alter memory compressor, memory file format, or fusion geometry in the first version.
- Do not change training profile.

### 5.4 Possible CLI fields

```bash
--query_feature_adapter legacy
--query_dpt_layers 6 12 18 24
--query_dpt_dim 256
--query_dpt_out_dim 1024
--query_dpt_fusion_mode topdown_add
--query_dpt_residual_gate_init 0.0
--query_dpt_norm layernorm
```

Default must remain:

```bash
--query_feature_adapter legacy
```

DPT-Q first test:

```bash
--query_feature_adapter dpt_residual
--query_dpt_layers 6 12 18 24
--query_dpt_dim 256
--query_dpt_out_dim 1024
--query_dpt_fusion_mode topdown_add
--query_dpt_residual_gate_init 0.0
```

### 5.5 Likely code touchpoints

The code execution agent must inspect exact names before editing. Likely areas:

- DINOv2 feature extraction path in training and testing.
  - Need intermediate layer outputs for query images.
  - Must ensure train/test use identical layer extraction.

- Buffer creation / raw buffer schema.
  - If current buffers store only one feature tensor, DPT-Q may need either:
    1. compute DPT-Q features before buffer storage and store final 1024-dim query feature only, or
    2. store multiple layer features in the buffer.
  - Prefer option 1 for first version to avoid large buffer/schema expansion.

- Fusion/head input path.
  - Insert adapter before existing LMCFeatureFusion or wherever the current query feature enters the fusion path.
  - Preserve legacy path exactly when adapter is disabled.

- Checkpoint/test loading.
  - Save adapter config and state dict.
  - Old checkpoints must load with `query_feature_adapter=legacy`.

### 5.6 Risks

- Multi-layer DINOv2 extraction may increase runtime/VRAM.
- If buffer stores only final features, the adapter must run during buffer fill; this may change what is cached and make debugging harder.
- If adapter output is too strong, it may destabilize a stable layer12 representation. Use zero gate and monitor gate value.
- DINOv2 intermediate layer indexing must be carefully matched to memory feature layer indexing; do not assume labels without inspecting existing extraction code.

---

## 6. Alternative direction B: MLR — Memory Late Residual

### 6.1 Hypothesis

Multi-level memory features may still help if they are added as a residual correction after the accepted baseline compressed memory, rather than replacing it.

### 6.2 Architecture

```text
baseline path:
  selected_key_concat_value -> GeoLMC -> Z_base

residual path:
  multi-level memory features -> lightweight projection/pooling/cross-attn -> Z_residual

merge:
  Z_out = Z_base + beta * Z_residual
```

with:

```text
beta init = 0.0
```

### 6.3 Difference from B3-lite

B3-lite replaced the compressor output with a gated merge of per-level latent outputs. MLR preserves the baseline compressed memory and only adds a late residual correction. This should be safer if layer12 is an important anchor.

### 6.4 Possible CLI fields

```bash
--lmc_multilevel_residual_mode off
--lmc_multilevel_residual_layers 6 12 18 24
--lmc_multilevel_residual_dim 256
--lmc_multilevel_residual_gate_init 0.0
--lmc_multilevel_residual_share_latents True
```

### 6.5 Priority

Lower priority than DPT-Q because it still operates mainly on the memory side, which B3-lite suggests may not be the best place for hierarchy.

---

## 7. Alternative direction C: DPT-MemTeacher — multi-level distillation

### 7.1 Hypothesis

Multi-level features may be useful as a training signal without increasing inference complexity. A frozen DPT-style multi-level teacher can guide the existing selected-layer ACE-G path during S1.

### 7.2 Architecture

```text
teacher = DPT-style fused multi-layer feature
student = current selected-layer ACE-G feature path
loss = original S1 loss + lambda * distill(student, teacher)
```

Inference remains unchanged.

### 7.3 Possible CLI fields

```bash
--s1_multilevel_distill_weight 0.0
--s1_multilevel_distill_layers 6 12 18 24
--s1_multilevel_distill_mode cosine
--s1_multilevel_teacher_mode dpt_fusion
```

First exploratory weight if used:

```bash
--s1_multilevel_distill_weight 0.05
```

### 7.4 Priority

Lower than DPT-Q. It is safer at inference but harder to design because the teacher target may not align with coordinate regression unless carefully normalized.

---

## 8. Recommended priority

Current priority for future multi-level work:

1. `DPT-Q`: query-side DPT adapter. Best aligned with dense prediction practice.
2. `MLR`: baseline memory compression plus zero-gated multi-level residual. Safer B3 successor if memory-side hierarchy is revisited.
3. `DPT-MemTeacher`: multi-level teacher/distillation. Useful if inference cost must remain unchanged.

Do not continue B3-lite variants such as shared-proj/no-shared-attn/token-gate/entropy as the next step unless diagnostics reveal a concrete implementation bug. The current metric drop is too large for small B3 gate variants to be the most efficient next direction.

---

## 9. First DPT-Q experiment contract

When ready to implement, the first DPT-Q run should preserve the FGPI-4090 contract and change exactly one primary mechanism:

```text
query_feature_adapter = dpt_residual
```

Keep:

```text
scene = /home/xwh/data/indoor6_ace/scene2a
memory = 20260508_103423/memory_bse.pt
train_preset = memory_compare_ace_g_v1
data_backend = ace
use_lmc = True
lmc_profile = legacy
lmc_mode = global
lmc_auto_mode_by_visibility = False
lmc_fps_start_policy = farthest_from_center
lmc_key_slice_idx = 2
s1_loss_step_mode = per_iter
lmc_feature_hierarchy_mode = selected_key_concat_value
lmc_fusion_geometry_mode = value_only_raw
batch_size = 10240
post_train_eval_seeds = 1305 2026 4242 7777 9001
post_train_hypotheses = 256
```

Do not combine with:

```text
GBR/FPE Step12 flags
mapany_flow_v1
B3-lite
value_only_norm/geokey_norm
CPE scene-scale
aux-ref
```

---

## 10. Promote / reject criteria

Compare against FGPI-4090:

```text
pct5 = 83.66
pct2 = 34.24
med_t = 2.6233
med_r = 0.2831
```

Promote / repeat if:

- `pct5 >= 84.2` and `pct2 >= 34.0`, with no med_t/med_r regression; or
- pct5 roughly tied but `pct2 >= 35.5`; or
- first run is `83.0 <= pct5 < 84.2` with stable pct2 and diagnostics show the residual gate learned a non-trivial value.

Reject if:

- `pct5 < 82.5`; or
- `pct2 < 33.0`; or
- residual gate remains effectively zero and metrics do not improve; or
- train/test runtime/memory cost becomes impractical relative to the gain.

---

## 11. Diagnostics to log if practical

For DPT-Q:

- final residual gate value
- per-layer adapter projection norm
- DPT residual norm relative to baseline query feature norm
- train/test runtime overhead
- buffer memory overhead if the buffer schema changes

For MLR:

- residual gate beta
- residual norm relative to `Z_base`
- attention entropy/effective token changes if residual uses attention

For distillation:

- distillation loss curve
- ratio of distill loss to original S1 loss
- teacher/student cosine similarity before and after training

---

## 12. Final recommendation

The next serious multi-level feature plan should be DPT-Q, not another B3-lite variant. The reason is architectural: dense prediction methods benefit from multi-level features because they fuse them on the dense query grid near the prediction head. ACE-G should test the same principle by adding a zero-gated query-side DPT adapter while preserving the accepted memory/compressor contract.

---

## 13. Literature-informed update: multi-level compression and fusion

Updated: 2026-05-20

This section adds a targeted literature-informed refinement after reviewing dense prediction and memory-token architectures. It does not replace the DPT-Q recommendation above; it clarifies why B3-lite failed and expands the design space for future multi-level fusion.

### 13.1 Relevant architecture patterns

#### DPT / Depth Anything V2

DPT-style dense prediction does not merely average or scalar-mix intermediate ViT layers. It uses a structured decoder:

```text
ViT / DINOv2 intermediate layers
  -> reassemble tokens into image-like feature maps
  -> per-level channel projection / neck
  -> progressive feature fusion / refinement blocks
  -> dense prediction head
```

Important DPT-style concepts to preserve in ACE-G design:

- explicit intermediate layer selection
- image-like dense feature maps
- per-level projection before fusion
- progressive fusion/refinement
- dense prediction head close to fused query features

Depth Anything V2 reinforces this pattern by using a DINOv2 encoder with a DPT decoder and intermediate features rather than relying only on the final transformer layer.

#### Perceiver IO

GeoLMC already resembles the input-compression half of Perceiver:

```text
large memory input -> cross-attention -> fixed latent array
```

But Perceiver IO also includes an output-query decoding stage:

```text
output queries -> cross-attend latent array -> structured/dense outputs
```

ACE-G currently compresses memory into latent tokens and then fuses them into image features, but it does not explicitly treat dense query patches as Perceiver-style output queries that decode from the memory latents. This suggests that fusion, not just compression, may be the missing architectural piece.

#### Mask2Former / pixel decoder separation

Mask2Former separates dense pixel feature construction from query decoding:

```text
multi-scale pixel decoder -> dense pixel features
transformer decoder queries -> attend/refine using pixel features
```

ACE-G should similarly separate:

```text
query dense feature refinement
memory latent conditioning
```

rather than forcing both into the memory compressor.

#### FPN / FaPN-style alignment

Pyramid fusion literature emphasizes that multi-level features must be aligned before fusion. Even though DINOv2 intermediate layers share token resolution, their semantic spaces differ. Future ACE-G multi-level fusion should include:

- per-layer normalization
- per-layer projection
- residual alignment
- zero-gated injection into the accepted baseline path

### 13.2 Updated interpretation of B3-lite failure

B3-lite failed not because multi-level DINOv2 features are useless, but because it missed several patterns that successful dense prediction architectures use.

B3-lite was primarily:

```text
multi-layer memory features
  -> per-layer latent compression
  -> global level gate
  -> replacement compressed memory
```

It lacked:

1. DPT-style query-grid dense decoder.
2. Perceiver IO-style output-query decoding from memory latents.
3. FPN/FaPN-style alignment and refinement before multi-level fusion.

Therefore, do not spend the next multi-level iteration on small B3-lite variants such as:

```text
shared projection
no shared attention
token gate
entropy gate
```

unless diagnostics reveal a concrete implementation bug. The larger issue is the location and form of hierarchy usage.

### 13.3 Updated direction A: DPT-Q / Pixel Adapter

This remains the highest-priority multi-level direction.

Architecture:

```text
DINOv2 query intermediate layers
  -> DPT-style reassemble/project/fuse
  -> dpt_query_feature
  -> query_feature_out = baseline_query_feature + alpha * Proj(dpt_query_feature)
  -> existing ACE-G fusion/head
```

Key constraints:

- preserve current memory compressor contract
- preserve layer12 selected-key anchor
- zero-initialize residual gate `alpha`
- output 1024-dim features so downstream interfaces remain stable
- do not combine with Step12 PE/bias changes in the first run

This direction is most directly supported by DPT / Depth Anything-style dense prediction practice.

### 13.4 New direction B: PIO-Fusion — Perceiver IO-style memory decoder

This direction focuses on the fusion module rather than the memory compressor.

Hypothesis:

GeoLMC may already compress memory adequately, but ACE-G lacks an explicit dense output-query decoder. Query patch tokens should decode from memory latents through cross-attention, similar to Perceiver IO.

Architecture:

```text
Z_mem = GeoLMC(memory)
Q_img = dense query image tokens

Q_mem = CrossAttention(
  query = Q_img,
  key   = Z_mem,
  value = Z_mem
)

Q_out = Q_img + beta * Q_mem
```

with:

```text
beta init = 0.0
```

Possible CLI fields:

```bash
--lmc_fusion_decoder_mode legacy
--lmc_fusion_decoder_layers 1
--lmc_fusion_decoder_heads 4
--lmc_fusion_decoder_dim 1024
--lmc_fusion_decoder_gate_init 0.0
```

First test:

```bash
--lmc_fusion_decoder_mode perceiver_io_residual
```

Keep all other FGPI-4090 contract fields unchanged.

Why this may be more effective than B3-lite:

- It keeps memory compression stable.
- It changes how dense query tokens read memory.
- It matches Perceiver IO's output-query decoding pattern.

Risks:

- Changes fusion behavior and query-token memory conditioning.
- Adds per-query-token cross-attention over K=64 memory tokens.
- Needs careful old checkpoint compatibility.

### 13.5 New direction C: Aligned Late Multi-level Residual

This is the safer B3 successor if memory-side or query-side multi-level residuals are still desired.

Architecture:

```text
baseline_feature = accepted ACE-G feature path
multi_level_residual = AlignFuse([L6, L12, L18, L24])
out = baseline_feature + gamma * multi_level_residual
```

Required alignment components:

- per-layer LayerNorm
- per-layer Linear projection
- shared output projection
- zero-gated residual

Possible CLI fields:

```bash
--multilevel_late_residual_mode off
--multilevel_late_residual_layers 6 12 18 24
--multilevel_late_residual_dim 256
--multilevel_late_residual_gate_init 0.0
--multilevel_late_residual_location pre_head
```

This direction is lower priority than DPT-Q and PIO-Fusion because it is less directly tied to a full dense decoder, but it is safer than replacing the compressor output as B3-lite did.

### 13.6 Updated priority

Recommended future multi-level order:

1. `DPT-Q / Pixel Adapter`: best aligned with DPT and Depth Anything dense prediction.
2. `PIO-Fusion / Perceiver IO-style memory decoder`: best aligned with latent-memory architectures and may fix the current fusion bottleneck.
3. `Aligned Late Multi-level Residual`: safest residual fallback if the first two are too invasive.

Do not continue B3-lite micro-variants as the next main direction.

### 13.7 First experiment contracts

All first tests should preserve the FGPI-4090 contract:

```text
scene = /home/xwh/data/indoor6_ace/scene2a
memory = 20260508_103423/memory_bse.pt
train_preset = memory_compare_ace_g_v1
data_backend = ace
use_lmc = True
lmc_profile = legacy
lmc_mode = global
lmc_auto_mode_by_visibility = False
lmc_fps_start_policy = farthest_from_center
lmc_key_slice_idx = 2
s1_loss_step_mode = per_iter
lmc_feature_hierarchy_mode = selected_key_concat_value
lmc_fusion_geometry_mode = value_only_raw
batch_size = 10240
post_train_eval_seeds = 1305 2026 4242 7777 9001
post_train_hypotheses = 256
```

Change exactly one primary mechanism per first run:

```text
DPT-Q:       --query_feature_adapter dpt_residual
PIO-Fusion:  --lmc_fusion_decoder_mode perceiver_io_residual
Late-MLR:    --multilevel_late_residual_mode aligned_residual
```

Do not combine these with:

```text
B3-lite
Step12 FPE/GBR/PointRoPE/CRPB
mapany_flow_v1
value_only_norm/geokey_norm
CPE scene-scale
aux-ref expansion
```

### 13.8 Sources consulted

- DPT: Vision Transformers for Dense Prediction — https://arxiv.org/abs/2103.13413
- Hugging Face DPT documentation — https://huggingface.co/docs/transformers/en/model_doc/dpt
- Depth Anything V2 architecture overview — https://deepwiki.com/DepthAnything/Depth-Anything-V2/2-model-architecture/
- Perceiver IO — https://arxiv.org/abs/2107.14795
- Hugging Face Perceiver documentation — https://huggingface.co/docs/transformers/v4.14.1/en/model_doc/perceiver
- Mask2Former — https://arxiv.org/abs/2112.01527
- Hugging Face Mask2Former documentation — https://huggingface.co/docs/transformers/v4.52.3/model_doc/mask2former
- MPViT — https://arxiv.org/abs/2112.11010
- FaPN — https://arxiv.org/abs/2108.07058
