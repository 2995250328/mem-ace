# Step 14 — PointRoPE and Continuous Relative Bias Plan

Date: 2026-05-19

Purpose: preserve the literature-informed follow-up design for replacing crude Fourier positional encoding and improving geometric distance bias in ACE-G. This document builds on Step 12 (`GeoBias-RBF` and `Fourier PE v2`) and Step 13 (`DPT-Q` multi-level query-side features). It should be used only after the current Step 12 main/control runs are interpreted.

This is a planning document only. Do not implement or launch training from this document without a separate code execution review.

---

## 1. Motivation

The Step 12 work separates two geometry-conditioning hypotheses:

```text
FPE / Fourier PE:
  absolute coordinate representation for compressor positional features

GBR / distance bias:
  relative latent-memory attention prior in compressor attention logits
```

The user raised a valid concern that the current Fourier encoding may be crude and that both positional encoding and distance bias should borrow more from established open work rather than relying only on hand-designed variants.

Literature scan suggests two natural successors:

1. `PointRoPE-main`: axis-wise 3D RoPE / Point-RoPE for compressor q/k positional modulation.
2. `CRPB-main`: continuous relative position bias over `[dx, dy, dz, distance, log-distance]` for compressor attention logits.

These should not replace the already-running Step 12 experiments. They are next-round candidates after `FPE-main`, `GBR-main`, `S12-legacy-control`, and targeted sensitivity runs are interpreted.

---

## 2. Relevant external ideas

### 2.1 RoPE for vision transformers

RoPE rotates query/key features as a function of position, so attention dot products naturally encode relative offsets. RoPE-ViT shows that RoPE is practical in vision transformer settings and can improve extrapolation to different image resolutions with low overhead.

Implication for ACE-G:

- RoPE is more naturally applied to q/k inside attention than as an extra absolute feature appended to tokens.
- For GeoLMC compressor, the relevant attention is latent-to-memory cross-attention, so q/k rotary modulation is a plausible replacement or complement to absolute Fourier PE.

### 2.2 Point-RoPE / axis-wise 3D RoPE

LitePT adapts RoPE to 3D point coordinates using Point-RoPE:

```text
feature f_i at point p_i=(x_i,y_i,z_i)
feature channels split into x/y/z subspaces
apply 1D RoPE to each subspace using x_i, y_i, z_i separately
concatenate rotated subspaces
```

Conceptually:

```text
f_i = [f_i^x ; f_i^y ; f_i^z]
PointRoPE(f_i, xyz_i) = concat(
  RoPE_1D(f_i^x, x_i),
  RoPE_1D(f_i^y, y_i),
  RoPE_1D(f_i^z, z_i)
)
```

Implication for ACE-G:

- Use centered/scaled metric coordinates, not raw token indices.
- Apply to compressor q/k, not value, in the first version.
- Treat as a separate mechanism from Fourier PE v2 and RBF bias.

### 2.3 Point Transformer / 3D relative position encoding

Point Transformer-style attention uses relative position information between point pairs, often via an MLP over coordinate differences, to affect attention weights and sometimes value aggregation.

Simplified idea:

```text
delta_ij = p_i - p_j
relative_encoding = MLP(delta_ij)
attention depends on q_i, k_j, and relative_encoding
```

Implication for ACE-G:

- A stronger successor to simple distance-only RBF bias is a learnable continuous relative position bias using both direction and distance.
- First version should only add an attention-logit bias, not alter values.

### 2.4 RBF distance encodings in 3D attention

Molecular and geometric transformers commonly use Euclidean distances expanded by radial basis functions to create relative attention features. This supports Step 12 `GBR-RBF` as a literature-aligned first step.

Implication for ACE-G:

- `GBR-RBF` is not arbitrary; it is a valid simple radial relative bias.
- If it is weak, the likely limitation is that it is direction-agnostic, not necessarily that distance bias is useless.

---

## 3. Candidate A: PointRoPE-main

### 3.1 Hypothesis

The current Fourier positional encoding may be suboptimal because it injects absolute coordinate features, while compressor cross-attention mainly needs relative geometry between latent coordinates and memory coordinates. Axis-wise 3D Point-RoPE applied to q/k may provide a cleaner, parameter-free relative geometry signal.

### 3.2 Architecture

Apply Point-RoPE inside GeoLMC compressor attention:

```text
q = Wq(latent_feature)
k = Wk(memory_feature)

latent_xyz_norm = normalize(latent_xyz, scene_center, radius)
memory_xyz_norm = normalize(memory_xyz, scene_center, radius)

q_rope = PointRoPE(q, latent_xyz_norm)
k_rope = PointRoPE(k, memory_xyz_norm)

attention_logits = q_rope @ k_rope^T / sqrt(d) + existing_geo_bias
```

First version:

- Apply only to q/k.
- Do not apply to value.
- Do not combine with `GBR-RBF` or `CRPB` initially.
- Keep existing memory/fusion contracts.

### 3.3 Coordinate normalization

Do not feed raw meter coordinates directly into RoPE. Use centered/scaled coordinates:

```text
xyz_norm = (xyz - scene_center) / radius
```

First radius:

```text
radius = 4.0
```

A later sensitivity can test `radius=2.0`, but do not launch a grid before the main result.

### 3.4 Proposed CLI/config fields

```bash
--lmc_pos_encoding_mode fourier_legacy
--lmc_point_rope_coord_norm scene_radius
--lmc_point_rope_radius 4.0
--lmc_point_rope_base 10000
--lmc_point_rope_axes xyz_split
--lmc_point_rope_apply_to qk
```

First PointRoPE run:

```bash
--lmc_pos_encoding_mode point_rope
--lmc_point_rope_coord_norm scene_radius
--lmc_point_rope_radius 4.0
--lmc_point_rope_base 10000
--lmc_point_rope_axes xyz_split
--lmc_point_rope_apply_to qk
--lmc_geo_bias_mode legacy
```

Checkpoint config should include all PointRoPE fields. Old checkpoints should default to `fourier_legacy`.

### 3.5 Implementation touchpoints

Likely files:

- `ace_compressor.py`
  - Implement axis-wise 3D Point-RoPE helper.
  - Insert into q/k computation in compressor attention.
  - Ensure feature dimension can be split across axes and pairwise RoPE dimensions. If not divisible cleanly, project to a compatible q/k dim or leave a remainder unrotated.

- `ace_dinov2_lmc/options_dinov2_lmc.py`
  - Add CLI fields.

- `ace_dinov2_lmc/trainer_dinov2_lmc.py`
  - Pass fields into GeoLMC.
  - Save to `lmc_config`.

- `ace_dinov2_lmc/test_ace_dinov2_lmc.py`
  - Restore PointRoPE config from checkpoints.
  - Preserve old checkpoint compatibility.

### 3.6 Risks

- RoPE frequency scale may still be wrong if coordinate normalization radius is wrong.
- Axis-wise split assumes x/y/z have comparable meaning after normalization; indoor scenes may have anisotropic ranges.
- Applying RoPE on top of an existing geo bias may over-condition geometry. First test should keep `lmc_geo_bias_mode=legacy` and no extra RBF/CRPB.
- Query/key dimensions must satisfy RoPE even-pair requirements.

---

## 4. Candidate B: CRPB-main — Continuous Relative Position Bias

### 4.1 Hypothesis

RBF distance bias may be too weak because it only knows distance, not direction. A small continuous relative position bias MLP over normalized coordinate deltas can encode both direction and distance while remaining simpler than full Point Transformer relative value encoding.

### 4.2 Architecture

For latent coordinate `p_i` and memory coordinate `p_j`:

```text
delta_ij = (p_j - p_i) / radius
dist_ij = ||delta_ij||
input_ij = [dx, dy, dz, dist, log(dist + eps)]
bias_ij = MLP(input_ij)
attention_logits += bias_ij
```

First version:

- Bias only; do not alter values.
- One scalar bias shared across heads, or optionally per-head in later ablation.
- Zero-initialize last layer so initial behavior is legacy.

### 4.3 Proposed CLI/config fields

```bash
--lmc_geo_bias_mode legacy
--lmc_geo_bias_crpb_dim 32
--lmc_geo_bias_crpb_input delta_dist_log
--lmc_geo_bias_crpb_radius 4.0
--lmc_geo_bias_crpb_per_head False
--lmc_geo_bias_crpb_zero_init True
```

First CRPB run:

```bash
--lmc_geo_bias_mode crpb
--lmc_geo_bias_crpb_dim 32
--lmc_geo_bias_crpb_input delta_dist_log
--lmc_geo_bias_crpb_radius 4.0
--lmc_geo_bias_crpb_per_head False
--lmc_geo_bias_crpb_zero_init True
--lmc_pos_encoding_mode fourier_legacy
```

Checkpoint config should include all CRPB fields. Old checkpoints should default to `legacy`.

### 4.4 Implementation touchpoints

Likely files:

- `ace_compressor.py`
  - Add small MLP for continuous relative position bias.
  - Compute normalized pairwise deltas between latent and memory coordinates.
  - Add output to attention logits at the same point as existing geometry bias.

- `ace_dinov2_lmc/options_dinov2_lmc.py`
  - Add CLI fields.

- `ace_dinov2_lmc/trainer_dinov2_lmc.py`
  - Pass fields and save config.

- `ace_dinov2_lmc/test_ace_dinov2_lmc.py`
  - Restore config and preserve old checkpoints.

### 4.5 Risks

- CRPB is more expressive than RBF and can overfit scene2a.
- Pairwise MLP over all latent-memory pairs may add runtime overhead, though K=64 makes it likely manageable.
- Coordinate radius normalization is still critical.
- If zero-initialized, learning may be slow; but this is safer for first test.

---

## 5. Relationship to Step 12 runs

Current / near-term Step 12 should remain:

1. `GBR-main`: RBF residual distance bias.
2. `FPE-main`: Fourier PE v2.
3. `S12-legacy-control`: default path compatibility.
4. `FPE-radius2`: radius sensitivity if useful.

Do not replace or reinterpret these runs mid-flight.

Next-round decision logic:

| Step 12 outcome | Next candidate |
| --- | --- |
| FPE-main/FPE-radius2 negative | Try `PointRoPE-main` because Fourier injection may be the wrong form |
| FPE positive | Repeat FPE before trying PointRoPE |
| GBR-RBF negative but not catastrophic | Try `CRPB-main` because direction-aware relative bias may be needed |
| GBR-RBF positive | Repeat GBR before trying CRPB |
| S12-legacy-control drifts below baseline | Fix default-path compatibility before any PointRoPE/CRPB conclusions |

---

## 6. First experiment contracts

### 6.1 PointRoPE-main contract

Preserve FGPI-4090 contract:

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

Only core change:

```text
lmc_pos_encoding_mode = point_rope
```

Keep:

```text
lmc_geo_bias_mode = legacy
```

### 6.2 CRPB-main contract

Same FGPI-4090 contract.

Only core change:

```text
lmc_geo_bias_mode = crpb
```

Keep:

```text
lmc_pos_encoding_mode = fourier_legacy
```

---

## 7. Promote / reject criteria

Compare with FGPI-4090:

```text
pct5 = 83.66
pct2 = 34.24
med_t = 2.6233
med_r = 0.2831
```

Promote/repeat if:

- `pct5 >= 84.2` and `pct2 >= 34.0`, no med_t/med_r regression; or
- pct5 roughly tied but `pct2 >= 35.5`; or
- `83.0 <= pct5 < 84.2` with stable pct2 and diagnostics indicate the new PE/bias has non-trivial effect.

Reject if:

- `pct5 < 82.5`; or
- `pct2 < 33.0`; or
- default-path compatibility is not established; or
- added module remains effectively inactive and metrics do not improve.

---

## 8. Diagnostics to record

For PointRoPE:

- coordinate normalization radius and coordinate range after normalization
- q/k norm before and after RoPE if practical
- attention entropy/effective token count compared with legacy
- runtime overhead

For CRPB:

- CRPB output mean/std/min/max
- attention entropy/effective token count compared with legacy
- learned first/last layer norm or final bias distribution
- runtime overhead

For both:

- exact command
- config snapshot
- checkpoint path
- 5-seed post-train eval summary
- old checkpoint compatibility result

---

## 9. What not to do

Do not run first-round combinations:

```text
PointRoPE + GBR
PointRoPE + CRPB
PointRoPE + FPE
CRPB + FPE
CRPB + GBR
PointRoPE/CRPB + B3-lite
PointRoPE/CRPB + mapany_flow_v1
```

Do not launch broad grids over radius, frequency base, per-head bias, or learnable scales until a main single-variable run shows a signal.

---

## 10. Sources consulted

- LitePT / Point-RoPE: https://litept.github.io/
- RoPE-ViT: https://huggingface.co/papers/2403.13298
- RoPE-ViT ECCV PDF: https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/01584.pdf
- Point Transformer: https://openaccess.thecvf.com/content/ICCV2021/papers/Zhao_Point_Transformer_ICCV_2021_paper.pdf
- Stratified Transformer: https://openaccess.thecvf.com/content/CVPR2022/papers/Lai_Stratified_Transformer_for_3D_Point_Cloud_Segmentation_CVPR_2022_paper.pdf
- Relative molecule self-attention transformer: https://pmc.ncbi.nlm.nih.gov/articles/PMC10765783/
- Rethinking and Improving Relative Position Encoding for Vision Transformer: https://openaccess.thecvf.com/content/ICCV2021/papers/Wu_Rethinking_and_Improving_Relative_Position_Encoding_for_Vision_Transformer_ICCV_2021_paper.pdf
- DPT paper: https://arxiv.org/abs/2103.13413
- Hugging Face DPT docs: https://huggingface.co/docs/transformers/v4.43.0/en/model_doc/dpt
- Depth Anything V2 architecture overview: https://deepwiki.com/DepthAnything/Depth-Anything-V2/2-model-architecture/

---

## 11. Final recommendation

Do not treat Fourier PE and distance bias as solved by hand-designed Step 12 alone. Keep Step 12 as the first controlled test, but if it is weak, use literature-aligned successors:

1. `PointRoPE-main` for Fourier replacement / q-k geometric rotary modulation.
2. `CRPB-main` for stronger direction-aware relative distance bias.

Both should preserve the accepted FGPI-4090 contract and be tested independently.
