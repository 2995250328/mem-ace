# Step 12 — GeoBias-RBF and Fourier PE v2 Spare-3090 Plan

Date: 2026-05-19

Purpose: provide a handoff-ready implementation and experiment plan for two independent ACE-G geometry/compressor experiments that can run on the two spare 3090 GPUs while B3-lite and TP-mapany-flow-v1 are already running.

This is a planning document only. Do not launch training from this document until the code execution agent has implemented the flags, verified default compatibility, and confirmed the target 3090 GPU IDs.

---

## 1. Current context

Accepted reference:

- Run family: FGPI-4090 / `4090_forceglobal_s1_periter`
- Scene/protocol: indoor6 scene2a / c0_p4, 5-run post-train median, hypotheses=256
- Memory: `20260508_103423/memory_bse.pt`
- Contract: true global, no auto fallback, `s1_loss_step_mode=per_iter`, `lmc_key_slice_idx=2`, `lmc_fusion_geometry_mode=value_only_raw`
- Metrics: pct5=83.66, pct2=34.24, med_t=2.6233, med_r=0.2831

Already done / do not rerun unchanged:

- A0/A1/A2
- CPE-A1 / compressor PE scene-scale variant
- current B1 scalar_mix, including 3090 repeat, because weights collapse to layer 12
- selected key-layer ablations key1/layer6 and key3/layer18
- geokey_norm / value_only_norm as standalone default directions
- C1 aux-ref / alpha scaling without a new paired repeated-eval win

Currently running elsewhere:

- B3-lite Levelwise Latent Merge main/diag on local GPUs
- TP-mapany-flow-v1 profile validation on local GPU

Therefore this plan focuses on two independent code-change experiments for the two spare 3090 GPUs:

1. `GBR-main`: residual multi-scale RBF geometry bias in GeoLMC attention logits.
2. `FPE-main`: Fourier positional encoding v2 for compressor coordinate encoding.

The two experiments must be implemented and run separately. Do not combine `GBR` and `FPE` in the first round.

---

## 2. Experiment A: GBR-main — residual RBF geometry attention bias

### 2.1 Hypothesis

The existing GeoLMC distance/geometry bias may be too rigid for scene2a-scale memory aggregation. A learnable multi-scale radial basis residual in the cross-attention logits may help latent tokens attend to memory points at appropriate local/mid/far spatial radii without changing the memory feature hierarchy or value feature distribution.

This differs from rejected CPE-A1 because it changes the relative attention bias shape, not the absolute compressor positional encoding normalization and not `value_only_norm`.

### 2.2 Architecture contract

Current conceptual attention:

```text
attention_logits = qk / sqrt(d) + existing_geo_bias
```

GBR adds a zero-initialized residual term:

```text
attention_logits = qk / sqrt(d) + existing_geo_bias + residual_rbf_distance_bias

residual_rbf_distance_bias = alpha * sum_j softmax(w)_j * exp(-dist^2 / (2 * sigma_j^2))
```

Initial behavior should be identical or near-identical to legacy:

- `lmc_geo_bias_mode=legacy` by default.
- `lmc_geo_bias_rbf_alpha_init=0.0` for `rbf_residual` first test.
- The new branch is only active when `--lmc_geo_bias_mode rbf_residual`.

Recommended first scales in meters:

```text
0.25, 0.5, 1.0, 2.0, 4.0
```

### 2.3 Code touchpoints

Likely files:

- `ace_compressor.py`
  - Add GeoLMC constructor arguments for the new bias mode and RBF configuration.
  - Implement a helper that takes latent coordinates and memory coordinates and returns a broadcastable attention-bias tensor.
  - Add the residual bias only at the same point where existing geometry/distance bias is added to attention logits.
  - Keep legacy path byte-behavior compatible when mode is `legacy`.

- `ace_dinov2_lmc/options_dinov2_lmc.py`
  - Add CLI fields listed below.

- `ace_dinov2_lmc/trainer_dinov2_lmc.py`
  - Pass parsed CLI fields into GeoLMC construction.
  - Save fields into `lmc_config`.
  - If runtime stats already support compressor internals, optionally log RBF alpha/weights/effect stats.

- `ace_dinov2_lmc/test_ace_dinov2_lmc.py`
  - Restore GeoLMC with new config fields when present.
  - Default missing fields to legacy for old checkpoints.

### 2.4 CLI/config fields

Add fields with these defaults:

```bash
--lmc_geo_bias_mode legacy
--lmc_geo_bias_rbf_scales 0.25 0.5 1.0 2.0 4.0
--lmc_geo_bias_rbf_alpha_init 0.0
--lmc_geo_bias_rbf_learn_weights True
--lmc_geo_bias_rbf_per_head False
```

Checkpoint `lmc_config` should include:

```text
geo_bias_mode
geo_bias_rbf_scales
geo_bias_rbf_alpha_init
geo_bias_rbf_learn_weights
geo_bias_rbf_per_head
```

If practical, also save final learned values in checkpoint metadata or best metadata:

```text
final_geo_bias_rbf_alpha
final_geo_bias_rbf_weights
```

---

## 3. Experiment B: FPE-main — Fourier positional encoding v2

### 3.1 Hypothesis

The current Fourier coordinate encoding in the compressor is likely crude: it may use fixed frequencies and raw meter coordinates in a way that is poorly matched to scene2a scale. A cleaner normalized Fourier PE with a residual gate may improve memory/latent coordinate conditioning without changing attention bias, hierarchy, fusion geometry, or training profile.

This is not CPE-A1. CPE-A1 changed compressor PE scene-scale/value normalization recipe and was negative. FPE-main should isolate the Fourier coordinate encoding only while keeping `value_only_raw` and selected-key/default value contract.

### 3.2 Architecture contract

New mode:

```text
coordinate xyz
  -> subtract scene_center
  -> divide by fixed radius
  -> clamp or leave normalized coordinates according to implementation safety
  -> sin/cos multi-frequency Fourier features
  -> small projection to model positional dimension
  -> gated residual into the existing positional branch
```

First-round design choices:

- Fixed frequencies, not learnable.
- Fixed radius, not scene-adaptive.
- Residual gate initialized to 0.0 to protect the legacy initialization.
- Do not combine with `GBR` in first round.
- Do not enable `value_only_norm`, `geokey_norm`, CPE scene-scale, or query-coordinate GeoMatch.

### 3.3 Code touchpoints

Likely files:

- `ace_compressor.py`
  - Find the current Fourier / coordinate positional encoding implementation inside GeoLMC.
  - Add a mode switch: legacy Fourier vs `fourier_v2`.
  - Implement normalized Fourier PE helper.
  - Add a residual gate parameter for the v2 branch.
  - Ensure mode `fourier_legacy` exactly preserves old behavior.

- `ace_dinov2_lmc/options_dinov2_lmc.py`
  - Add CLI fields listed below.

- `ace_dinov2_lmc/trainer_dinov2_lmc.py`
  - Pass parsed CLI fields into GeoLMC construction.
  - Save fields into `lmc_config`.

- `ace_dinov2_lmc/test_ace_dinov2_lmc.py`
  - Restore new PE config for new checkpoints.
  - Default missing fields to `fourier_legacy` for old checkpoints.

### 3.4 CLI/config fields

Add fields with these defaults:

```bash
--lmc_pos_encoding_mode fourier_legacy
--lmc_pos_fourier_v2_scales 1.0 2.0 4.0 8.0 16.0
--lmc_pos_fourier_coord_norm scene_radius
--lmc_pos_fourier_radius 4.0
--lmc_pos_fourier_learnable_scale False
--lmc_pos_fourier_residual_gate_init 0.0
```

Checkpoint `lmc_config` should include:

```text
pos_encoding_mode
pos_fourier_v2_scales
pos_fourier_coord_norm
pos_fourier_radius
pos_fourier_learnable_scale
pos_fourier_residual_gate_init
```

If practical, also save final learned values:

```text
final_pos_fourier_residual_gate
```

---

## 4. Compatibility and validation requirements

Before launching long training, the code execution agent should verify:

1. Legacy default compatibility:
   - Running parser defaults should keep `lmc_geo_bias_mode=legacy` and `lmc_pos_encoding_mode=fourier_legacy`.
   - Existing checkpoints without new fields should load and test normally.

2. Construction smoke tests:
   - Construct GeoLMC with `rbf_residual`.
   - Construct GeoLMC with `fourier_v2`.
   - Confirm output tensor shapes match legacy.

3. Isolation:
   - `GBR-main` must use `lmc_pos_encoding_mode=fourier_legacy`.
   - `FPE-main` must use `lmc_geo_bias_mode=legacy`.

4. Checkpoint round-trip:
   - Save a tiny/debug checkpoint with each mode if convenient.
   - Confirm test-time model reconstruction does not require manual flags.

5. No default behavior drift:
   - Do not change current B3-lite, TP-mapany-flow-v1, or FGPI-4090 command behavior unless the new flags are explicitly set.

---

## 5. First-round experiment matrix

Use the two spare 3090 GPUs. Do not run diagnostics in the first round.

| Run | GPU | Core change | Runtime stats | Purpose |
| --- | --- | --- | --- | --- |
| `GBR-main` | 3090 GPU0 | `--lmc_geo_bias_mode rbf_residual` | False | Test residual multi-scale RBF attention bias |
| `FPE-main` | 3090 GPU1 | `--lmc_pos_encoding_mode fourier_v2` | False | Test normalized Fourier PE v2 |

Both runs should preserve the FGPI-4090 contract:

```text
scene=/home/xwh/data/indoor6_ace/scene2a
memory=.../20260508_103423/memory_bse.pt
train_preset=memory_compare_ace_g_v1
data_backend=ace
use_lmc=True
lmc_profile=legacy
lmc_mode=global
lmc_auto_mode_by_visibility=False
lmc_fps_start_policy=farthest_from_center
lmc_key_slice_idx=2
s1_loss_step_mode=per_iter
lmc_feature_hierarchy_mode=selected_key_concat_value
lmc_fusion_geometry_mode=value_only_raw
training_buffer_size=2560000
buffer_size_final=7680000
buffer_on_cpu=false
buffer_on_cpu_final=true
buffer_sample_valid_coords=True
buffer_valid_coord_sample_ratio=1.0
buffer_valid_coord_neighbor_radius=1
buffer_valid_coord_neighbor_mode=cross
c1_aux_ref_loss_weight=0.0
batch_size=10240
post_train_eval_seeds=1305 2026 4242 7777 9001
post_train_hypotheses=256
```

---

## 6. Command templates

Replace `<GPU0>` and `<GPU1>` with the actual GPU IDs on the 3090 machine.

### 6.1 GBR-main

```bash
cd /home/xwh/project/ace_depth

python ace_dinov2_lmc/train_ace_dinov2_lmc.py \
  /home/xwh/data/indoor6_ace/scene2a \
  scene2a_c0_p4_GBR_main_20260519.pt \
  --run_name scene2a_c0_p4_GBR_main_20260519 \
  --train_preset memory_compare_ace_g_v1 \
  --data_backend ace \
  --device cuda:<GPU0> \
  --post_train_eval_device cuda:<GPU0> \
  --use_lmc True \
  --memory_path /home/xwh/project/ace_depth/ace_dinov2_lmc/memory_extraction/04_evaluation/memory_extract/scene2a/40v_v0.05_bilinear_bse_ut0.02_noaug_asb_adaptive_pool1.0_rgate4_prepair8_sor_gm_l2/20260508_103423/memory_bse.pt \
  --dinov2_path /home/xwh/data/checkpoints/dinov2_vitl14_pretrain.pth \
  --lmc_profile legacy \
  --lmc_mode global \
  --lmc_auto_mode_by_visibility False \
  --lmc_fps_start_policy farthest_from_center \
  --lmc_key_slice_idx 2 \
  --s1_loss_step_mode per_iter \
  --lmc_feature_hierarchy_mode selected_key_concat_value \
  --lmc_fusion_geometry_mode value_only_raw \
  --lmc_geo_bias_mode rbf_residual \
  --lmc_geo_bias_rbf_scales 0.25 0.5 1.0 2.0 4.0 \
  --lmc_geo_bias_rbf_alpha_init 0.0 \
  --lmc_geo_bias_rbf_learn_weights True \
  --lmc_geo_bias_rbf_per_head False \
  --lmc_pos_encoding_mode fourier_legacy \
  --experiment_subdir indoor6_full_baselines_GBR_geo_bias_rbf \
  --training_buffer_size 2560000 \
  --buffer_size_final 7680000 \
  --buffer_on_cpu false \
  --buffer_on_cpu_final true \
  --buffer_sample_valid_coords True \
  --buffer_valid_coord_sample_ratio 1.0 \
  --buffer_valid_coord_neighbor_radius 1 \
  --buffer_valid_coord_neighbor_mode cross \
  --c1_aux_ref_loss_weight 0.0 \
  --c1_aux_depth_root /home/xwh/data/mapanything-dataset/wai_data/indoor6/scene2a_train \
  --c1_aux_depth_kind gt_depth \
  --batch_size 10240 \
  --post_train_eval_seeds 1305 2026 4242 7777 9001 \
  --post_train_hypotheses 256
```

### 6.2 FPE-main

```bash
cd /home/xwh/project/ace_depth

python ace_dinov2_lmc/train_ace_dinov2_lmc.py \
  /home/xwh/data/indoor6_ace/scene2a \
  scene2a_c0_p4_FPE_main_20260519.pt \
  --run_name scene2a_c0_p4_FPE_main_20260519 \
  --train_preset memory_compare_ace_g_v1 \
  --data_backend ace \
  --device cuda:<GPU1> \
  --post_train_eval_device cuda:<GPU1> \
  --use_lmc True \
  --memory_path /home/xwh/project/ace_depth/ace_dinov2_lmc/memory_extraction/04_evaluation/memory_extract/scene2a/40v_v0.05_bilinear_bse_ut0.02_noaug_asb_adaptive_pool1.0_rgate4_prepair8_sor_gm_l2/20260508_103423/memory_bse.pt \
  --dinov2_path /home/xwh/data/checkpoints/dinov2_vitl14_pretrain.pth \
  --lmc_profile legacy \
  --lmc_mode global \
  --lmc_auto_mode_by_visibility False \
  --lmc_fps_start_policy farthest_from_center \
  --lmc_key_slice_idx 2 \
  --s1_loss_step_mode per_iter \
  --lmc_feature_hierarchy_mode selected_key_concat_value \
  --lmc_fusion_geometry_mode value_only_raw \
  --lmc_geo_bias_mode legacy \
  --lmc_pos_encoding_mode fourier_v2 \
  --lmc_pos_fourier_v2_scales 1.0 2.0 4.0 8.0 16.0 \
  --lmc_pos_fourier_coord_norm scene_radius \
  --lmc_pos_fourier_radius 4.0 \
  --lmc_pos_fourier_learnable_scale False \
  --lmc_pos_fourier_residual_gate_init 0.0 \
  --experiment_subdir indoor6_full_baselines_FPE_fourier_v2 \
  --training_buffer_size 2560000 \
  --buffer_size_final 7680000 \
  --buffer_on_cpu false \
  --buffer_on_cpu_final true \
  --buffer_sample_valid_coords True \
  --buffer_valid_coord_sample_ratio 1.0 \
  --buffer_valid_coord_neighbor_radius 1 \
  --buffer_valid_coord_neighbor_mode cross \
  --c1_aux_ref_loss_weight 0.0 \
  --c1_aux_depth_root /home/xwh/data/mapanything-dataset/wai_data/indoor6/scene2a_train \
  --c1_aux_depth_kind gt_depth \
  --batch_size 10240 \
  --post_train_eval_seeds 1305 2026 4242 7777 9001 \
  --post_train_hypotheses 256
```

---

## 7. Promote / reject criteria

Compare both runs against FGPI-4090:

```text
pct5=83.66
pct2=34.24
med_t=2.6233
med_r=0.2831
```

Promote or repeat if one of these holds:

- `pct5 >= 84.2` and `pct2 >= 34.0` with no med_t/med_r regression.
- pct5 roughly tied but `pct2 >= 35.5`.
- pct5 in `83.0~84.2` and no pct2/med_t regression; repeat once before combining with B3-lite or TP.

Reject if any of these holds:

- `pct5 < 82.5`.
- `pct2 < 33.0`.
- Training becomes unstable or test-time checkpoint loading requires manual non-checkpoint flags.
- Learned residual gates remain effectively zero and metrics do not improve.

Do not combine with B3-lite or TP-mapany-flow-v1 until a standalone positive signal is observed and repeated.

---

## 8. Reporting requirements

For each run, record:

- Full command.
- Git commit / code diff summary.
- Output directory and checkpoint path.
- `config.json` or equivalent config snapshot.
- Best checkpoint metadata.
- 5-seed post-train eval summary.
- For GBR, final RBF alpha and weights if available.
- For FPE, final PE residual gate if available.

Recommended compare note after results:

```text
ace_g_global_refactor/COMPARE_GBR_FPE_scene2a_202605XX.md
```

Append final accepted/rejected decision to:

```text
ace_g_global_refactor/EXPERIMENT_REFERENCE.md
```

---

## 9. Cautions

- Keep `GBR` and `FPE` separate in the first round.
- Do not enable diagnostics in the first round; use both 3090 GPUs for main metric runs.
- Do not enable `value_only_norm`, `geokey_norm`, CPE scene-scale, query-coordinate GeoMatch, or aux-ref expansion.
- Do not change `lmc_profile` away from `legacy` in these two experiments; TP-mapany-flow-v1 is already running separately.
- Defaults must preserve old behavior and old checkpoint compatibility.
