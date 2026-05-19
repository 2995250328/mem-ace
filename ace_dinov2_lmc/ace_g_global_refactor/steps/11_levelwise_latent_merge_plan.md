# Step 11 - B3-lite Levelwise Latent Merge Plan

Status: handoff-ready design
Updated: 2026-05-19

## Purpose

Design and hand off a controlled multi-layer memory compression ablation for
ACE-G true-global LMC. The goal is to test whether multi-layer DINOv2 /
MapAnything memory features are useful when they are preserved through
compression, instead of being collapsed before the compressor.

This document is for the code execution agent. It is not an implementation
record. Do not treat it as evidence that the code already exists.

## Current Evidence

Current clean reference on `scene2a/c0_p4`:

| Run | Mode | S1 step | Key | pct5 | pct2 | med_t | med_r |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| `FGPI-4090` / `4090_forceglobal_s1_periter` | true global | `per_iter` | slice 2 / layer 12 | 83.66 | 34.24 | 2.6233 | 0.2831 |

Rejected or diagnostic-only results:

| Run | Main change | pct5 | pct2 | med_t | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| CPE-A1 | `value_only_norm` + compressor PE `scene_scale` | 81.32 | 30.74 | 2.6946 | reject |
| CPE-A1 supplement | same, diagnostic package | 78.60 | 28.79 | 2.7660 | reject |
| B1 scalar mix | sharp/current scalar mix key | 79.77 | 31.91 | 2.8101 | reject |
| B1 scalar mix supplement | repeat/current scalar mix key | 78.21 | 31.91 | 2.9068 | reject |

B1 scalar mix repeatedly collapsed to layer 12:

```text
0.000343, 0.000342, 0.998629, 0.000341, 0.000344
layers: 0, 6, 12, 18, final
```

Conclusion:

- Do not rerun current B1 scalar mix unchanged.
- Do not expand CPE-A1.
- A useful multi-layer test must preserve layer-specific information longer than
  input-side scalar mixing.

## Recommended Method

Name:

```text
B3-lite Levelwise Latent Merge
```

Core idea:

```text
multi-layer memory features
  -> per-layer projection
  -> per-layer cross-attention into the same K spatial latent coordinates
  -> per-layer latent outputs Z_l
  -> small global level gate
  -> final compressed memory Z
  -> existing LMCFeatureFusion unchanged
```

The method keeps level-specific keys and values until after latent compression.
It therefore avoids the main B1 failure mode: input-side scalar mix collapsing to
one selected layer before compression.

## Non-Goals

Do not include any of the following in the first implementation or first run:

- CPE / compressor PE `scene_scale` changes.
- `value_only_norm` / `geokey_norm` fusion geometry changes.
- Query-coordinate GeoMatch.
- Aux-ref loss or alpha scaling.
- Force-global fixed-zero as a candidate default.
- Memory extraction changes.
- ACE head architecture changes.
- Multi-scene training.
- Token-wise level gates in the first run.
- Entropy regularization in the first run.

## Architecture Contract

### Input

Memory files may provide multi-layer features as concatenated features:

```text
pooled_features: [N, L*C]
layers_idx: [0, 6, 12, 18, final]
```

Typical current case:

```text
L = 5
C = 1024
```

The new branch reshapes internally:

```text
F: [N, L, C]
P: [N, 3]
latent_coords: [K, 3]
```

### Per-layer projection

For each memory layer:

```text
F_l_proj = LayerNorm(Linear_l(F_l))
```

First implementation choice:

```text
per-layer projection: yes
shared projection across layers: no
```

Rationale:

- Different DINOv2 / MapAnything layers have different distributions and roles.
- Independent projections make the first test less likely to underfit layer
  differences.
- Parameter growth is controlled elsewhere by sharing the cross-attention module
  and using a global level gate.

### Per-layer latent compression

For each layer `l`:

```text
Z_l = CrossAttention(
  query = latent_tokens + PE(latent_coords),
  key   = F_l_proj + PE(memory_points),
  value = F_l_proj
)
```

Output:

```text
Z_l: [K, D]
Z_all: [L, K, D]
```

First implementation choice:

```text
shared cross-attention weights across layers: yes
per-layer cross-attention weights: no
```

Rationale:

- Keeps capacity increase moderate.
- Makes level differences mostly attributable to input feature levels and
  projection, not separate attention modules.
- Keeps debugging simpler.

### Level merge

First implementation uses a global softmax gate:

```text
gate = softmax(level_logits)
Z = sum_l gate_l * Z_l
```

Gate shape:

```text
[L]
```

Do not implement token-wise gates in the first version. Token-wise gates are more
expressive but reduce interpretability and add another collapse mode.

### Output compatibility

The final compressed memory must have the same shape and downstream semantics as
current GeoLMC output:

```text
compressed_memory_z: [K, D]
compressed_memory_points / latent coords: unchanged
```

`LMCFeatureFusion` should not need a behavior change for the first version.

## Code Locations

Primary files:

- `/home/xwh/project/ace_depth/ace_compressor.py`
  - `GeoLMC`
  - current key/value feature path
  - current cross-attention compression path
  - add `levelwise_latent_merge` branch here

- `/home/xwh/project/ace_depth/ace_dinov2_lmc/options_dinov2_lmc.py`
  - add CLI fields with default values that preserve old behavior

- `/home/xwh/project/ace_depth/ace_dinov2_lmc/trainer_dinov2_lmc.py`
  - pass new args into compressor construction
  - write new fields into `lmc_config`
  - include fields in checkpoint / best metadata / eval summaries
  - include runtime diagnostics if enabled

- `/home/xwh/project/ace_depth/ace_dinov2_lmc/test_ace_dinov2_lmc.py`
  - restore compressor from checkpoint metadata
  - provide old-checkpoint defaults

- Optional:
  - `/home/xwh/project/ace_depth/ace_dinov2_lmc/test_ace_dinov2_lmc_ensemble.py`
    if ensemble eval reconstructs LMC compressor independently.

## New CLI / Config / Checkpoint Fields

All defaults must preserve current behavior.

| Field | Default | First B3-lite value | Meaning |
| --- | --- | --- | --- |
| `--lmc_feature_hierarchy_mode` | `selected_key_concat_value` | `levelwise_latent_merge` | Select old path vs B3-lite path |
| `--lmc_level_merge_mode` | `softmax_gate` | `softmax_gate` | Level merge mechanism |
| `--lmc_level_merge_init` | `uniform` | `uniform` | Level gate initialization |
| `--lmc_level_proj_shared` | `False` | `False` | Whether projection is shared across layers |
| `--lmc_level_cross_attn_shared` | `True` | `True` | Whether cross-attention is shared across layers |
| `--lmc_level_gate_entropy_weight` | `0.0` | `0.0` | Optional future entropy regularizer, off initially |
| `--lmc_level_token_gate` | `False` | `False` | Future token-wise gate, off initially |

Recommended checkpoint / summary fields:

```text
lmc_feature_hierarchy_mode
lmc_level_merge_mode
lmc_level_merge_init
lmc_level_proj_shared
lmc_level_cross_attn_shared
lmc_level_gate_entropy_weight
lmc_level_token_gate
lmc_level_merge_weights
lmc_level_gate_entropy
layers_idx
```

Runtime diagnostics when `--lmc_log_runtime_stats True`:

```text
per_level_latent_norm_mean
per_level_latent_norm_std
per_level_attention_entropy_mean
per_level_effective_memory_token_count
final_level_merge_weights
final_level_gate_entropy
fusion_effective_token_count
fusion_attention_entropy
```

Keep diagnostics scalar-only. Do not save large attention tensors.

## Old Checkpoint Compatibility

When loading old checkpoints, use:

```text
missing lmc_feature_hierarchy_mode -> selected_key_concat_value
missing lmc_level_merge_mode -> softmax_gate
missing lmc_level_merge_init -> uniform
missing lmc_level_proj_shared -> False
missing lmc_level_cross_attn_shared -> True
missing lmc_level_gate_entropy_weight -> 0.0
missing lmc_level_token_gate -> False
```

Old checkpoints must continue to load and evaluate with old behavior.

## Implementation Checklist

1. Add parser fields in `options_dinov2_lmc.py`.
2. Add a default-preserving `feature_hierarchy_mode` branch in `GeoLMC`.
3. Implement B3-lite modules:
   - per-layer projection modules,
   - shared layerwise cross-attention path,
   - global level softmax gate,
   - scalar diagnostics for merge weights and entropy.
4. Pass args from trainer into compressor construction.
5. Save new fields into `lmc_config` and checkpoint metadata.
6. Restore new fields in test-time compressor reconstruction.
7. Add post-train summary fields.
8. Run lightweight verification only after implementation:
   - parser accepts new flags,
   - old default command still builds old path,
   - B3-lite command builds new path,
   - old checkpoint loads,
   - no large attention tensors are written.

## First Experiment Contract

Use the same scene and memory contract as the current accepted reference.

Scene:

```text
/home/xwh/data/indoor6_ace/scene2a
```

Memory:

```text
/home/xwh/project/ace_depth/ace_dinov2_lmc/memory_extraction/04_evaluation/memory_extract/scene2a/40v_v0.05_bilinear_bse_ut0.02_noaug_asb_adaptive_pool1.0_rgate4_prepair8_sor_gm_l2/20260508_103423/memory_bse.pt
```

Core fixed flags:

```text
--train_preset memory_compare_ace_g_v1
--data_backend ace
--use_lmc True
--lmc_mode global
--lmc_auto_mode_by_visibility False
--lmc_fps_start_policy farthest_from_center
--lmc_key_slice_idx 2
--s1_loss_step_mode per_iter
--lmc_fusion_geometry_mode value_only_raw
--training_buffer_size 2560000
--buffer_size_final 7680000
--buffer_on_cpu false
--buffer_on_cpu_final true
--buffer_sample_valid_coords True
--buffer_valid_coord_sample_ratio 1.0
--buffer_valid_coord_neighbor_radius 1
--buffer_valid_coord_neighbor_mode cross
--c1_aux_ref_loss_weight 0.0
--batch_size 10240
--post_train_eval_seeds 1305 2026 4242 7777 9001
--post_train_hypotheses 256
```

B3-lite flags:

```text
--lmc_feature_hierarchy_mode levelwise_latent_merge
--lmc_level_merge_mode softmax_gate
--lmc_level_merge_init uniform
--lmc_level_proj_shared False
--lmc_level_cross_attn_shared True
--lmc_level_gate_entropy_weight 0.0
--lmc_level_token_gate False
```

## Experiment Matrix

Available hardware:

- Local machine: 3 usable GPUs.
- 3090 server: 2 usable GPUs.

Do not launch all rows blindly. Run in stages and stop early on clear failure.

### Stage A - Implementation smoke / parser checks

Run locally on one GPU only after code changes.

| ID | Machine | GPU count | Purpose | Required outcome |
| --- | --- | ---: | --- | --- |
| A0 | local | 1 | parser + construction smoke | new flags parse; old defaults unchanged |
| A1 | local | 1 | old checkpoint eval smoke | old checkpoint loads with old path |
| A2 | local | 1 | B3-lite tiny train/eval smoke if feasible | no shape error; summaries contain level fields |

These are engineering checks, not research metrics.

### Stage B - Main research run

Run one full B3-lite training first.

| ID | Machine | GPU | Variant | Runtime stats | Decision |
| --- | --- | --- | --- | --- | --- |
| B3L-main | local preferred | one of 3 local GPUs | `levelwise_latent_merge` | False | Primary metric run |

Use one local GPU to avoid occupying all available hardware. Keep the other two
local GPUs free for unrelated work or quick diagnostics unless the user says to
parallelize.

### Stage C - Diagnostic run

Only run if `B3L-main` is not a clear reject.

| ID | Machine | GPU | Variant | Runtime stats | Purpose |
| --- | --- | --- | --- | --- | --- |
| B3L-diag | local or 3090 | 1 | same as B3L-main | True | Explain level usage and attention behavior |

### Stage D - Repeat / paired reliability

Only run if `B3L-main` is close to or above reference.

| ID | Machine | GPU | Variant | Purpose |
| --- | --- | --- | --- | --- |
| B3L-repeat-local | local | 1 | same as main, different training seed if supported | training repeat |
| B3L-repeat-3090 | 3090 server | 1 | same as main | hardware/server replication |

Given 3 local GPUs and 2 3090 GPUs, a reasonable parallel plan after a promising
main run is:

```text
local GPU 0: B3L-repeat-local
local GPU 1: B3L-diag
3090 GPU 0: B3L-repeat-3090
```

Keep at least one local GPU and one 3090 GPU free unless explicitly approved.

### Stage E - Follow-up ablations, only after positive signal

Do not run these unless B3L-main / repeat show a plausible win.

| ID | Change | Question |
| --- | --- | --- |
| B3L-shared-proj | `--lmc_level_proj_shared True` | Is the gain from level structure or extra projection params? |
| B3L-token-gate | `--lmc_level_token_gate True` | Do different latent tokens prefer different layers? |
| B3L-entropy | small `--lmc_level_gate_entropy_weight` | Does preventing gate collapse help after baseline B3L? |

## Suggested Main Command Template

Fill in device and output names when launching. Do not run this from this plan
creation session.

```bash
cd /home/xwh/project/ace_depth
python ace_dinov2_lmc/train_ace_dinov2_lmc.py \
  /home/xwh/data/indoor6_ace/scene2a \
  scene2a_c0_p4_B3L_levelwise_latent_merge_20260519.pt \
  --train_preset memory_compare_ace_g_v1 \
  --data_backend ace \
  --device cuda:<GPU_ID> \
  --post_train_eval_device cuda:<GPU_ID> \
  --use_lmc True \
  --memory_path /home/xwh/project/ace_depth/ace_dinov2_lmc/memory_extraction/04_evaluation/memory_extract/scene2a/40v_v0.05_bilinear_bse_ut0.02_noaug_asb_adaptive_pool1.0_rgate4_prepair8_sor_gm_l2/20260508_103423/memory_bse.pt \
  --lmc_mode global \
  --lmc_auto_mode_by_visibility False \
  --lmc_fps_start_policy farthest_from_center \
  --lmc_key_slice_idx 2 \
  --s1_loss_step_mode per_iter \
  --lmc_fusion_geometry_mode value_only_raw \
  --lmc_feature_hierarchy_mode levelwise_latent_merge \
  --lmc_level_merge_mode softmax_gate \
  --lmc_level_merge_init uniform \
  --lmc_level_proj_shared False \
  --lmc_level_cross_attn_shared True \
  --lmc_level_gate_entropy_weight 0.0 \
  --lmc_level_token_gate False \
  --experiment_subdir indoor6_full_baselines_B3L_levelwise_latent_merge \
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

Diagnostic command is the same plus:

```text
--lmc_log_runtime_stats True
--lmc_runtime_stats_interval 100
--lmc_runtime_stats_max_pixels 4096
```

## Promote / Reject Criteria

Compare against FGPI-4090:

```text
pct5=83.66
pct2=34.24
med_t=2.6233
med_r=0.2831
```

### Strong promote

```text
pct5 > 83.66
pct2 >= 34.24
med_t <= 2.65
med_r <= 0.29
```

and:

```text
max(level_merge_weights) < 0.90
```

### Weak promote / repeat required

```text
pct5 >= 83.0
pct2 > 34.24
med_t <= 2.70
```

and no severe level collapse.

### Reject

Reject if any of the following holds:

```text
pct5 < 82.0
pct2 < 32.5
med_t > 2.85
max(level_merge_weights) > 0.95 and dominant layer is layer 12
```

If rejected, do not tune a grid of entropy / temperature / token gates. Move to a
separate B2-lite fused-neck plan or revisit S1/S2 training contract.

## Reporting Requirements

Append completed results to:

```text
/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor/EXPERIMENT_REFERENCE.md
```

Include:

- exact command or `run_command.txt` path,
- scene and memory path,
- checkpoint path,
- eval protocol and seeds,
- `pct5`, `pct2`, `med_t`, `med_r`, avg ms,
- level merge weights and entropy,
- whether runtime diagnostics were enabled,
- decision: promote / weak-promote / reject.

Also create or update a focused comparison note under:

```text
/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor/
```

Suggested filename:

```text
COMPARE_B3L_levelwise_latent_merge_scene2a_202605XX.md
```

## Cautions For Code Execution Agent

- Do not change default behavior. Old commands and old checkpoints must keep the
  selected-key/all-layer-value path.
- Do not silently map old `slice` / `selected` names incorrectly. Preserve all
  existing aliases if present.
- Do not add new files unless necessary; prefer modifying existing code and docs.
- Do not start full experiments until parser/checkpoint smoke checks pass.
- Do not run CPE-A1 or B1 scalar mix unchanged; both are completed and negative.
- Do not claim multi-layer compression works if level weights collapse to layer
  12. A metric win with collapse is a selected-layer variant, not evidence for
  multi-layer hierarchy.
