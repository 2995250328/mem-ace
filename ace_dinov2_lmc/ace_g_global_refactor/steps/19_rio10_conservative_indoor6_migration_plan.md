# Step 19: Conservative RIO10 Migration from the Best Indoor6 ACE-G Baseline

## 1. Purpose

This note defines the most conservative migration recipe from the current best Indoor6 ACE-G baseline to RIO10.

The goal is not to introduce a new architecture. The goal is to preserve the accepted Indoor6 behavior as much as possible, change only the dataset domain, and use RIO10 sparse depth in the lowest-risk way.

The intended migration principle is:

```text
Indoor6 best ACE-G baseline
+ RIO10 data domain
+ sparse-depth guided sampling only
+ matching MapAnything memory construction contract
```

This document has one required prerequisite and two optional supporting references.

Required first read:

- `steps/18_rio10_wai_training_fix_log.md` — this is mandatory because it establishes the corrected RIO10 backend chain, ACE-format training path, resize fix, sparse-depth attachment behavior, and the observed sparse-depth coverage limits that justify guided sampling without auxiliary supervision.

Read next if needed:

- `EXPERIMENT_REFERENCE.md` for the accepted Indoor6 baseline decision;
- `steps/16_large_scene_multi_memory_reference_frame_plan.md` if considering future multi-memory extensions.

## 2. Baseline to Preserve

The accepted Indoor6 reference is the FGPI-style 4090 run:

```text
4090_forceglobal_s1_periter
```

The important contract is not just its metric value, but the fact that it is a stable, true-global ACE-G baseline with per-iteration S1 semantics.

For RIO10 migration, preserve these settings exactly unless a later controlled ablation explicitly changes one of them:

```text
--use_lmc True
--lmc_flow ace_g
--lmc_mode global
--lmc_profile legacy

--s1_loss_step_mode per_iter

--lmc_key_slice_idx 2
--lmc_feature_hierarchy_mode selected_key_concat_value
--lmc_fusion_geometry_mode value_only_raw
```

The semantic requirements are:

```text
requested_lmc_mode = global
effective_lmc_mode = global
lmc_auto_mode_by_visibility = False
```

In other words, this migration must remain true global. It must not silently become effective-local through visibility fallback or scene heuristics.

## 3. What Sparse Depth Is Allowed to Do

Sparse depth should be used only as a sampling prior.

Allowed:

- attach sparse depth / sparse coordinate masks to the training dataset;
- use valid sparse-coordinate locations to bias buffer sampling;
- keep empty masks for frames without sparse depth;
- compare the same model with and without guided sampling.

Not allowed in the first migration:

- auxiliary sparse-depth supervision;
- auxiliary reference-coordinate loss;
- depth consistency loss;
- latent or fusion loss derived from sparse depth;
- any extra loss term that makes sparse depth a supervised target.

The intended training-side meaning is:

```text
sparse depth -> tells the buffer which patch locations are more geometrically supported
sparse depth -> does not supervise the network with an extra objective
```

The minimal config contract is therefore:

```text
--buffer_sample_valid_coords True
--c1_aux_ref_loss_weight 0.0
```

If existing valid-coordinate sampling parameters are used, they should be treated as sampling policy knobs only:

```text
--buffer_valid_coord_sample_ratio <keep from controlled recipe>
--buffer_valid_coord_neighbor_radius <keep from controlled recipe>
--buffer_valid_coord_neighbor_mode <keep from controlled recipe>
```

Do not interpret these as supervision parameters.

## 4. RIO10 Data Backend Split

RIO10 should keep the established backend split:

```text
memory extraction: WAI / MapAnything path
ACE-G training:    ACE backend
```

Reason:

- MapAnything memory extraction needs the WAI loader and MapAnything forward path.
- Indoor6 baseline training parity is closest when ACE-G training uses ACE-format `rgb`, `poses`, and `calibration` directories.
- Step18 already records that RIO10 has been converted to ACE format at:

```text
/data/xwh/RIO10_ace/scene01_seq01_01
```

The sparse-depth root remains external metadata, attached during ACE-backend training:

```text
--c1_aux_depth_root /data/xwh/RIO10_sparse_depth
--c1_aux_depth_kind sparse_depth
```

Step18 observed:

```text
matched sparse depth = 4356 / 4380
missing sparse depth = 24
valid sparse coord patches in frame-000000 = 51
```

This supports guided sampling, but it also warns that sparse-depth coverage is limited. That is another reason not to use sparse depth as an auxiliary supervision target in the first migration.

## 5. Matching Memory Construction Contract

The RIO10 memory should follow the same conservative global MapAnything contract used by the accepted ACE-G path.

Required memory construction choices:

```text
dataset_loader = wai
dataset_type = rio10
wai_view_mode = anchor_support / asb
MapAnything forward = single forward only
memory regime = single global memory
```

The critical constraint is single-forward consistency.

MapAnything internally produces reference-conditioned geometry and features. If multiple groups are forwarded separately, each group may have a different implicit reference frame. Concatenating those outputs as one memory would mix incompatible coordinates and reference-conditioned features.

Therefore, for this migration:

- do not split one memory into multiple MapAnything forwards;
- do not concatenate multiple separately extracted memories;
- do not introduce multi-memory routing yet;
- do not train a shared model over several reference-conditioned memories in the same first migration experiment.

Use one reference-consistent global memory first. Multi-memory routing belongs to the later Step16 direction, not to this conservative RIO10 migration.

## 6. Sparse-Depth Memory Sampling: Use `nearest_valid`

The one recommended memory-extraction correction for RIO10 sparse depth is:

```text
--patch_depth_sampling nearest_valid
```

instead of:

```text
--patch_depth_sampling nearest
```

Rationale:

- RIO10 sparse depth is sparse and may contain empty or invalid patch positions.
- `nearest` can select the nearest pixel even when it is invalid or unrepresentative.
- `nearest_valid` keeps the geometry construction tied to valid sparse depth inside the patch.
- This improves memory geometry quality without changing the ACE-G model architecture or the training objective.

This is a data-contract repair, not a new model mechanism.

The memory extraction contract should therefore be:

```text
WAI / MapAnything extraction
+ ASB view selection
+ single forward
+ sparse-depth-aware patch sampling with nearest_valid
+ output in the existing pooled/BSE-compatible memory format
```

## 7. Minimal Experiment Matrix

The first RIO10 migration should contain only two runs.

### Run A: Pure Indoor6 Baseline Migration

Purpose: measure what happens when the Indoor6 best baseline is moved to RIO10 with no sparse-depth guided sampling.

Keep:

```text
true global
per_iter
legacy profile
selected layer12 key
selected_key_concat_value
value_only_raw
same memory construction contract
```

Disable:

```text
buffer_sample_valid_coords = False
c1_aux_ref_loss_weight = 0.0
```

### Run B: Baseline + Sparse-Depth Guided Sampling

Purpose: isolate the effect of sparse depth as a sampling prior.

Identical to Run A except:

```text
buffer_sample_valid_coords = True
c1_aux_ref_loss_weight = 0.0
```

The only intended difference between A and B is whether valid sparse-coordinate locations influence buffer sampling.

If Run B improves over Run A, the interpretation is clean:

```text
sparse depth improved the sampled training locations
```

not:

```text
sparse depth added a second supervision task
```

## 8. Variables That Must Not Change in the First Migration

Do not change architecture:

- no PointRoPE;
- no Cascading Internal Fusion Assembly;
- no query-side DPT/FPN adapter;
- no multi-level compression change;
- no view-token or `all_scale_tokens` conditioning change.

Do not change memory regime:

- no multi-memory concatenation;
- no routing;
- no clustered memory package passed directly to the standard trainer;
- no mixed-reference memory.

Do not change loss semantics:

- no sparse-depth auxiliary loss;
- no C1 auxiliary reference supervision;
- no new geometry consistency loss.

Do not change evaluation semantics:

- keep `best_metric=pct5` if matching the accepted checkpoint selection rule;
- keep post-train evaluation protocol consistent with the Indoor6 reference when reporting final results;
- do not rank incomplete runs by their latest visible eval file;
- preserve exact source files and iteration provenance in result summaries.

## 9. Recommended Execution Order

1. Build or verify the RIO10 ACE-format scene.

```text
/data/xwh/RIO10_ace/scene01_seq01_01
```

2. Extract RIO10 memory with WAI / MapAnything:

```text
dataset_loader = wai
dataset_type = rio10
wai_view_mode = anchor_support
patch_depth_sampling = nearest_valid
single MapAnything forward
```

3. Train Run A with ACE backend and no guided sampling.

4. Train Run B with ACE backend and guided sampling only.

5. Evaluate both with the same checkpoint-selection and post-train evaluation protocol.

6. Compare A vs B before trying any architectural extension.

## 10. Why This Is the Conservative Migration

This recipe changes the minimum number of variables:

```text
changed: dataset domain
changed: optional sampling prior in Run B
fixed: ACE-G architecture
fixed: global/per_iter baseline semantics
fixed: memory reference-frame contract
fixed: loss semantics
fixed: evaluation semantics
```

This is important because the earlier RIO10 sparse-depth full run showed weak improvement and several confounders:

- sparse depth was very sparse at patch level;
- some training frames had no sparse-depth match;
- memory extraction used `nearest` despite sparse-depth recommendations favoring `nearest_valid`;
- the run used an older RIO10 preset;
- S1 appeared to saturate early;
- the result was incomplete, with best currently from an intermediate iteration.

The safest response is not to add more mechanisms. The safest response is to restore the strongest Indoor6 contract, repair the sparse-depth memory sampling contract, and isolate guided sampling as one controlled variable.

## 11. Final Recipe

The recommended RIO10 migration is:

```text
Indoor6 FGPI-style true-global ACE-G baseline
+ RIO10 ACE-format training data
+ WAI/MapAnything ASB single-forward global memory
+ patch_depth_sampling = nearest_valid
+ sparse depth attached only for valid-coordinate guided sampling
+ c1_aux_ref_loss_weight = 0.0
```

This should be the default RIO10 stabilization path before testing PointRoPE, CIFA, multi-level fusion, view tokens, or multi-memory routing.
