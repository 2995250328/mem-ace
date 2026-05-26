# Step 16: Large-scene Multi-memory Reference-frame Plan

## 1. Purpose

This document records the current design decision for extending ACE-DINOv2-LMC / ACE-G to large scenes that require multiple MapAnything memories.

The key problem is not only memory size or inference cost. MapAnything produces geometry and features under a reference-conditioned forward. If a large scene is split into multiple memories, each memory may have a different reference frame and a different reference-conditioned feature space. Therefore the system must not silently treat all memory tokens as if they came from one coherent global memory.

This step answers:

1. whether multiple memories should be concatenated into one large memory,
2. whether each memory should be trained/evaluated independently,
3. how routing and final pose selection should be introduced,
4. what metadata must be saved before more ambitious shared-model designs.

## 2. Current code and experiment constraints

Current code is single-memory by contract:

- training accepts one `--memory_path`,
- `load_memory_features()` accepts one memory path,
- clustered memory packages are explicitly rejected and should be passed as per-cluster memory files,
- the trainer builds one `memory_dict`,
- `GeoLMC` compresses one memory,
- fusion expands one compressed memory to the query batch,
- test-time evaluation caches one compressed memory for all test frames.

Existing ensemble support is not native multi-memory fusion. It evaluates independent checkpoint + memory bundles and selects the final pose by geometric evidence such as DSAC inlier count.

Prior reference-frame work also constrains the design:

- C0 remains the current default: MapAnything features may be reference-conditioned, but the ACE head predicts world-frame scene coordinates.
- C1 exists: memory points/targets can be transformed to a reference frame and predictions recovered to world before reprojection.
- C1 was validated but underperformed repaired C0 in prior scene2a experiments, so it should not replace C0 as the first large-scene path.
- One coherent MapAnything memory should come from one MapAnything forward. Multiple forwards may use different `T0` references and therefore produce incompatible reference-conditioned feature spaces.

## 3. Decision: do not concatenate memories as the first path

Directly concatenating multiple MapAnything memories into one large memory is not recommended now.

The superficial appeal is clear:

```text
memory_big = concat(memory_0, memory_1, ..., memory_n)
train once with --memory_path memory_big.pt
```

However this is unsafe because:

1. **Reference frames differ.** Each memory may be produced under a different MapAnything reference view.
2. **Features are reference-conditioned.** Even if points are transformed into a common world frame, the associated features may still carry different reference-context semantics.
3. **Current compressor/fusion has no frame awareness.** It consumes `pooled_points`, `pooled_features`, and `scene_center` as one coherent set, without `memory_id`, `frame_id`, or per-token reference metadata.
4. **Failure modes become hard to diagnose.** A negative result could come from coverage, routing, feature mismatch, coordinate mismatch, latent capacity, or training instability.

A direct-concat experiment should only be reconsidered after explicit frame-aware metadata and model support exist.

## 4. Immediate path: independent local-memory bundles

The recommended first large-scene path is:

```text
memory_0 -> checkpoint_0 -> pose hypothesis_0
memory_1 -> checkpoint_1 -> pose hypothesis_1
memory_2 -> checkpoint_2 -> pose hypothesis_2
...
query -> route / evaluate hypotheses -> final pose
```

Each MapAnything memory is treated as an independent local-map bundle:

```text
bundle_i = {
  memory_path_i,
  reference metadata_i,
  training config_i,
  checkpoint_i,
  eval outputs_i
}
```

This keeps each reference-conditioned feature space isolated and reuses the current single-memory training/evaluation code.

The default training target should remain C0 world regression:

```text
memory features: reference-conditioned within the bundle
memory points: world-frame C0 contract when available
head output: P_world
loss/eval: world reprojection / DSAC
```

C1 can remain a diagnostic branch, but it should not be the main large-scene baseline unless new evidence shows it beats C0.

## 5. Test-time routing and final pose selection

The first routing layer should be conservative and non-learned.

### 5.1 Baseline all-bundle evaluation

For a small initial matrix, run every bundle on every query frame:

```text
for query in test_set:
  for bundle in bundles:
    predict pose_hypothesis(bundle, query)
  select final pose by geometric score
```

Initial final-pose rule:

```text
final_pose = pose from bundle with highest DSAC inlier count
```

If inlier count is insufficient, compare:

```text
score = inlier_count - lambda * reprojection_error
```

or a small hand-designed ranking based on:

- inlier count,
- reprojection error,
- pose stability,
- confidence if already available.

### 5.2 Required reports

For every experiment, report:

1. best single bundle,
2. routed ensemble result,
3. per-bundle standalone results,
4. per-bundle selection frequency,
5. oracle upper bound.

The oracle upper bound is analysis-only:

```text
oracle(query) = hypothesis with lowest true pose error
```

It answers whether multiple memories contain useful complementary hypotheses. If oracle is weak, the problem is memory coverage or per-bundle model quality. If oracle is strong but routed ensemble is weak, the problem is routing.

## 6. Later path: top-k memory routing

After all-bundle routed evaluation shows a gain over the best single bundle, introduce top-k routing to reduce inference cost.

Possible non-learned routing signals:

1. query-to-reference image retrieval using DINOv2/global descriptors,
2. nearest selected memory views,
3. scene-region or coverage metadata,
4. coarse pose prior if available,
5. previous-frame temporal continuity for video-like sequences.

The routing objective should prioritize recall first. A wrong early route discards the correct local map before geometry can verify it.

Recommended first top-k policy:

```text
retrieve top-k memory bundles by query/reference image similarity
run ACE-G only on top-k
select final pose by DSAC inlier count or reprojection score
```

Use `k=2` or `k=3` before trying `k=1`.

## 7. Metadata contract

A multi-memory system needs stronger metadata than the current single-memory setting. Each memory bundle should record:

```text
memory_id
scene_id
mapanything_forward_id
reference_image_id
reference_pose
reference_frame_convention
world_frame_convention
T_world_ref or T_ref_world, with direction explicitly named
selected_view_ids
selected_view_poses
feature_model
feature_model_checkpoint
feature_conditioning_reference
memory_coordinate_frame
was_transformed_to_world
transform_validation_status
coverage_region / bbox / center / radius
```

If a memory lacks reference-frame and transform metadata, it must not participate in any concat or shared-latent experiment. It can only be used as an independent local bundle.

## 8. Future research: shared model with per-memory latents

After independent bundles and routing are validated, consider a shared-model design:

```text
shared DINOv2 / shared ACE head
memory_i -> compressed_latent_i
query -> router -> top-k latents
query + latent_i -> pose hypothesis_i
rank hypotheses geometrically
```

Possible variants:

1. shared head with per-memory compressed latent cache,
2. per-memory lightweight adapter or LoRA-style specialization,
3. memory-id / reference-pose embedding in fusion,
4. frame-aware attention masks,
5. learned query router trained after non-learned routing establishes an upper bound.

This is not the immediate path because it requires changes to:

- checkpoint config,
- memory loader API,
- trainer S1/S2 loop,
- test-time compressed-memory cache,
- routing supervision/evaluation,
- metadata validation.

## 9. Validation plan

### 9.1 Smoke test

Use two memory bundles first:

```text
bundle_0: local memory A
bundle_1: local memory B
```

Run reduced training:

```text
lmc_iterations = 1 or 2
small buffer if needed
same C0 config for both bundles
```

Confirm:

- each bundle trains independently,
- each checkpoint reconstructs its own memory at test time,
- ensemble/routing script can compare hypotheses,
- per-frame selection records are saved.

### 9.2 Full test

After smoke test passes:

1. train each bundle with the accepted ACE-G contract,
2. evaluate each bundle independently,
3. evaluate all-bundle routed ensemble,
4. evaluate top-k routing if image retrieval metadata exists,
5. compare against the best single-memory baseline and any existing scene-level ensemble baseline.

### 9.3 Promotion criteria

Promote the multi-memory route only if:

- routed ensemble beats the best single bundle on primary metrics,
- oracle upper bound shows meaningful complementarity,
- no single bundle dominates nearly all selected frames unless it also beats the baseline,
- failures can be explained by routing or coverage diagnostics,
- metadata is sufficient to reproduce bundle/reference-frame contracts.

Reject or pause if:

- routed ensemble does not beat best single bundle,
- oracle is not better than best single bundle,
- routing frequently selects geometrically invalid hypotheses,
- missing metadata prevents frame-contract diagnosis,
- gains require direct concatenation without frame-aware support.

## 10. Relationship to Step15 directions

This large-scene multi-memory plan is orthogonal to Step15 architecture directions.

Do not combine it initially with:

- new PointRoPE radius policies,
- query-side multi-level residual adapters,
- Layer12-anchored residual compression,
- new distance-bias modules.

First validate the system-level multi-memory route with the accepted C0/FGPI-style contract. Only after the routing/bundle interface is stable should Step15 architecture improvements be applied inside each local bundle.

## 11. Current recommendation

For large scenes requiring multiple MapAnything memories:

1. do not concatenate memories now,
2. create independent local memory bundles,
3. train/evaluate each bundle separately under C0 world regression,
4. select final pose by geometric evidence,
5. add top-k retrieval routing only after all-bundle ensemble shows a gain,
6. postpone shared-model/per-memory-latent fusion until the independent-bundle baseline is validated.
