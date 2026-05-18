# ACE-G Long-Term Research Plan

This file tracks ideas from the local `silu/` design documents that are useful
but too broad for the current ACE-G global baseline stabilization track.

Near-term, compatibility-focused work remains in `PLAN.md` and `steps/`.

## Guiding Rule

Do not implement these before the true-global ACE-G baseline, attention
diagnostics, and Progressive Geometry Injection / GeoKey v0 ablations are
understood.

Each item below changes multiple contracts at once and should get its own
experiment family, output directory, and checkpoint metadata.

## Dependency Order

The long-term items are not independent. The intended dependency is:

```text
single-scene true-global baseline
  -> diagnostics and Progressive Geometry Injection / GeoKey v0
  -> explicit coordinate-frame metadata
  -> reference-frame memory contract
  -> normalized coordinate contract
  -> multi-scene joint training
  -> sub-memory routing / reranking for large scenes
  -> head-level expert or anchor-residual designs
```

Training-free MapAnything/VGGT acceleration is useful for memory construction,
but it is not a prerequisite for the current ACE-G architecture experiments and
should be tracked as a separate systems-efficiency branch.

## Long-Term Directions

### 1. Multi-Scene Joint LMC Training

Goal:

- Move beyond single-scene, single-FPS-memory ACE-G training.
- Train an LMC that can consume memories from multiple scenes and learn a more
  general memory-compression / memory-fusion prior.

Current state:

- Current experiments are single scene and single memory.
- The compressor is trained for one scene distribution at a time.
- This is good for stabilizing ACE-G but does not yet prove a reusable LMC.

Why it may help:

- A jointly trained LMC can learn common geometric/semantic patterns across
  scenes instead of overfitting one memory.
- It is the path toward a practical feed-forward memory-conditioned localizer.

Required contracts:

- Per-scene metadata must be explicit.
- Coordinate frame and normalization policy must be explicit.
- Checkpoints must record scene-scale, coordinate-frame, and memory-shard
  assumptions.
- Batches must be able to mix scenes without ambiguous loss scaling.
- The model must know which memory or sub-memory each query is conditioned on;
  scene identity must not be inferred from path names or output folders.
- Metrics must be reported per scene and aggregated across scenes, so one large
  or easy scene cannot hide regressions elsewhere.

Risks:

- Larger scenes can dominate gradients without normalization.
- Different scene centers and coordinate frames can silently leak into the
  model unless represented consistently.
- Joint training will make debugging harder unless single-scene behavior is
  already stable.
- A joint LMC is not proven by concatenating scenes into one memory. It needs an
  explicit batching/routing contract and per-scene coordinate recovery.

### 2. Reference-Frame Memory Coordinate Contract

Goal:

- Remove the gap between MapAnything/reference-frame features and memory point
  coordinates.
- Convert point clouds into the selected reference coordinate frame so features
  and 3D points are represented in the same frame.
- Use the recorded camera/reference pose transforms explicitly for this
  conversion; do not leave memory features in a reference-frame convention while
  memory points stay silently in world coordinates.
- At inference/evaluation time, predict or assemble coordinates in reference
  space, then convert back to world coordinates for DSAC / PnP.

Important constraint:

- Keep the current `scene_center` policy: use the mean of all camera centers.
- This policy has shown good empirical behavior and should not be replaced
  casually by reference-frame origin or point-cloud centroid.
- If a sub-memory also needs local metadata, record that separately. Do not
  silently redefine the global scene center.

Proposed coordinate chain:

```text
P_world
  -> reference-frame transform
P_ref
  -> optional normalization
P_ref_norm
  -> model prediction / loss
P_ref
  -> inverse reference transform
P_world
  -> DSAC / PnP evaluation
```

Why it may help:

- MapAnything-derived features and points should have a cleaner one-to-one
  coordinate interpretation.
- Anchor/residual heads become easier to reason about when feature and point
  frames match.

Risks:

- There are now two transformations to track when normalization is also enabled:
  world -> reference -> normalized, and inverse normalized -> reference ->
  world.
- Pose, depth, scene center, invalid-depth thresholds, and checkpoint metadata
  must all agree.
- Single-scene equivalence tests are mandatory before multi-scene training.

Open design questions:

- Which reference frame is authoritative: the memory reference view, a selected
  canonical view, or a cluster/sub-memory reference?
- Should sub-memories have their own reference transforms?
- How should reference transforms be recorded in memory files and checkpoints?
- Should the model output `P_ref`, `P_ref_norm`, or a residual relative to
  reference-frame anchors?
- Which losses operate before denormalization, and which diagnostics must be
  reported after recovery to world coordinates?

### 3. Normalized Coordinate Contract

Goal:

- Train and evaluate with explicit per-scene coordinate normalization.
- In the future reference-frame design, the likely target is:
  `P_ref_norm = (P_ref - mu_ref) / sigma_scene`.
- Denormalize before converting back to world coordinates and before DSAC / PnP
  evaluation.
- Treat the two operations separately:
  world/reference transform handles coordinate frame;
  normalization/denormalization handles numeric scale.

Why it may help:

- Multi-scene training can otherwise be dominated by scene scale.
- It creates a cleaner bridge toward metric-bypass and anchor-residual designs.
- This matches the direction used by feed-forward 3D systems such as
  VGGT/MapAnything-style pipelines, where normalized coordinates reduce scale
  imbalance.

Risks:

- Depth thresholds, invalid loss, reprojection loss, memory coordinates, and
  checkpoint reconstruction must all agree on the same coordinate space.
- Needs single-scene equivalence tests before multi-scene use.
- Combined with reference-frame memory, there are two coordinate operations:
  world/reference transform and normalization/denormalization.
- A normalized target can look numerically stable while world-space pose metrics
  regress if recovery metadata is wrong.

Minimum validation:

- On one scene, train/evaluate with normalization enabled and verify recovered
  world-coordinate evaluation is comparable to the unnormalized contract.
- Log `mu`, `sigma_scene`, source coordinate frame, and every recovery step in
  checkpoint metadata.
- Report loss in normalized units and pose metrics in world units.

### 4. Sub-Memory Construction And Routing For Large Scenes

Goal:

- Use sub-memories for large-scale scenes instead of forcing one global memory
  to represent everything.
- Borrow the high-level idea from ACE-style Cambridge handling: split a large
  scene/dataset into clusters/subsets, then build a separate memory for each
  subset.

Proposed direction:

- Cluster reference frames or spatial regions into sub-datasets.
- Build one sub-memory per cluster.
- Treat the clustering step as an isolation mechanism for memory construction:
  each clustered sub-dataset can produce its own sub-memory instead of forcing
  unrelated regions into one FPS memory.
- Keep a single model, but let it consume routed sub-memories.
- Each sub-memory may need its own reference transform and possibly its own
  local scene-center-like metadata, while the global scene-center policy remains
  explicitly recorded.
- The clustering idea can borrow from ACE Cambridge-style dataset subdivision,
  but the training contract differs: one model should learn to operate over
  multiple routed memories rather than one independent model per cluster.

Why it may help:

- Reduces memory compression burden for large scenes.
- Creates cleaner local geometry and avoids forcing unrelated regions into the
  same K-token global summary.
- Provides a natural path toward local/reference-frame memories.

Key routing question:

- Pre-route query to top-k sub-memories, then run inference only on those.
- Or run multiple sub-memories and fuse/rerank final results.

Pre-routing pros:

- Faster.
- Lower memory and compute.
- Necessary for very large scenes.

Pre-routing risks:

- Wrong route can fail the entire query.
- Needs top-k fallback and confidence estimation.

Post-inference fusion / reranking pros:

- More robust if routing is uncertain.
- Can use pose-level evidence such as RANSAC inliers or reprojection score.

Post-inference fusion / reranking risks:

- More expensive.
- Requires careful merging or selection of scene-coordinate hypotheses.

Recommended first design:

- Use pre-routing to top-2 or top-3 sub-memories.
- Run normal ACE-G inference for those candidates.
- Rerank by DSAC/RANSAC inliers or reprojection score.
- Do not average world coordinates from unrelated sub-memories unless they share
  a verified coordinate contract.
- Log router confidence, selected memory IDs, runner-up IDs, and whether the
  final pose came from top-1 routing or reranking.
- Keep a slower "run all candidate sub-memories and rerank" mode as an
  evaluation oracle for diagnosing router mistakes.

### 5. Anchor-Assisted Coordinate Head As Primary Path

Goal:

- Move from ACE-head absolute coordinate regression toward anchor-relative
  prediction:
  `P = anchor(memory_p_k) + delta_k`.

Why it may help:

- Makes latent coordinates physically meaningful as local experts.
- Reduces burden on the head to infer absolute translation from features.

Risks:

- Replaces a known-working ACE head contract.
- Can destabilize S1/S2 unless introduced first as an auxiliary branch.

### 6. MoE / Expert Heads

Goal:

- Treat memory tokens as explicit experts.
- Use temperature-scaled or Top-M routing, local residual heads, and optional
  load balancing.
- Explore whether the coordinate head itself should use multiple experts for
  different object/region types, foreground/background, or local geometry
  regimes.

Why it may help:

- Can reduce coordinate averaging in weak-texture regions.
- Gives a path toward interpretable local scene experts.
- Different objects and background regions may have different feature/geometry
  distributions; a single head may be forced to average incompatible mappings.

Important caution:

- MoE by object category is not automatically valid. The available supervision
  is scene-coordinate reprojection, not semantic labels.
- A token/anchor-routed MoE is easier to justify than a purely semantic MoE,
  because routing can be tied to memory geometry and localization evidence.
- Foreground/background separation may help if dynamic or weakly localized
  foreground content is detected, but it needs masks or reliable confidence.
- Object-specific experts are plausible only if the routing signal is observed
  and stable. Without semantic masks or reliable object labels, "different
  objects need different experts" is a hypothesis, not a training contract.

Risks:

- Simultaneously changes fusion, head, loss, and training dynamics.
- Requires token diagnostics before any load-balancing loss is justified.
- Expert collapse is likely without diagnostics and careful regularization.

### 7. Ray / Plucker Geometry Usage

Goal:

- Use currently stored but mostly unused memory geometry, including rays and
  Plucker-style ray maps, to improve LMC and head design.

Possible uses:

- Add ray-aware attention bias in compressor/fusion.
- Condition anchor residual heads on viewing direction.
- Add consistency diagnostics between query rays, memory rays, and predicted
  scene coordinates.
- Build local frames or visibility priors from reference observations.
- Use ray / Plucker fields as diagnostics first: check frame conventions,
  normalization, and whether they agree with stored points and camera poses
  before feeding them into the model.

Why it may help:

- Rays encode view geometry not present in point coordinates alone.
- They may help distinguish foreground/background, repeated structures, and
  ambiguous local texture.

Risks:

- Current query pipeline is calibrated; local-frame and uncalibrated assumptions
  must not be mixed casually.
- Requires reliable anchor visibility / direction metadata.
- Ray features are easy to overfit and should be introduced only after attention
  diagnostics exist.

### 8. Ray-Factorized / Local-Frame Residuals

Goal:

- Predict local residuals in a learned or geometry-derived local frame, possibly
  splitting depth-like and image-plane-like components.

Why it may help:

- Could make residual prediction easier and more geometry-aware.

Risks:

- Requires the reference-frame and ray metadata contracts to be clear first.
- Should start as an auxiliary branch rather than replacing ACE head.

### 9. DSD-Style Local / Deformable Compressor

Goal:

- Move beyond fixed FPS global tokens by using local neighborhoods, bounded
  token offsets, or task-driven token refinement.

Why it may help:

- Tokens can move from coverage-driven anchors toward localization-useful
  regions.

Risks:

- Changes compressor output distribution.
- S2 currently freezes compressor, so S1/S2 distribution drift can become worse.
- Should follow GeoMatch diagnostics, not precede them.

### 10. Training-Free MapAnything / VGGT Acceleration Ideas

Goal:

- Reduce MapAnything/VGGT-style multi-view inference memory pressure and speed
  up feature/memory construction.
- Track training-free ideas such as:
  - `AVGGT: Rethinking Global Attention for Accelerating VGGT`
  - `Faster VGGT with Block-Sparse Global Attention`
  - later MapAnything/VGGT inference acceleration methods.

Why it may help:

- MapAnything cannot process too many views at once under current GPU memory
  limits.
- Faster global-attention approximations could make memory extraction cheaper
  and enable larger reference sets.

Risks:

- This affects memory construction / feature extraction, not the immediate ACE-G
  compressor/fusion path.
- It should be tracked separately from LMC architecture changes to avoid mixing
  causes.
- If an acceleration method changes features numerically, it must get a new
  memory/extractor condition name and cannot be compared as the same memory.

### 11. Uncalibrated Query / P4Pf Solver Track

Goal:

- Handle query images without known intrinsics by jointly estimating focal length
  and pose.

Why it may help:

- Expands the system to a different deployment setting.

Risks:

- This is a different problem statement from the current ACE/DSAC calibrated
  pipeline.
- Should not be mixed into current indoor6 ACE-G baseline experiments.

## Promotion Criteria

A long-term item can move into the near-term `PLAN.md` only when:

- The current global baseline has a stable, named comparison point.
- Required observability exists.
- The item can be tested as one controlled ablation.
- Training and evaluation metadata can disambiguate the new contract from older
  checkpoints.
