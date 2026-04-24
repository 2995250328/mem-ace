# Reference-Consistent Adaptive Memory Construction Execution Plan

## 1. Execution Principles

This document is the implementation guide for the v3 design in
`memory_extraction/01_design/改进.md`. It is intentionally scoped to the
next engineering steps, not to broader research brainstorming.

The execution order is:

1. Make the current pipeline observable.
2. Validate the reference contract.
3. Upgrade the single-forward selector.
4. Add the two-level feasibility gate.
5. Add clustered fallback.
6. Add the query-time ensemble interface.

The first implementation must keep the current memory extraction path usable.
New behavior should be guarded by explicit config flags and should emit enough
metadata for offline analysis.

Current scope:

- Build reference-consistent memory construction.
- Support `C0` and prepare `C1`.
- Keep single-forward as the default path.
- Use cluster branch only as fallback.
- Use offline probe signals when training/reference poses are available.

Out of scope:

- BSE, voxel pooling, or point-density optimization.
- Backbone replacement.
- Learned scene policy.
- Learned query routing.
- Cross-cluster feature fusion.
- Online pose-free memory construction.

## 2. Contract Modes

The implementation must support two contract modes. The default is `C0` until
Q1 validation proves that `C1` is safe.

### C0: Reference-Conditioned Only

Use this as the initial default.

Semantics:

- MapAnything memory features are treated as reference-conditioned features.
- They are not assumed to be equivariant to the reference camera frame.
- The network output is `P_world`.
- Training and evaluation use the existing world-frame reprojection / DSAC path.
- `points_ref` and `points_ref_norm` may be written for diagnostics, but they
  are not regression targets.
- Reference metadata must still be available to the memory/fusion path.

Required memory fields:

```python
memory = {
    "mode": "single_forward" | "cluster_local",
    "contract_mode": "C0",
    "features_conditioned": Tensor,       # [N, D]
    "points_world": Tensor,               # [N, 3], primary geometry target
    "points_ref": Tensor | None,          # diagnostic
    "points_ref_norm": Tensor | None,     # diagnostic
    "conditioning_reference": {
        "reference_index": int,
        "T_ref_c2w_world": Tensor,        # [4, 4]
        "T_world_to_ref": Tensor,         # [4, 4]
    },
    "selection": dict,
}
```

Training and query behavior:

- Fusion/head receives memory features and reference metadata.
- The regressor predicts world-frame scene coordinates.
- PnP / DSAC consumes world-frame coordinates directly.

### C1: Reference-Coordinate Learning

Enable only after Q1 validation supports Assumption A2.

Semantics:

- MapAnything memory features are treated as approximately reference-camera
  frame equivariant.
- The network output is `P_ref` or `P_ref_norm`.
- If output is `P_ref_norm`, recover `P_ref` first.
- Always recover world coordinates before PnP / DSAC:

```text
P_world = R_ref * P_ref + C_ref
```

Required memory fields:

```python
memory = {
    "mode": "single_forward" | "cluster_local",
    "contract_mode": "C1",
    "features_conditioned": Tensor,       # [N, D]
    "points_ref": Tensor,                 # [N, 3], primary learning target
    "points_ref_norm": Tensor,            # [N, 3], optional normalized target
    "points_world": Tensor,               # [N, 3], PnP/debug target
    "conditioning_reference": {
        "reference_index": int,
        "T_ref_c2w_world": Tensor,        # [4, 4]
        "T_world_to_ref": Tensor,         # [4, 4]
    },
    "normalization_ref": {
        "mu_ref": Tensor,                 # [3]
        "sigma_ref": float,
    },
    "selection": dict,
}
```

First-version loss policy:

- Predict `P_ref` or `P_ref_norm`.
- Recover to `P_world`.
- Use the existing world-frame reprojection loss path.

Do not introduce normalized-pose reprojection in the first C1 prototype. That
would add a second variable and make Q1 harder to interpret.

## 3. Phase 0: Instrumentation and Schema Preparation

Goal: make the current system observable without changing its behavior.

Primary implementation area:

- `memory_extraction/run_memory_extraction.py`

Add structured reporting around the existing view selection, MapAnything infer,
PoseEval, and memory save flow.

Required report file:

```text
memory_policy_report.json
```

Required report sections:

```python
{
    "schema_version": "policy_report_v1",
    "selection": {
        "mode": str,
        "selected_view_ids": list[int],
        "reference_index": int | None,
        "memory_index_groups": list[list[int]] | None,
    },
    "coverage_stats": {
        "mean": float | None,
        "p95": float | None,
        "max": float | None,
    },
    "topology_stats": {
        "num_components": int | None,
        "isolated_count": int | None,
        "degree_min": int | None,
        "degree_median": float | None,
    },
    "reference_risk_stats": {
        "ref_dist_q50": float | None,
        "ref_dist_q90": float | None,
        "ref_dist_max": float | None,
        "far_view_count": int | None,
    },
    "probe_stats": {
        "trans_q90": float | None,
        "trans_max": float | None,
        "pose_eval_ok": bool | None,
    },
    "policy": {
        "decision": "baseline" | "single_forward" | "cluster_branch" | None,
        "reason_codes": list[str],
    },
    "stop": {
        "stop_reason": str | None,
        "final_view_count": int,
    },
}
```

Implementation notes:

- Prefer small helper functions for stats computation.
- Keep report generation tolerant of missing covisibility or PoseEval data.
- Do not fail memory extraction only because a diagnostic stat is unavailable.
- Include all relevant thresholds and config values in the report.

Acceptance criteria:

- Existing memory extraction commands still run.
- Generated memory files remain compatible with current training.
- `memory_policy_report.json` is produced for every run.
- FPS, current ASB, and adaptive ASB can be compared offline through the report.

## 4. Phase 1: Q1 Contract Validation

Goal: decide whether the system should default to `C0` or can safely enable
`C1`.

### Q1a: Reference-Swap Diagnostic Without Training

Purpose: detect whether changing the reference causes uncontrolled feature or
pose-consistency drift before implementing a full predictor.

Procedure:

1. Fix one selected view set for a scene.
2. Choose two or more reference indices from that same selected set.
3. Run MapAnything infer for each reference setting.
4. Align each predicted reference pose to GT as the current PoseEval path does.
5. Compare residual distributions.

Metrics:

- PoseEval translation residual `q50`, `q90`, `max`.
- PoseEval rotation residual `q50`, `q90`, `max`.
- Difference in selected-view residual distributions across references.
- Optional feature distribution drift if reliable feature correspondences exist.

Do not use `points_ref` round-trip as a main metric. That is a geometry identity
test and does not validate feature-frame equivariance.

Acceptance criteria:

- A reference-swap diagnostic script or mode can run on one scene.
- It writes a JSON report with metrics for each reference choice.
- It identifies whether reference changes create large consistency drift.

### Q1b: Minimal Predictor C0 vs C1

Purpose: compare the two contract modes with the smallest predictor that can
exercise memory-conditioned output.

Procedure:

1. Use the same scene and selected view set as Q1a.
2. Build one memory under C0 and one memory under C1.
3. Train or run a minimal predictor for both modes.
4. Convert outputs to world frame.
5. Compare world-frame coordinate and pose consistency.

Metrics:

- `mean ||P_world_C0 - P_world_GT||`
- `mean ||P_world_C1 - P_world_GT||`
- `mean ||P_world_ref_a - P_world_ref_b||` after recovery.
- PnP pose difference.
- DSAC inlier overlap.
- Validation relocalization metrics.

Decision rule:

- If C1 is unstable across reference swaps, default to C0.
- If C1 does not clearly improve or simplify the downstream result, default to C0.
- Enable C1 only when it is both stable and measurably useful.

Acceptance criteria:

- `contract_mode_default` is recorded as `C0` or `C1`.
- The decision is backed by Q1a and Q1b reports.
- Later phases read this decision instead of hardcoding C1.

## 5. Phase 2: Single-Forward Selector Upgrade

Goal: upgrade the single-forward path while preserving a fixed-K fallback.

Primary implementation area:

- Existing `anchor_support_select_views()` path in
  `memory_extraction/run_memory_extraction.py`.

New selector behavior:

- Accept `min_views` and `max_views`.
- Maintain selected-set coverage, topology, and reference-risk stats.
- Add views greedily using reference-risk-aware TA-ASB.
- Stop once the selected set is single-reference-safe.

Greedy score:

```text
Score(v | S)
  = w_c * DeltaCoverage(v)
  + w_s * DeltaSupport(v, S)
  + w_t * DeltaTopology(v, S)
  - w_r * ReferenceRisk(v, S)
```

First-version online score must not include heavy MapAnything probe. Probe is
reserved for Phase 3.

Stop rule:

```python
if (
    len(S) >= min_views
    and coverage_ok
    and topology_ok
    and ref_risk_ok
    and marginal_gain_saturated
):
    stop
```

Required stop metadata:

- `stop_reason`
- `final_view_count`
- coverage stats at stop
- topology stats at stop
- reference-risk stats at stop
- recent marginal gain trace

Repair policy:

1. Connectivity repair.
2. Coverage-hole repair.
3. All repair candidates must pass topology and reference-risk gates.

Acceptance criteria:

- Existing fixed-K behavior remains available.
- Auto-stop can select fewer than `max_views`.
- Every run records why it stopped.
- Small-scene memory construction still uses a single MapAnything forward.

## 6. Phase 3: Two-Level Feasibility Gate

Goal: decide whether a scene has a feasible single reference-local domain.

### Top-M Reference Candidates

Generate a small candidate set from the union of:

- nearest views to mean camera center;
- highest average covisibility;
- high graph centrality;
- low reference-risk candidates.

Default target:

```text
M = 4 to 8
```

### Light Gate

Use only cheap geometry/graph statistics.

Checks:

- `coverage_potential_ok`
- `topology_potential_ok`
- `ref_risk_potential_ok`

Example conditions:

- `ref_dist_q90 <= tau_ref_q90_light`
- `supportable_ratio >= tau_support`
- graph not excessively fragmented
- bounded FPS/coverage estimate is not obviously failing

### Probe Gate

Run only on candidates that pass the light gate.

Use bounded selector probe:

- selected set size at most `max_views_probe`;
- simplified TA-ASB;
- no final memory construction unless the candidate is chosen.

Probe metrics:

```python
probe_ok = (
    trans_q90 < tau_probe_q90
    and trans_max < tau_probe_max
)
```

Important constraint:

- This is an offline memory-construction policy.
- It assumes reference/training poses are available.
- It is not a query-time online policy.

Policy decision:

```python
if any(candidate satisfies coverage_ok, topology_ok, ref_risk_ok, probe_ok):
    decision = "single_forward"
    selected_reference = best_candidate
else:
    decision = "cluster_branch"
```

Required output:

```text
policy_decision.json
```

It must include:

- candidate list;
- light gate result per candidate;
- probe gate result per candidate;
- selected candidate if any;
- reason codes;
- runtime cost for light gate and probe gate.

Acceptance criteria:

- Can run policy decision without building clustered memories.
- Can compare light-gate-only and light-plus-probe policies.
- Decision and reason codes are reproducible from the saved report.

## 7. Phase 4: Cluster Fallback

Goal: provide a controlled fallback when no feasible single domain exists.

Cluster planner:

- Start with all views in one cluster.
- Split the largest unsafe cluster.
- Use k=2 clustering on camera centers.
- Continue until clusters are feasible or a hard stop is reached.

Hard stop conditions:

- `max_num_clusters`
- `max_split_depth`
- `min_cluster_size`
- `min_views_per_cluster`

Cluster feasibility:

- First version should use light feasibility during the split tree.
- Avoid running full heavy probe at every intermediate split node.
- For final clusters, run one selected-reference probe before memory construction.

Unsplittable clusters:

- Mark as `low_confidence`.
- Build best-effort cluster-local memory.
- Save `unsplittable_failure_reason`.

Cluster memory schema:

Each cluster-local memory must be schema-compatible with a single memory:

```python
cluster_memory = {
    "mode": "cluster_local",
    "cluster_id": int,
    "contract_mode": "C0" | "C1",
    "features_conditioned": Tensor,
    "points_world": Tensor,
    "points_ref": Tensor | None,
    "points_ref_norm": Tensor | None,
    "conditioning_reference": dict,
    "selection": dict,
    "cluster_metadata": {
        "view_ids": list[int],
        "cluster_center_world": list[float],
        "low_confidence": bool,
        "failure_reason": str | None,
    },
}
```

Clustered package:

```python
memory_package = {
    "mode": "clustered",
    "contract_mode": "C0" | "C1",
    "clusters": list[cluster_memory],
    "policy": dict,
}
```

Acceptance criteria:

- Cluster split cannot recurse indefinitely.
- Low-confidence clusters are explicit, not silent failures.
- Each cluster builds memory using the same single-domain constructor.
- Clustered memory package preserves all reference transforms needed for
  query-time recovery.

## 8. Phase 5: Query-Time Relocalization Interface

Goal: make single and clustered memory packages usable by evaluation.

Single mode:

1. Compute query backbone feature once.
2. Fuse with the single memory.
3. If C0, output `P_world`.
4. If C1, output `P_ref` or `P_ref_norm`, recover to `P_world`.
5. Run PnP / DSAC.
6. Return world-frame pose.

Clustered mode:

1. Compute query backbone feature once.
2. For each cluster memory:
   - run independent prediction;
   - if C0, output `P_world`;
   - if C1, recover `P_ref_i` to `P_world`;
   - run PnP / DSAC;
   - record score and inlier count.
3. Select final pose by DSAC score / inlier count.

Explicitly forbidden in first version:

- learned query routing;
- cross-cluster feature fusion;
- per-cluster query MapAnything forward.

Required evaluation log:

```python
{
    "mode": "single_forward" | "clustered",
    "contract_mode": "C0" | "C1",
    "cluster_results": [
        {
            "cluster_id": int,
            "pose_score": float,
            "inlier_count": int,
            "selected_as_final": bool,
        }
    ],
    "final_pose_source": "single" | "cluster:<id>",
}
```

Acceptance criteria:

- Single and clustered paths both output a pose in dataset world frame.
- Clustered path can explain which cluster produced the final pose.
- Query feature is computed once per query.

## 9. Evaluation Definitions

### Single-Domain Feasible

A selected single memory is feasible if it reaches the predefined validation
threshold on the target validation split.

Default metrics to report:

- Pose Recall @ 5cm/5deg.
- Pose Recall @ 10cm/5deg.
- median translation error.
- median rotation error.
- DSAC inlier count.

### False Positive

The gate predicts `single_forward`, but the selected single memory fails the
feasibility threshold.

### False Negative

The gate predicts `cluster_branch`, but the best single candidate reaches the
feasibility threshold.

### Cluster Necessary

The single path fails the feasibility threshold and the clustered path reaches
it.

### Cluster Unnecessary

The single path reaches the feasibility threshold but policy still chooses
clustered mode.

These definitions must be used for Q2 and Q4. Do not use subjective failure
labels.

## 10. Test Plan

### Sanity Tests

- `T_world_to_ref @ T_ref_c2w_world` is identity within tolerance.
- `P_ref -> P_world -> P_ref` round-trip error is below tolerance.
- C0 memory can be loaded by current world-coordinate training/evaluation path.
- C1 memory recovers world coordinates before PnP / DSAC.
- Cluster memory schema matches single memory schema plus cluster metadata.

### Integration Tests

- Existing extraction command produces a memory and `memory_policy_report.json`.
- Auto-stop selector produces a valid single memory.
- Two-level gate can run in decision-only mode.
- Cluster fallback creates a bounded number of cluster-local memories.
- Query-time interface evaluates both single and clustered memory packages.

### Research Tests

- Q1: C0 vs C1 contract validation.
- Q2: feasibility gate false positive / false negative.
- Q3: fixed-K vs auto-stop.
- Q4: always single vs always clustered vs auto-policy.

## 11. Recommended Implementation Order

1. Phase 0: instrumentation and schema preparation.
2. Phase 1: Q1a and Q1b contract validation.
3. Phase 2: single-forward selector upgrade.
4. Phase 3: two-level feasibility gate.
5. Phase 4: cluster fallback.
6. Phase 5: query-time ensemble interface.

Do not implement cluster fallback before Q1 and the single-forward selector are
usable. The core dependency is contract validity, not clustering.

