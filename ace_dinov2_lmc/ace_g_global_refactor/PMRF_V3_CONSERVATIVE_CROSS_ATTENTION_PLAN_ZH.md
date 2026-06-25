# PMRF-v3 保守式 Cross-Attention Adapter 冻结总规划

Date: 2026-06-22
Workspace: `/home/xwh/project/ace_depth/ace_dinov2_lmc`

Authoritative workspace plan:
`ace_g_global_refactor/PMRF_V3_CONSERVATIVE_CROSS_ATTENTION_PLAN_ZH.md`

The workspace document is the source of truth. This memory note exists to recover the plan across
sessions and should not fork into an independent design.

## Frozen direction

- Preserve the existing single fusion exactly as first read.
- Implement only a conservative second-read cross-attention adapter.
- Reuse first-read K/V and use scene-aware H1 only to update query routing.
- Use per-head QK L2 normalization with learnable positive temperature initialized at sqrt(head_dim).
- Preserve centered identity with `out_proj(C2) - out_proj(C1)`; never use a biased
  `out_proj(C2-C1)` unless that projection is explicitly bias-free.
- Decompose Delta into patch-varying and patch-common parts.
- Use per-channel LayerScale: gamma_patch init 0.01, gamma_common init 0.0.
- Keep `post_norm=False`; do not add a second full FFN/block, assembly MLP, or geometry bias.
- Proposed mode: `centered_reread_qknorm_layerscale`.

## Research rationale

This combines QK Normalization, ReZero/LayerScale residual control, head-wise scaling ideas from
NormFormer, and iterative cross-attention patterns from Perceiver/DETR. It is not claimed as a
direct reproduction of any one prior method.

## Execution order

1. Finish exact legacy single K64 reproduction for current code and commit
   `65789847cb0c8f3b0f5cc5450c6f19844b275b90` before new PMRF-v3 training.
2. Implement compatibility and diagnostics in ace_fusion/options/trainer/evaluator.
3. Verify single numerical non-regression, centered zero-delta identity, gradients, state-dict
   round-trip, and train/eval reconstruction consistency.
4. Run the minimal two-GPU matrix on Bears, Squarebench, Cubes, Tendrils:
   single, pmrf_base, centered_s0, pmrf_v3.
5. Promote to all target Wayspots scenes except Winter Sign and Statue only if primary metrics hold.

## Metrics and workflow constraints

- Only GPUs 0 and 1 are available.
- Long jobs use tmux and GPU placement is checked immediately.
- Train/eval commands run from `/home/xwh/project/ace_depth`.
- Torch/model smoke uses `conda run --no-capture-output -n mapanything python`.
- Aggregate every metric independently over iter/cross/post-train seeds and retain source fields.
- Wayspots decisions prioritize Acc50 and Acc25; Indoor6 prioritizes Acc25 and median errors;
  Cambridge prioritizes median errors.

## Avoid

- No return to `post_norm=True`.
- No geometry reread priority.
- No broad alpha/gate/tau sweeps.
- No conclusion from one iteration log; use `summary.tsv`, post-train evaluations, and sources.
- If v3 fails, test only one fallback at a time: no-QK-norm, scalar gamma, or fixed
  zero common branch. Do not form another hyperparameter matrix.

## Promotion rules

- Improve Acc50/Acc25 with stable medians: promote to the all-scene reproduction.
- Preserve Acc50/Acc25 and improve strict thresholds: retain as a tail-quality candidate.
- Regress Acc50/Acc25 while only Acc10/strict thresholds improve: do not replace Wayspots single.
- If PMRF-base remains more stable across scenes, retain nonorm alpha=0.10 as the main PMRF line.

## References

- QK Normalization: https://arxiv.org/abs/2010.04245
- ReZero: https://arxiv.org/abs/2003.04887
- LayerScale/CaiT: https://arxiv.org/abs/2103.17239
- NormFormer: https://arxiv.org/abs/2110.09456
- Perceiver IO: https://arxiv.org/abs/2103.03206
- DETR: https://arxiv.org/abs/2005.12872

## 当前前置决策计划

baseline 恢复判据、v2 双旧版重训和后续结构分支见：

[`NEXT_AFTER_LEGACY_SINGLE_RECOVERY_PLAN_ZH.md`](./NEXT_AFTER_LEGACY_SINGLE_RECOVERY_PLAN_ZH.md)
