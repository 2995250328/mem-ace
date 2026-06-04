# Review Report：SquareBench Stage2 Global Reliability 实验分析

## 1. 执行摘要

SquareBench recovery experiments 表明，`wayspots_squarebench` 是一个有效的 negative-transfer stress case，而不是数据、memory 或 Stage1 损坏导致的失败。local 路线本身很强：ACE 已经有竞争力，ACE-FCN-LMC Stage1 在 Acc5 上略高于 ACE，并且 ACE-FCN memory sanity checks 通过。失败发生在 Stage2 通过 `glace_concat` 注入 GLACE global features 时。

核心发现是 scene-dependent global reliability：

- 在 Bears 上，raw Stage2 GLACE global conditioning 将 Acc5 从 Stage1 的 77.070 提升到 83.280。
- 在 SquareBench 上，同样的 raw Stage2 conditioning 将 Acc5 从 Stage1 的 53.030 降到 32.580。

因此，问题不只是注入多少 global feature。真正问题是当前模型缺少判断 GLACE global information 何时可信的可靠机制。

## 2. 来源与指标说明

主要来源：

```text
stage2_global_reliability/02_review/squarebench_recovery_log.md
```

日志说明，除非额外注明，固定参考数字都是 post-train aggregated。底层 run directories 的 main suite root 是：

```text
/data/xwh/ace_dinov2_lmc/04_evaluation/wayspots_ace_fcn_lmc_suite/20260530_181745
```

日志列出的重要 SquareBench result directories：

```text
stage2_global_gate_matrix_squarebench_gatefix
stage2_global_gate_matrix_squarebench_gatefix_bounded
stage2_global_gate_matrix_squarebench_gatefix_bounded01
stage2_global_reliability_squarebench_cons01
stage2_global_reliability_squarebench_cons1
stage2_global_reliability_squarebench_guard01
stage2_global_reliability_squarebench_guard1
stage2_reliability_select_squarebench_safe
stage2_residual_identity_check_squarebench
stage2_residual_unified_squarebench_basefix_smoke
stage2_residual_unified_squarebench_basefix_it12
stage2_film_squarebench_it12
stage2_film_squarebench_controls_it12
stage2_concat_gate_l1_squarebench_it12
```

日志列出的重要 Bears comparison directories：

```text
stage2_global_gate_matrix_bears_gatefix
stage2_global_gate_matrix_bears_gatefix_bounded
stage2_global_gate_matrix_bears_gatefix_bounded01
stage2_global_reliability_bears_cons01
stage2_global_reliability_bears_cons1
stage2_global_reliability_bears_guard01
stage2_global_reliability_bears_guard1
stage2_reliability_select_bears_strong
stage2_residual_unified_bears_basefix_it12
stage2_film_bears_it12
stage2_concat_gate_l1_bears_it12
```

指标包括 Acc50、Acc25、Acc10、Acc5，以及 median rotation/translation error。Acc 越高越好；median error 越低越好。

## 3. 固定参考结果

| Scene | Method | Acc50 | Acc25 | Acc10 | Acc5 | Median |
|---|---|---:|---:|---:|---:|---|
| SquareBench | ACE | 99.653 | 99.480 | 69.844 | 52.166 | 0.689 deg / 4.816 cm |
| SquareBench | GLACE | 94.974 | 94.801 | 67.244 | 42.634 | 0.654 deg / 5.855 cm |
| SquareBench | ACE-FCN-LMC Stage1 | 99.827 | 99.310 | 70.020 | 53.030 | 0.655 deg / 4.786 cm |
| SquareBench | Stage2 raw `glace_concat` | 78.683 | 78.680 | 49.740 | 32.580 | 0.768 deg / 10.098 cm |
| Bears | ACE | 97.931 | 95.172 | 86.207 | 73.276 | 1.111 deg / 3.506 cm |
| Bears | GLACE | 97.414 | 94.310 | 87.069 | 77.069 | 1.011 deg / 3.080 cm |
| Bears | ACE-FCN-LMC Stage1 | 97.069 | 94.830 | 91.030 | 77.070 | 1.059 deg / 3.214 cm |
| Bears | Stage2 raw `glace_concat` | 98.103 | 97.760 | 93.280 | 83.280 | 0.946 deg / 3.072 cm |

关键差值：

- SquareBench raw Stage2 vs Stage1：Acc5 下降 20.450 点。
- Bears raw Stage2 vs Stage1：Acc5 提升 6.210 点。

这种相反行为就是 global reliability 问题的核心。

## 4. 实验矩阵总结

| Family | Variant | SquareBench Acc25 / Acc10 / Acc5 | Bears Acc25 / Acc10 / Acc5 | Outcome |
|---|---|---:|---:|---|
| Scalar gate | `glace_g001_learn` | 77.99 / 49.39 / 32.76 | 97.07 / 92.93 / 81.72 | Bears 尚可，但 SquareBench 仍崩。 |
| Scalar gate bounded | `glace_g001_max001_learn` | 99.83 / 70.19 / 51.65 | 95.17 / 90.86 / 79.48 | SquareBench 安全，但 Bears 收益弱。 |
| Scalar gate bounded | `glace_g001_max01_learn` | 91.51 / 59.10 / 37.44 | 96.21 / 92.07 / 82.41 | Bears 更好，但仍伤害 SquareBench。 |
| Consistency | `glace_g001_max01_cons01` | 85.79 / 55.63 / 35.70 | 98.62 / 93.10 / 82.24 | 没有保护住 SquareBench。 |
| Consistency | `glace_g001_max01_cons1` | 87.69 / 57.54 / 36.05 | 97.24 / 91.90 / 82.59 | SquareBench 仍然差。 |
| Guard loss | `glace_g001_max01_guard01` | 89.08 / 53.38 / 32.41 | 97.41 / 92.76 / 82.24 | Guard loss 没阻止 negative transfer。 |
| Guard loss | `glace_g001_max01_guard1` | 89.60 / 57.89 / 38.65 | 97.24 / 93.10 / 83.45 | Bears 好，SquareBench 坏。 |
| Manual select | SquareBench safe max=0.001, Bears strong max=0.1 | 99.65 / 71.58 / 48.70 | 97.59 / 92.41 / 81.90 | 人工 per-scene selection 可用，但不 paper-clean。 |
| Residual identity | `glace_residual_identity` | 99.48 / 70.36 / 52.69 | n/a | Identity path 匹配 Stage1，证明 wrapper 安全。 |
| Residual unified | smoke / basefix | 99.48 / 70.54 / 52.51 | n/a | 安全，但没有真正 global benefit。 |
| Residual unified | it12 | 99.31 / 69.67 / 53.03 | 96.03 / 90.69 / 78.10 | 退化为 local/Stage1 行为；global 基本未被使用。 |
| FiLM | `glace_film_unified` | 99.83 / 70.88 / 50.61 | 95.17 / 89.66 / 77.93 | 比较安全，但 Bears 弱，低于目标。 |
| FiLM control | `random_film` | 99.65 / 69.84 / 48.18 | n/a | 与 GLACE FiLM 接近，缺少 GLACE semantics 被使用的证据。 |
| FiLM control | `zero_film` | 99.31 / 70.36 / 50.78 | n/a | 与 GLACE/random FiLM 接近，强化 degeneration 结论。 |
| Concat gate + L1 | `glace_concat_gate_unified` | 86.66 / 57.37 / 34.66 | 96.90 / 91.03 / 81.55 | Bears 保留部分 global 收益，但 SquareBench 仍崩。L1 不是统一解。 |

## 5. 按实验族分析

### 5.1 Raw `glace_concat`

Raw Stage2 是最初失败模式。它在 Stage1 local stack 上把 GLACE global features concat 到 final head。它改善 Bears，但严重伤害 SquareBench。关键结论是 raw global conditioning 并非总是坏，而是 scene-dependent。

### 5.2 Scalar Gates

Scalar gate 家族证明了 tradeoff。`gate_max=0.001` 这样极小的 cap 能保护 SquareBench，但牺牲 Bears 的较大收益。`gate_max=0.1` 能恢复 Bears，但又重新引入 SquareBench collapse。因此，固定 scalar amplitude cap 不能解决统一 reliability 问题，除非它能从额外信号中学习可靠性。

### 5.3 Consistency and Guard Losses

Stage1 consistency 和 pixel guard losses 的目标是让 Stage2 不要远离 local teacher。但在已测试配置中，它们没有保护 SquareBench。早期 adaptive/reliability variants 还出现过数值不稳定，包括 `Valid: 0.0%`、`naninf=4096` 和 NaN gate diagnostics。当前 consistency/guard route 如果不重新设计，不适合作为最终方法。

### 5.4 Residual Identity and Residual Unified

Residual identity/basefix 实验是重要的实现检查。它们说明，当 residual/global path 被禁用或接近零时，Stage2 wrapper 可以安全复现 Stage1-like behavior。然而 full residual-unified run 基本退化到 local predictor：SquareBench 安全，但 Bears 失去 global benefit。该架构解决了 safety 问题，但没有解决 usefulness 问题。

### 5.5 FiLM and FiLM Controls

FiLM 路径通过 zero-initialized adapter 调制 frozen local representation，而不是 raw concat。它比 raw concat 更能保护 SquareBench，但 random 和 zero controls 接近，而且 Bears 表现弱。这说明当前 FiLM 实现更多是 conservative control path，而不是有效的 GLACE semantic-conditioning path。

### 5.6 Concat Gate with L1

最新 `glace_concat_gate_unified` 实验回到 concat，但加入 learnable bounded scalar gate 和显式 L1 regularization：

```bash
--ace_lmc_global_gate_init 0.01
--ace_lmc_global_gate_learnable True
--ace_lmc_global_gate_max 0.1
--ace_lmc_global_gate_l1_weight 0.001
```

完成结果：

| Scene | Acc25 | Acc10 | Acc5 | Median |
|---|---:|---:|---:|---|
| Bears | 96.90 | 91.03 | 81.55 | 0.91 deg / 2.75 cm |
| SquareBench | 86.66 | 57.37 | 34.66 | 0.71 deg / 7.97 cm |

它保留了 Bears 的中等提升，但 SquareBench 仍远低于 Stage1 和 ACE。weak scalar sparsity prior 不足；单个 learnable bounded scalar gate 仍然会允许 harmful GLACE conditioning 进入模型。

## 6. 已失败或不足的路线

当前形式下，不建议继续大规模 sweep 以下路线：

1. **Single scalar gate 或 scalar gate max sweeps。** 人工 per-scene selection 有效，但不是 principled paper method。
2. **简单 Stage1 consistency 或 guard loss。** 已测试 losses 不能阻止 SquareBench collapse，且早期 variant 有不稳定记录。
3. **Conservative residual 或 FiLM 作为最终方法。** 它们主要通过抑制 global information 来保护 SquareBench，因此也移除了 Bears 的原始收益。
4. **Concat gate with weak L1。** 它保留部分 Bears 收益，但 SquareBench 仍严重失败。

这些路线仍然适合作为 ablations，用来说明 reliability-aware global conditioning 的必要性。

## 7. 科学解释

这些实验支持以下 paper-facing interpretation：

1. SquareBench 是真实的 negative-transfer stress case。
2. GLACE global information 在 Bears 有帮助，但在 SquareBench 有害。
3. ACE-FCN-LMC Stage1 是强且稳定的 local memory route。
4. Raw GLACE global concat 不能作为安全统一方法。
5. 现有 scalar 或 conservative controls 没有解决核心 reliability 问题。

因此正确研究问题不是普通 feature fusion，而是 scene- 或 sample-dependent global reliability estimation。

## 8. 推荐下一步方向

### 8.1 Local-Global Disagreement Gate

使用 local-only 和 global-conditioned predictions 之间的分歧作为 reliability signal。候选信号包括：

- Stage1 local reprojection error。
- Stage1 coordinates 与 Stage2 coordinates 的差异。
- Global-conditioned residual norm。
- Per-pixel reprojection delta。
- Local-only validity 或 confidence。

如果 global conditioning 与 confident local predictor 强烈分歧，就降低或拒绝 global path。

### 8.2 Dual-Head Selection or Mixture

保留两条预测路径：

1. Local-only Stage1/Stage2 head。
2. Local+global head。

然后用 learned 或 non-learned reliability rule 进行选择或混合。选择可以是 per-scene、per-image 或 per-pixel。这直接匹配 Bears/SquareBench 分裂现象，避免强迫所有 scene 使用同一种 fusion behavior。

### 8.3 RANSAC/Post-Hoc Reliability Selection

对每个 query 同时运行 local-only 和 local+global predictions，然后用 SCR 原生 confidence signals 选择 pose：

- RANSAC inlier count。
- Pose score。
- Reprojection residual。
- Coordinate validity ratio。

这可能是最 practical 的下一步实验。它与 SCR pipeline 自然一致，可能在保护 SquareBench 的同时保留 Bears 收益，代价是推理时间增加。

## 9. 对最终论文的影响

SquareBench 不应被隐藏。它是有价值的 reviewer-facing negative-transfer analysis。它证明：

- Stage1 local memory 很强。
- memory/data path 没坏。
- 当 global priors 弱时，global conditioning 可能有害。
- paper-clean Stage2 方法需要 reliability control。

如果最终方法没有稳健解决 reliability，raw Stage2 `glace_concat` 应作为 ablation，而不是 unified method。更稳的 narrative 是：

> LMC Stage1 提供稳定 local memory compressor。GLACE global conditioning 在可靠时可以提供额外收益，但需要显式 reliability selection 或 gating 来避免 negative transfer。

## 10. 推荐短期实验列表

1. **Dual inference with RANSAC inlier selection。** 在 Bears 和 SquareBench 上逐 query 比较 local-only vs local+global。
2. **Non-learned local-global disagreement gate。** 先用 coordinate delta、residual norm 或 reprojection delta 作为 reliability criterion，再考虑 learnable complexity。
3. **Per-image gate instead of scene scalar gate。** Scene-level scalar gates 对 paper-clean solution 来说太粗。
4. **Ablation consolidation。** 保留 scalar gate、consistency、residual、FiLM 和 concat-gate-L1 结果，作为 reliability 必要性的证据。

## 11. Stage-2 Review Verdict

本 review 否定 scalar-amplitude control 作为最终 reliability method。下一阶段 architecture 应设计基于 local-global disagreement、dual-head selection 或 RANSAC/post-hoc pose selection 的 reliability 机制。目标是在保留 Bears 有效 global gains 的同时，把 SquareBench 恢复到 Stage1-level 或更好。