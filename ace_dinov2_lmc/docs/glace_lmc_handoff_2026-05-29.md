# GLACE-LMC Memory Fusion Handoff - 2026-05-29

本文档用于新对话接手当前 GLACE-LMC / memory fusion 实验。重点只记录和 `memory 融合进 GLACE` 相关的代码、实验、结论和下一步建议。

## 1. 当前目标

原始设想：

```text
vanilla GLACE scene head 初始化
+ GLACE encoder / global feature source 冻结
+ LMC 只融合 local feature
+ global feature bypass
+ local residual / gate 控制 memory delta
+ head 可选共同适配
```

核心希望验证：压缩 memory token 是否能通过 local feature 增强 GLACE 的 scene coordinate regression。

## 2. 当前代码状态

已实现：

- `--lmc_fusion_target decoder|local`
  - `decoder`: 旧路径，fusion query 是完整 `concat(global, local)`。
  - `local`: 新路径，只对 local feature 做 LMC fusion，global bypass 后拼回 GLACE head input。
- `--local_residual_mode none|fixed_alpha`
  - `none`: `local_out = local_fused`，即 direct replace。
  - `fixed_alpha`: `local_out = local + alpha * (local_fused - local)`。
- `--local_residual_alpha`
  - 要求 `[0, 1]`。
- `--glace_freeze_encoder`
  - 默认 `True`。
- `--glace_freeze_head`
  - 默认 `None`，不显式传时沿用旧的 `--glace_freeze_base_network`。
  - 显式 `False` 时，head 会进入 S1/S2 optimizer。
- eval 路径已同步 local-only/fixed-alpha 逻辑。
- checkpoint `lmc_config` / resume strict config / eval summary 已记录 fusion target、local residual、freeze policy。

关键代码位置：

- `options_dinov2_lmc.py`
  - `--lmc_fusion_target`
  - `--local_residual_mode`
  - `--local_residual_alpha`
  - `--glace_freeze_encoder`
  - `--glace_freeze_head`
- `trainer_dinov2_lmc.py`
  - `_get_glace_global_features()`
  - `_split_glace_decoder_feature_maps()`
  - `_build_glace_head_input()`
  - `_fuse_lmc_features_for_head()`
  - `_glace_freeze_encoder()`
  - `_glace_freeze_head()`
- `test_ace_dinov2_lmc.py`
  - eval local-only split/fuse/rebuild logic

验证过的静态检查：

```bash
python -m py_compile options_dinov2_lmc.py trainer_dinov2_lmc.py test_ace_dinov2_lmc.py train_ace_dinov2_lmc.py
git diff --check -- options_dinov2_lmc.py trainer_dinov2_lmc.py test_ace_dinov2_lmc.py docs/glace_local_fusion_experiment_plan.md docs/glace_lmc_review_map.md
```

## 3. 重要评测流程说明

训练中目前可能出现两类 eval：

```text
iteration eval:
  trainer_dinov2_lmc.py::_evaluate_checkpoint()
  --eval_each_iteration True 时触发
  hypotheses=64, 单 seed
  速度较快

post-train eval:
  train_ace_dinov2_lmc.py::run_post_train_eval()
  --eval_after_train True 时触发
  memory_compare_ace_g_v2 默认 seeds=1305,2026,4242 且 hypotheses=256
  等价于 3 次重评测，明显更慢
```

快速筛方案时建议加：

```bash
--eval_after_train False
```

如果仍想保留最终 eval 但加速，用：

```bash
--post_train_eval_seeds 1305 --post_train_hypotheses 64
```

## 4. 实验结果汇总

统一场景：

```text
scene: /data/xwh/Wayspots/wayspots_bears
memory: /home/xwh/project/ace_depth/ace_dinov2_lmc/04_evaluation/wayspots_lmc/memory/wayspots_bears/32v_v0.05_bilinear_bse_ut0.02_noaug_asb_adaptive_pool1.0_rgate4_prepair8_sor_gm_l2/20260525_223355/memory_bse.pt
GLACE init head: /home/xwh/project/ace_depth/ace_dinov2_lmc/04_evaluation/wayspots_baselines/20260525_190055/wayspots_bears/glace/model.pt
eval: seeds=1305,2026,4242, hypotheses=256, aggregation=median
```

### Baseline / compatibility

Step1a decoder compat:

```text
Median 1.00 deg / 3.04 cm
25cm/5deg 95.00%
10cm/5deg 89.14%
5cm/5deg 78.79%
2cm/2deg 14.14%
1cm/1deg 1.21%
```

Step2 decoder baseline:

```text
Median 1.00 deg / 3.07 cm
25cm/5deg 95.00%
10cm/5deg 89.14%
5cm/5deg 77.59%
2cm/2deg 14.48%
1cm/1deg 0.86%
```

### Local-only direct replace

Step2 local-only direct replace:

```text
Median 29.45 deg / 116.07 cm
25cm/5deg 0.52%
10cm/5deg 0.17%
5cm/5deg 0.00%
2cm/2deg 0.00%
1cm/1deg 0.00%
```

结论：`local_out = local_fused` 直接崩，不可继续。

### Fixed alpha local residual, head frozen

Step3 alpha=0.2:

```text
Median 69.10 deg / 159.06 cm
全部主阈值 0.00%
```

Step3 alpha=0.1:

```text
Median 1.06 deg / 3.40 cm
25cm/5deg 94.14%
10cm/5deg 85.86%
5cm/5deg 73.10%
2cm/2deg 8.97%
1cm/1deg 0.17%
```

Step3 alpha=0.03:

```text
Median 1.04 deg / 3.15 cm
25cm/5deg 94.48%
10cm/5deg 86.90%
5cm/5deg 75.86%
2cm/2deg 11.21%
1cm/1deg 0.52%
```

Step3 alpha=0.015:

```text
Median 0.99 deg / 3.09 cm
25cm/5deg 94.31%
10cm/5deg 88.10%
5cm/5deg 76.03%
2cm/2deg 13.79%
1cm/1deg 1.21%
```

Step3 alpha=0.01:

```text
Median 0.99 deg / 2.97 cm
25cm/5deg 94.83%
10cm/5deg 88.45%
5cm/5deg 77.24%
2cm/2deg 15.00%
1cm/1deg 0.86%
```

Step3 alpha=0.005:

```text
Median 0.99 deg / 3.02 cm
25cm/5deg 95.17%
10cm/5deg 88.62%
5cm/5deg 76.55%
2cm/2deg 14.66%
1cm/1deg 1.55%
```

结论：

- 可用 alpha 极小，大约 `0.005 ~ 0.01`。
- 小 alpha 有零星高精度指标波动，但主指标没有稳定超过 decoder baseline。
- `alpha >= 0.03` 明显劣化，`alpha >= 0.2` 崩。
- 继续精细扫 alpha 没有价值。

### Head co-adaptation

Step4 head trainable, alpha=0.01:

```text
Median 1.06 deg / 3.23 cm
25cm/5deg 95.00%
10cm/5deg 87.07%
5cm/5deg 75.86%
2cm/2deg 9.66%
1cm/1deg 1.21%
```

Step4 head trainable, alpha=0.005:

```text
Median 1.04 deg / 3.32 cm
25cm/5deg 94.66%
10cm/5deg 85.86%
5cm/5deg 77.24%
2cm/2deg 10.69%
1cm/1deg 1.03%
```

结论：head 解冻没有救回 local memory fusion，整体还更差。

## 5. 总体结论

目前这条主线不成立：

```text
LMC compressed memory -> local feature delta -> GLACE head
```

已验证：

1. full direct replace 崩。
2. 中等 alpha 明显劣化或崩。
3. 很小 alpha 只能产生弱小、局部、不可稳定扩展的指标波动。
4. head co-adaptation 后没有改善，反而更差。

最可能原因：

```text
local_fused - local_raw 的方向大多不在 GLACE head 可解释的 feature manifold 上。
```

这不是继续调 alpha 或训练轮数能解决的问题。

## 6. 不建议继续做的事

不要继续：

- 继续扫 `alpha=0.006/0.0075/0.012` 之类的小数。
- 继续跑 `local_residual_mode none` direct replace。
- 继续用同一结构加大训练步数或 iteration 数。
- 继续通过 guard loss 把输出拉回 vanilla，因为这会让 memory branch 更接近无效。
- 继续只看最终 pose 指标，不看 feature / pixel-level 诊断。

## 7. 建议下一步

下一步建议从“训练新模型”切换到“诊断 memory delta 是否有用”。优先实现诊断日志或离线分析：

```text
对同一 batch / eval frame 同时计算：
base:  GLACE head(concat(global, local_raw))
fused: GLACE head(concat(global, local_raw + alpha * (local_fused - local_raw)))
```

记录：

- `delta_local_ratio = ||local_out - local_raw|| / ||local_raw||`
- `delta_cos = cos(local_fused - local_raw, local_raw)`
- `err_fused - err_base` 的像素级分布
- 按 base reprojection error 分桶：easy / medium / hard
- attention entropy / effective token number / top-k memory token usage
- 哪些像素变好，哪些像素变坏

目标是回答：

```text
memory delta 是否只在少数 hard pixels 有帮助？
还是大多数区域都在破坏 GLACE local feature？
```

如果诊断显示多数像素变差，应放弃“直接改 local feature”的结构，转向更弱的使用方式：

- memory 只预测 gate / uncertainty / confidence；
- memory 只作用于 hard pixels；
- memory 作为辅助 loss / contrastive alignment，而不是直接改 head input；
- memory 做 retrieval/context conditioning，不直接生成 local feature delta。

## 8. 新对话启动建议

新对话可以直接说：

```text
请阅读 docs/glace_lmc_handoff_2026-05-29.md。
我们已经验证 GLACE-LMC local feature delta 主线效果不好。
不要继续扫 alpha。
下一步做 pixel-level / feature-level 诊断，判断 memory delta 为什么破坏 GLACE head。
```

同时提醒新 agent：

- 当前 worktree 有未提交修改。
- 不要回滚用户已有改动。
- 如果要跑快速训练，建议加 `--eval_after_train False`，避免 post-train 三 seed 256 hypotheses 慢评测。
