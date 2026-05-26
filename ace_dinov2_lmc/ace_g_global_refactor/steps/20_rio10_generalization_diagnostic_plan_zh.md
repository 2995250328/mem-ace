# Step 20：RIO10 泛化问题诊断计划

## 1. 目的

本文档定义如何诊断当前 RIO10 ACE-DINOv2-LMC / ACE-G run 泛化很差的问题。

目标不是加入新架构。目标是在继续投入 PointRoPE、CIFA、multi-level fusion、view token 或 multi-memory routing 之前，先定位失败模式。

当前优先级是：

```text
先诊断 RIO10 泛化问题
再决定架构修改是否有意义
```

如果一个方法只在单个 indoor 场景上很强，但在 RIO10 上失效，那么在 RIO10 失败模式被理解之前，不能把它当成已经验证的方法。

## 2. 当前证据

当前 run 位于：

```text
/home/xwh/project/ace_depth/ace_dinov2_lmc/04_evaluation/train_compare/rio10_conservative_indoor6_migration
```

它应该被视为一个不完整的诊断信号，而不是最终结果。

已观察状态：

```text
configured iterations = 28
observed artifacts/logs stop around iter11
trainer-selected best iteration = iter8
best_metric = pct5
best pct5 = 8.31
best median error = 15.30 deg / 45.39 cm
latest observed iter11 pct5 = 7.93
latest observed iter11 median error = 15.50 deg / 45.67 cm
```

目前 data split 不像是首要问题。

当前证据显示：

```text
ace-g official train -> local train
ace-g official val   -> local test
ace-g official test  -> local hidden_test

current ACE-format train scene = /data/xwh/RIO10_ace/scene01_seq01_01
current train frames = 4380
current eval frames = 2069
```

这与使用 RIO10 scene01 training sequence 训练、使用 public validation sequence 评测是一致的。

当前 memory / sparse-depth 背景是：

```text
memory = 40-view ASB single-forward BSE memory
memory points = 10705
memory feature dim = 3840
sparse depth matched = 4356 / 4380
missing sparse depth = 24
c1_aux_ref_loss_weight = 0.0
buffer_sample_valid_coords = True
buffer_valid_coord_sample_ratio = 1.0
```

最近的 Indoor6 `scene2a/c0_p4` Danchor 控制实验仍然健康。2026-05-23 记录的三组结果中，`Danchor keybias` 最强：

```text
median = 0.28 deg / 2.64 cm
25cm/5deg = 99.61
10cm/5deg = 95.33
5cm/5deg = 80.93
2cm/2deg = 35.80
1cm/1deg = 10.12
```

记录位置：

```text
ace_g_global_refactor/COMPARE_FGPI_Danchor_scene2a_20260523.md
```

这说明 Indoor6 ACE-G 路径本身仍然是健康的，但不能解释或解决 RIO10。RIO10 仍应优先检查 data/eval contract、resize/memory geometry、domain shift 和 sparse-guided sampling。

split 仍然需要做完整 integrity audit，但下一步调查不应该默认差结果只是因为用了错误测试集。

## 3. 主要假设

### H1：Data / Pose / Calibration Contract 问题

即使 split 正确，ACE-format 转换仍可能有问题。

需要审计的风险：

- RGB / pose / calibration 文件独立排序，只检查数量；
- `rgb`、`poses`、`calibration` 的文件 stem 可能没有对齐；
- symlink 可能指向非预期 source；
- pose convention 可能和 DSAC evaluation 预期不同；
- calibration 在 resize 前正确，但 resize 后使用不正确；
- translation scale 可能不是 meters；
- train/test sequence metadata 正确，但某个转换目录可能是 stale 的。

如果这个假设成立，train-set evaluation 可能也会很差。

### H2：RIO10 对 Vanilla DINO ACE Baseline 本身就困难

在责怪 LMC 或 memory 前，必须建立同 split 的非 LMC baseline。

如果 vanilla DINO ACE 在同一个 RIO10 ACE-format scene 上也很差，问题可能是：

- domain shift；
- RIO10 pose/calibration/data 难度；
- DSAC threshold / hypotheses 设置；
- vanilla training recipe 不足；
- ACE-DINOv2 本身在这个 scene 上有限制。

### H3：LMC / Memory Fusion 损害泛化

如果 vanilla DINO ACE 合理，但 ACE-G 很差，则失败大概率由 memory path 引入。

可能原因：

- memory coverage 不能代表 eval trajectory；
- 40 个 ASB views 对 RIO10 scene01 不够；
- BSE memory 太稀疏或有偏；
- memory geometry quality 差；
- S2 fusion 过拟合 memory artifacts；
- S1 alignment 没有学到有用的 memory-conditioned features；
- 3840-dim selected key/value memory contract 在 RIO10 上太脆弱。

### H4：Sparse Guided Sampling 过于激进

当前 run 使用：

```text
buffer_sample_valid_coords = True
buffer_valid_coord_sample_ratio = 1.0
```

这意味着在 frame 有 sparse-depth-valid coordinates 时，所有 buffer sampled positions 都来自这些位置。

考虑到 sparse-depth coverage 有限，这可能降低 sample diversity，并让训练过拟合到很窄的 patch 子集。Sparse depth 仍可能有用，但 ratio `1.0` 是高风险设定。

### H5：Training Parameters 不够保守

当前 run 不只是数据迁移。它还用了几个对 RIO10 可能过于激进的设置：

```text
batch_size = 10240
learning_rate_min = learning_rate_max = 1e-4
s2_learning_rate_max = 1e-3
head_lr_multiplier_s2 = 1.5
lmc_profile = legacy
buffer_sampling_replacement = True
ace_g_fusion_in_s2 = True
```

这些设置可能和 RIO10 数据规模、sparse guided sampling、memory quality 相互作用。

## 4. 诊断阶梯

按下面顺序执行诊断。不要从新架构开始。

### Step 0：把当前结果冻结为 partial

将当前 run 记录为 incomplete：

```text
best known = iter8
source = best_checkpoint_meta.json + eval_summary_scene01_seq01_01_iter_08.txt
status = incomplete / interrupted around iter11
```

不要把它当成完整 28-iteration 结果排名。

### Step 1：Data / Pose / Calibration Integrity Audit

审计 ACE-format scene：

```text
/data/xwh/RIO10_ace/scene01_seq01_01
```

检查：

1. 确认 train/test 的 `rgb`、`poses`、`calibration` 数量；
2. 确认排序后的 filename stem 对齐；
3. 确认 RGB symlink target 指向预期 RIO10/WAI source；
4. 抽样比较 ACE-format poses 和 WAI metadata transforms；
5. 抽样比较 ACE-format calibrations 和 WAI metadata intrinsics；
6. 确认 `dataset_dinov2.py` 使用的 resized intrinsics 与 image resize 行为一致；
7. 检查 train/test camera center 分布和 translation units；
8. 确认 test sequence 对应 public validation，不是 hidden test 或 train leakage。

预期解释：

```text
发现 mismatch -> 先修 data contract，再训练更多模型
没有 mismatch -> 继续 train/eval sanity checks
```

### Step 2：Train-Set Evaluation Sanity Check

在 training split 或 training subset 上评测当前 best checkpoint。

目的：

```text
区分 fit/memorization failure 和 generalization failure
```

解释：

```text
train eval 差 + test eval 差 -> data/eval/optimization failure
train eval 强 + test eval 差 -> generalization/memory/sampling failure
```

这是最高价值的快速诊断。如果 train-set pose accuracy 也很差，不要继续做架构 ablation。

### Step 3：同 Split 的 Vanilla DINO ACE Baseline

训练或查找同 split 的 vanilla baseline：

```text
use_lmc = False
data_backend = ace
scene = /data/xwh/RIO10_ace/scene01_seq01_01
same image_resolution
same evaluation protocol
```

这个 baseline 是解释 ACE-G 的锚点。

解释：

```text
vanilla 差 -> RIO10 baseline/data/eval/training 问题
vanilla 好 -> LMC/memory/fusion/sampling 问题
```

不要只把 ACE-G 和 Indoor6 比。RIO10 需要本地 non-LMC reference。

### Step 4：DSAC / Evaluation Sensitivity Sweep

对同一个 checkpoint 做受控 test-time sweep。

变化参数：

```text
hypotheses: 64, 256, 512
threshold: 5, 10, 20
maybe inlier alpha / max pixel error if exposed
```

解释：

```text
放宽 DSAC 后显著提升 -> coordinates noisy but useful
没有提升 -> coordinate predictions 或 pose/calibration contract 可能错误
```

这一步不用于修改正式 metric，只用于诊断。

### Step 5：Per-Frame 和 Trajectory Diagnostics

按 frame index 和 camera position 绘制或统计 per-frame errors。

关注：

- 所有 frames 都差；
- 只有特定 trajectory segments 差；
- error 随 distance from memory views 增加；
- error 和 sparse-depth missing frames 相关；
- error 和 camera rotation/view direction 相关；
- 重复 pose flip 或类似 scale bias 的 translation error。

解释：

```text
localized failure -> memory coverage / sequence region issue
global failure    -> data/eval/model contract issue
```

### Step 6：Sparse Guided Sampling Ablation

使用 Step19 的同一 model contract，只改变 sampling。

最小矩阵：

```text
A: buffer_sample_valid_coords = False
B: buffer_sample_valid_coords = True, buffer_valid_coord_sample_ratio = 0.25
C: buffer_sample_valid_coords = True, buffer_valid_coord_sample_ratio = 0.50
D: buffer_sample_valid_coords = True, buffer_valid_coord_sample_ratio = 1.00
```

保持：

```text
c1_aux_ref_loss_weight = 0.0
same memory
same seed if possible
same evaluation protocol
```

解释：

```text
A >= D      -> sparse guided sampling 有害或 sparse masks 有偏
B/C > A/D   -> sparse depth 只适合作为 partial prior
D best      -> full sparse-guided sampling 才有依据
```

基于当前证据，ratio `1.0` 在被证明有效之前应视为可疑。

### Step 7：LMC / Fusion Ablation

如果 vanilla 好而 ACE-G 差，隔离 memory path。

最小矩阵：

```text
L0: vanilla DINO ACE, no LMC
L1: ACE-G, no sparse guided sampling
L2: ACE-G, sparse guided sampling ratio 0.25 or 0.50
L3: ACE-G, ace_g_fusion_in_s2 = False
L4: ACE-G, ace_g_fusion_in_s2 = True
```

解释：

```text
L1 poor vs L0 -> memory compression/fusion path 有害
L3 better than L4 -> S2 fusion 过拟合或不稳定
L2 better than L1 -> sparse sampling 在不过激时有帮助
```

### Step 8：Training Parameter Ablation

只有完成 Steps 1-7 后，才测试训练参数嫌疑项。

优先改动：

1. 如果当前 run 使用 `10240`，恢复 Indoor6-style batch size：

```text
batch_size = 5120
```

2. 测试更低或更少 boost 的 S2 learning：

```text
s2_learning_rate_max = 5e-4 or 1e-4
head_lr_multiplier_s2 = 1.0
```

3. 只有在保持主 contract 可解释时，才比较 legacy profile 和更不激进的 profile。

4. 如可行，降低 replacement-driven buffer bias：

```text
buffer_sampling_replacement = False
```

解释：

```text
lower LR / batch improves -> optimization overfit or instability
replacement off improves  -> repeated sparse/buffer samples caused bias
no change                 -> 回到 data/memory coverage
```

### Step 9：Memory Coverage and Density Ablation

如果 LMC 是主要嫌疑，检查 memory construction。

诊断：

1. 比较 memory view camera centers 和 train/test trajectory；
2. 计算每个 train/test frame 到最近 memory view 的距离；
3. 比较 errors 和 nearest-memory-view distance；
4. 比较 ASB memory 和 FPS-flat memory；
5. 如果可行，比较 40 views 和更大 budgets，例如 80 或 120；
6. 如果有，比较 BSE memory 和 simpler pooled memory；
7. 确认 sparse-depth memory extraction 使用 `patch_depth_sampling=nearest_valid`。

解释：

```text
more views improve -> memory coverage bottleneck
FPS improves       -> ASB selection too local or biased
simple pooled improves over BSE -> BSE sparsification/density issue
nearest_valid improves -> sparse-depth memory geometry issue
```

## 5. 最小实验顺序

推荐的第一轮诊断顺序是：

```text
D0: data/pose/calibration audit
D1: current best checkpoint train-set eval
D2: vanilla DINO ACE on same RIO10 ACE-format split
D3: DSAC sensitivity sweep on current best checkpoint
D4: no-guided vs guided-ratio 0.25/0.50/1.00
D5: fusion-in-S2 off vs on
D6: memory coverage 40 ASB vs larger/FPS memory
```

除非 GPU 空闲，否则不要在 D0-D2 之前运行 D4-D6，因为 D0-D2 决定失败是否真的在 LMC。

## 6. 决策树

使用下面的判断规则。

```text
train-set eval poor
  -> 优先检查 data/pose/calibration/eval contract 和 optimization sanity

train-set eval good, test eval poor
  -> true generalization failure；继续 vanilla baseline 和 memory coverage

vanilla DINO ACE poor
  -> RIO10 domain/data/eval/training baseline issue；不要先怪 LMC

vanilla DINO ACE good, ACE-G poor
  -> LMC/memory/fusion/sparse sampling issue

no-guided >= guided
  -> sparse guided sampling 有害或 bias 太强

guided 0.25/0.50 > guided 1.0
  -> sparse sampling 只适合作为 partial prior；ratio 1.0 损害 diversity

fusion-off > fusion-on
  -> S2 fusion continuation 在 RIO10 上有害

larger/FPS memory > 40 ASB memory
  -> memory coverage 或 selection 是瓶颈

DSAC relaxed threshold helps a lot
  -> coordinates noisy but partially useful；训练质量问题

DSAC sweep does not help
  -> coordinate predictions 或 data/eval contract 可能错误
```

## 7. 现在不要做什么

在这套诊断阶梯定位失败模式之前，不要用 RIO10 结果作为理由去添加：

- PointRoPE；
- Cascading Internal Fusion Assembly；
- query-side DPT/FPN adapters；
- multi-level compression changes；
- view-token conditioning；
- multi-memory routing；
- sparse-depth auxiliary supervision；
- reference-coordinate auxiliary loss。

这些方向后续可能有价值，但现在会引入混杂变量。

## 8. 预期输出

这轮诊断阶段应该得出下面结论之一：

```text
A. Data/eval contract 有问题，需要先修复。
B. Vanilla ACE-DINOv2 在 RIO10 上已经弱，因此 RIO10 需要新的 baseline recipe。
C. Vanilla 可接受，但 ACE-G memory/fusion 损害泛化。
D. Sparse guided sampling 只在 partial ratio 下有效，不适合 ratio 1.0。
E. Memory coverage/density 是限制因素。
F. DSAC/evaluation 设置显示 coordinates noisy but recoverable。
```

只有其中一个结论被证据支持后，项目才应该恢复架构修改。
