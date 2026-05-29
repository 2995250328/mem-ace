# GLACE-LMC Memory Fusion 代码审查与修改思路

日期：2026-05-29
范围：`ace_dinov2_lmc` 中 GLACE-LMC / memory fusion 相关训练、配置、checkpoint 和评测路径。
状态：只读审查。本文件未对应任何代码修改。

## 1. 背景

当前 GLACE-LMC 方向的目标，是验证压缩后的 LMC memory token 能否通过只修改 GLACE 的 local feature 流来提升 scene coordinate regression，同时让 global feature bypass memory fusion。

原始设想：

```text
vanilla GLACE scene head 初始化
+ GLACE encoder / global feature source 冻结
+ LMC 只融合 local feature
+ global feature bypass
+ local residual / fixed alpha 控制 memory delta
+ head 可选共同适配
```

核心研究问题是：

```text
压缩 memory token 是否能通过 local feature 增强 GLACE 的 scene coordinate regression？
```

已有实验显示，当前这条直接路径并不乐观：

```text
LMC compressed memory -> local feature delta -> GLACE head
```

direct replace 会崩，中等 alpha 会劣化或崩溃，极小 alpha 只产生弱小且不稳定的波动。

## 2. 当前已实现能力

### 2.1 Fusion target

代码支持：

```bash
--lmc_fusion_target decoder|local
```

语义：

```text
decoder:
  旧路径。fusion query 是完整 decoder feature：
  concat(global, local)。

local:
  GLACE-LMC 专用路径。只对 local feature 做 fusion。
  global feature bypass memory fusion，并在 GLACE head 前重新拼回。
```

相关文件：

```text
options_dinov2_lmc.py
trainer_dinov2_lmc.py
test_ace_dinov2_lmc.py
```

### 2.2 Local residual mode

代码支持：

```bash
--local_residual_mode none|fixed_alpha
--local_residual_alpha <float>
```

语义：

```text
none:
  local_out = local_fused

fixed_alpha:
  local_out = local_raw + alpha * (local_fused - local_raw)
```

`local_residual_alpha` 期望位于 `[0, 1]`。

### 2.3 GLACE freeze policy

相关参数：

```bash
--glace_freeze_encoder
--glace_freeze_head
--glace_freeze_base_network
```

当前有效语义：

```text
glace_freeze_encoder:
  默认 True。GLACE encoder 默认冻结。

glace_freeze_head:
  默认 None。为 None 时继承 glace_freeze_base_network。

glace_freeze_base_network:
  默认 True。
```

因此默认情况下：

```text
GLACE encoder 会被冻结。
GLACE head 也会被冻结。
```

### 2.4 Checkpoint 和 eval 同步

checkpoint 的 `lmc_config` 已记录关键 fusion 和 freeze 字段，包括：

```text
lmc_fusion_target
requested_lmc_fusion_target
effective_lmc_fusion_target
local_residual_mode
local_residual_alpha
glace_freeze_encoder
glace_freeze_head
```

test 路径会从 checkpoint 读取这些字段，并复现对应的 decoder 或 local-only fusion 路径。

## 3. 已有实验结论

统一评测设置：

```text
scene:
  /data/xwh/Wayspots/wayspots_bears

memory:
  /home/xwh/project/ace_depth/ace_dinov2_lmc/04_evaluation/wayspots_lmc/memory/wayspots_bears/32v_v0.05_bilinear_bse_ut0.02_noaug_asb_adaptive_pool1.0_rgate4_prepair8_sor_gm_l2/20260525_223355/memory_bse.pt

GLACE init head:
  /home/xwh/project/ace_depth/ace_dinov2_lmc/04_evaluation/wayspots_baselines/20260525_190055/wayspots_bears/glace/model.pt

eval:
  seeds = 1305,2026,4242
  hypotheses = 256
  aggregation = median
```

### 3.1 Decoder baseline

代表性结果：

```text
Median: 1.00 deg / 3.07 cm
25cm/5deg: 95.00%
10cm/5deg: 89.14%
5cm/5deg: 77.59%
2cm/2deg: 14.48%
1cm/1deg: 0.86%
```

说明 decoder path 能保持 GLACE baseline 水平。

### 3.2 Local-only direct replacement

配置：

```text
lmc_fusion_target = local
local_residual_mode = none
```

即：

```text
local_out = local_fused
```

代表性结果：

```text
Median: 29.45 deg / 116.07 cm
25cm/5deg: 0.52%
10cm/5deg: 0.17%
5cm/5deg: 0.00%
```

结论：

```text
直接替换 local feature 会崩溃。
这条路径不应继续。
```

### 3.3 Fixed-alpha local residual

配置：

```text
lmc_fusion_target = local
local_residual_mode = fixed_alpha
```

观测现象：

```text
alpha = 0.2:
  崩溃，主阈值基本为 0。

alpha = 0.1:
  明显劣化。

alpha = 0.03:
  仍然劣化。

alpha = 0.015 / 0.01 / 0.005:
  接近 baseline，但没有稳定超过 decoder baseline。
```

结论：

```text
可用 alpha 极小，大约 0.005 到 0.01。
小 alpha 只产生弱小、局部、不稳定的波动。
继续细扫 alpha 没有价值。
```

### 3.4 Head co-adaptation

观测现象：

```text
head trainable + alpha=0.01:
  没有改善，整体更差。

head trainable + alpha=0.005:
  没有改善，整体更差。
```

结论：

```text
head co-adaptation 没有救回 local memory fusion。
```

注意事项：

```text
该结论需要附带 exact effective freeze policy 和 optimizer parameter groups。
尤其要确认对应 run 中 GLACE head 确实进入了 optimizer。
```

## 4. 代码审查发现

## 4.1 高优先级：`glace_freeze_head` 默认行为可能污染实验解释

相关文件：

```text
options_dinov2_lmc.py
trainer_dinov2_lmc.py
```

当前行为：

```text
--glace_freeze_head 默认 None。
None 时继承 --glace_freeze_base_network。
--glace_freeze_base_network 默认 True。
因此 head 默认冻结。
```

风险：

```text
旧配置或隐式配置如果没有传 --glace_freeze_head False，GLACE head 可能实际一直冻结。
这会影响“head co-adaptation 没有帮助”的实验解释。
```

短期建议：

不要立刻修改默认行为。先增加显式日志和一致性检查：

```text
打印 effective freeze policy。
打印 trainable parameter counts。
打印 optimizer parameter groups。
如果用户期望 head trainable 但 optimizer 中没有 head 参数，则报错。
```

推荐日志内容：

```text
[GLACE-LMC config]
  freeze_base_network = ...
  freeze_encoder = ...
  freeze_head = ...
  lmc_fusion_target = ...
  effective_lmc_fusion_target = ...
  local_residual_mode = ...
  local_residual_alpha = ...
  ace_g_fusion_in_s2 = ...

[Trainable params]
  encoder = ...
  head = ...
  compressor = ...
  fusion = ...
  residual_adapter = ...

[Optimizer groups]
  group_name, lr, num_tensors, num_params
```

## 4.2 高优先级：`glace_freeze_encoder=False` 当前很可能是 no-op

当前行为：

```text
--glace_freeze_encoder False 会设置 encoder 参数 requires_grad=True。
```

但是：

```text
S1/S2 optimizer 没有包含 encoder 参数。
S1 online feature extraction 使用 torch.no_grad()。
```

因此：

```text
即使 requires_grad=True，encoder 也不会更新。
```

风险：

```text
任何 encoder-unfreeze 实验都可能无效。
```

短期建议：

在当前 GLACE-LMC flow 中拒绝尚未真正支持的 encoder unfreeze：

```text
if model_backend == "glace_lmc" and glace_freeze_encoder is False:
    raise ValueError(
        "glace_freeze_encoder=False is not supported yet: "
        "encoder params are not included in S1/S2 optimizers."
    )
```

之后如需真正训练 encoder，应单独设计：

```text
encoder optimizer group
encoder LR ratio
移除或重构 no_grad feature extraction
显存和稳定性策略
```

## 4.3 中优先级：local-only delta logging 可能使用了错误的维度约定

在 local-only 路径中：

```text
base_features_bC 已经是 local 维度。
```

但某些 delta logging 可能仍使用：

```python
base_features_bC[:, global_dim:]
```

这会导致 local delta 统计无效或难以解释。

建议修正：

显式拆分 logging 路径：

```python
if effective_lmc_fusion_target == "decoder":
    base_local = base_features_bC[:, global_dim:]
    head_local = head_features_bC[:, global_dim:]
    delta_stats_scope = "decoder_local_slice"

elif effective_lmc_fusion_target == "local":
    base_local = base_features_bC
    head_local = head_features_bC
    delta_stats_scope = "local_query"
```

将 `delta_stats_scope` 写入日志。

## 4.4 中优先级：iteration eval 和 post-train eval 设置不同

当前行为：

```text
iteration eval:
  hypotheses 固定为 64。
  seed 来自 eval_dsacstar_seed。

post-train eval:
  seeds 来自 post_train_eval_seeds。
  hypotheses 来自 post_train_hypotheses。
```

风险：

```text
iteration best checkpoint 指标和 post-train 多 seed 指标不能直接比较。
```

建议：

新增显式参数：

```bash
--iteration_eval_hypotheses
--iteration_eval_seed
```

或者至少在每个 summary 中写入：

```text
eval_type
hypotheses
seeds
deterministic
dsacstar_seed
```

## 4.5 中优先级：post-train summary 缺少完整 fusion/freeze 语义字段

standalone test summary 已经写入部分字段，但 post-train summary 不够完整。

建议 post-train summary 补齐字段：

```text
model_backend
lmc_flow
lmc_fusion_target
requested_lmc_fusion_target
effective_lmc_fusion_target
fusion_query_dim
local_residual_mode
local_residual_alpha
glace_freeze_base_network
glace_freeze_encoder
glace_freeze_head
ace_g_fusion_in_s2
eval_deterministic
post_train_eval_seeds
post_train_hypotheses
```

目标：

```text
只看 eval summary 就能理解实验语义，不必重新打开完整 config。
```

## 5. 不建议继续的方向

不要继续：

```text
1. 不要继续扫 alpha=0.006 / 0.0075 / 0.012。
2. 不要继续 local_residual_mode=none。
3. 不要单纯增加 iterations 或 training steps。
4. 不要只看最终 pose metrics 而不看 feature/pixel diagnostics。
5. 不要通过强化 guard loss 把 fused outputs 拉回 vanilla behavior。
6. 不要马上构建复杂 gate 网络。
7. 不要马上训练 encoder。
```

原因：

```text
当前未解决的核心问题不是训练时间是否足够。
当前未解决的核心问题是 memory delta 是否落在 GLACE head 可以解释的 feature manifold 上。
```

## 6. 推荐修改路线

### Step 1：增加 freeze / optimizer 诊断和 guard

目标：

```text
不改变训练算法。先让实验语义可见。
```

修改内容：

```text
1. 打印 effective GLACE-LMC config。
2. 打印 trainable parameter counts。
3. 打印 optimizer parameter groups。
4. 拒绝当前未真正支持的 encoder unfreeze。
5. 检查 head trainability 是否与 optimizer membership 一致。
```

优先文件：

```text
trainer_dinov2_lmc.py
options_dinov2_lmc.py
```

### Step 2：修正 local-only delta logging

目标：

```text
让 feature-level diagnostics 可信。
```

修改内容：

```text
1. 分离 decoder-fusion 和 local-fusion 的 logging 约定。
2. local-only features 不再按 global_dim 切片。
3. 记录 delta_stats_scope。
```

建议记录指标：

```text
local_raw_norm
local_fused_norm
local_delta_norm
delta_local_ratio
delta_cos
head_input_delta_ratio
```

优先文件：

```text
trainer_dinov2_lmc.py
```

### Step 3：补齐 post-train summary 字段

目标：

```text
让 post-train eval、iteration eval、standalone test summary 可横向比较。
```

修改内容：

```text
在 train_ace_dinov2_lmc.py 的 post-train summary 中加入完整 semantic fields。
```

优先文件：

```text
train_ace_dinov2_lmc.py
```

### Step 4：增加 base-vs-fused pixel-level diagnostics

目标：

回答核心问题：

```text
memory delta 是只帮助 hard pixels，还是破坏大多数 GLACE local feature？
```

在同一个 batch 上计算：

```text
base:
  GLACE head(concat(global, local_raw))

fused:
  GLACE head(concat(global, local_raw + alpha * (local_fused - local_raw)))
```

记录：

```text
err_base_px
err_fused_px
err_delta = err_fused_px - err_base_px
```

按 base reprojection error 分桶：

```text
easy:
  base_err < 5 px

medium:
  5 px <= base_err < 20 px

hard:
  base_err >= 20 px
```

输出：

```text
mean err_delta
median err_delta
improved ratio
worsened ratio
hard-pixel improved ratio
easy-pixel damaged ratio
```

解释：

```text
如果 hard pixels 少量改善但 easy pixels 大量受损：
  memory 应作为 gate / uncertainty / hard-pixel selector。

如果大多数 pixels 都变差：
  应放弃直接修改 local feature。

如果 hard pixels 明显改善且 easy pixels 不受损：
  gated memory fusion 才值得继续。
```

## 7. 最小实现顺序

推荐第一轮 patch：

```text
1. Freeze policy log。
2. Trainable parameter log。
3. Optimizer group log。
4. Encoder-unfreeze no-op guard。
5. Local-only delta logging fix。
6. Post-train summary field completion。
```

推荐第二轮 patch：

```text
1. Pixel-level base-vs-fused diagnostics。
2. Easy / medium / hard bucket statistics。
3. Attention entropy / token usage statistics。
```

推荐下一轮实验：

```text
1. 不先训练新模型。
2. 使用已有 checkpoint 或小的固定 batch。
3. 运行 diagnostics，识别 memory delta 的失败模式。
```

## 8. 未来可能的新结构方向

只有当 pixel-level diagnostics 显示 memory 能帮助 hard pixels 且不会广泛破坏 easy pixels 时，才考虑以下方向。

### 8.1 Memory 只预测 gate

memory 不直接生成 feature delta，而是预测 gate / confidence / uncertainty。

```text
local_out = local_raw + gate * small_delta
```

gate 应该在 easy pixels 接近 0，只在 hard pixels 打开。

### 8.2 Memory 只作用于 hard pixels

使用 base GLACE reprojection error 或 confidence 来识别 hard pixels：

```text
if base_err is high:
    allow memory fusion
else:
    keep vanilla local feature
```

### 8.3 Memory 作为 auxiliary loss

memory 不修改 head input，而是为 hard-pixel local features 提供辅助监督或 contrastive alignment。

### 8.4 Retrieval / context conditioning

memory 提供 scene-level 或 region-level context，但不直接生成 per-pixel local feature delta。

## 9. 当前推荐结论

当前最优先的不是继续训练 sweep，而是让实验语义和失败模式可观测：

```text
1. 确认哪些模块实际可训练。
2. 确认 optimizer parameter groups 与预期 freeze policy 一致。
3. 修正 local-only delta logging。
4. 增加 base-vs-fused pixel-level diagnostics。
```

在这些 diagnostics 完成前，不建议扩大实验矩阵。

推荐下一步：

```text
先实现 diagnostic patch，
再用已有 checkpoint 或小固定 batch 判断 memory delta 是帮助 hard pixels，
还是广泛破坏 GLACE feature manifold。
```
