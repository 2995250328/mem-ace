# GLACE Local-Only Memory Fusion 实施与实验计划

本文档记录下一轮 GLACE-LMC 改造主线。目标是把当前 `full decoder fusion` 改成：

```text
vanilla GLACE scene head 初始化
+ GLACE encoder / global feature source 冻结
+ LMC 只融合 local feature
+ global feature bypass
+ head 早期解冻共同适配
+ residual/gate 使用可控 local delta
```

核心判断：

```text
当前代码虽然支持 local_delta_tanh_scalar，但只是 residual 输出端不改 global。
fusion query 仍然是 concat(global, local)，所以还不是严格 local-only fusion。
下一步最该改的是 fusion query 端：只让 local feature 进入 LMC fusion。
```

## 0. 实施状态

2026-05-28 Step 1a 已落地：

- 新增 `--lmc_fusion_target decoder|local`，默认 `decoder`，旧实验行为不变。
- checkpoint / resume config 记录 `requested_lmc_fusion_target`、`effective_lmc_fusion_target`、`fusion_query_dim`。
- 新增 GLACE feature helper：`_get_glace_global_features()`、`_split_glace_decoder_feature_maps()`、`_build_glace_head_input()`。

2026-05-28 Step 2 已落地：

- `local` 模式下 fusion 构造维度切为 local / encoder feature dim。
- S1 online、S1 buffer、ACE-G S2 路径执行 local-only fusion + global bypass。
- `local` 模式不再经过完整 decoder residual adapter；`decoder` 模式继续保持旧 adapter 路径。

2026-05-28 Step 3 已落地：

- 新增 `--local_residual_mode none|fixed_alpha` 和 `--local_residual_alpha`。
- `fixed_alpha` 使用 `local + alpha * (fused_local - local)`，训练和 eval 保持一致。

2026-05-29 Step 4 已落地：

- 新增 `--glace_freeze_encoder` 和 `--glace_freeze_head`，旧 `--glace_freeze_base_network` 只作为 head 默认兼容。
- 支持 encoder 冻结、head 解冻的 GLACE-LMC co-adaptation 实验。

尚未落地：

- alpha warmup。
- head freeze 拆分与 early unfreeze。
- local delta / gate 诊断增强。

## 1. 当前真实代码状态

关键位置：

- `TrainerACEDINOv2LMC._create_regressor()`  
  [trainer_dinov2_lmc.py](/home/xwh/project/ace_depth/ace_dinov2_lmc/trainer_dinov2_lmc.py:1213)

- `TrainerACEDINOv2LMC.__init__()`  
  [trainer_dinov2_lmc.py](/home/xwh/project/ace_depth/ace_dinov2_lmc/trainer_dinov2_lmc.py:1260)

- `TrainerACEDINOv2LMC._build_glace_decoder_features()`  
  [trainer_dinov2_lmc.py](/home/xwh/project/ace_depth/ace_dinov2_lmc/trainer_dinov2_lmc.py:1135)

- `TrainerACEDINOv2LMC._fuse_features()`  
  [trainer_dinov2_lmc.py](/home/xwh/project/ace_depth/ace_dinov2_lmc/trainer_dinov2_lmc.py:3185)

- `TrainerACEDINOv2LMC._training_step_ace_g()`  
  [trainer_dinov2_lmc.py](/home/xwh/project/ace_depth/ace_dinov2_lmc/trainer_dinov2_lmc.py:5531)

- `GLACEDecoderFeatureResidualAdapter.forward()`  
  [glace_backend.py](/home/xwh/project/ace_depth/ace_dinov2_lmc/glace_backend.py:44)

当前路径：

```text
local encoder feature
  -> _build_glace_decoder_features()
  -> concat(global, local)
  -> _fuse_features(concat(global, local), compressed_memory)
  -> _mix_glace_decoder_features(base_decoder, fused_decoder)
  -> GLACE head
```

当前 `local_delta_tanh_scalar` 的实际行为：

```text
global_out = global_base
local_out  = local_base + gain * (local_fused - local_base)
head_input = concat(global_out, local_out)
```

但 `local_fused` 是从完整 `concat(global, local)` fusion 输出里切出来的，不是 strict local-only fusion。

## 2. 主改造目标

目标路径：

```text
global_raw = get_global_feature(img_idx)
local_raw  = GLACE local encoder feature

local_fused = LMCFusion(local_raw, compressed_memory)
local_out   = local_raw + alpha * gate * (local_fused - local_raw)

head_input = concat(expand(global_raw), local_out)
scene_coord = GLACE_head(head_input)
```

设计原则：

1. `global_raw` 不进入 LMC fusion。
2. `global_raw` 不被 residual / adapter 改写。
3. LMC 只增强 local/backbone feature。
4. GLACE head 使用 vanilla scene head 初始化。
5. head 不长期冻结，需要早期解冻共同适配。
6. guard loss 初期关闭，避免过早拉回 vanilla GLACE。

## 3. 代码实施步骤

### Step 1: 新增 local-only fusion 配置

新增参数：

```text
--lmc_fusion_target decoder|local
```

含义：

- `decoder`: 当前行为，fusion query 是完整 `concat(global, local)`。
- `local`: 新行为，fusion query 只使用 local feature。

验收：

1. 默认值保持 `decoder`，老实验不变。
2. `local` 模式下 fusion 构造维度使用 local feature dim，而不是 decoder dim。
3. resume config / checkpoint metadata 记录 `lmc_fusion_target`。

### Step 2: 新增 GLACE feature helper

建议新增函数：

```python
def _get_glace_global_features(self, img_idx_B, *, device, dtype):
    ...

def _split_glace_decoder_feature_maps(self, decoder_BCHW, *, stage_tag):
    ...

def _build_glace_head_input(self, global_BC, local_BCHW):
    ...
```

目的：

1. 避免在 S1/S2 多处手写 `global_dim` 切分。
2. 明确 `global` 和 `local` 的语义边界。
3. 后续做 diagnostics 时更容易记录 `delta_local_ratio`。

验收：

1. `decoder -> split -> rebuild` 后 shape 与原始 decoder 一致。
2. dtype / device 与原路径一致。
3. GLACE 非 backend 不受影响。

### Step 3: 实现 local-only fusion forward

主要修改：

- `_training_step_ace_g()`
- S1 buffer 训练路径中调用 `_fuse_features()` 的地方
- 普通 S1 路径中调用 `_fuse_features()` 的地方

目标：

```text
if lmc_fusion_target == "decoder":
    保持当前路径

if lmc_fusion_target == "local":
    split decoder feature -> global_raw, local_raw
    fused_local = _fuse_features(local_raw, compressed_memory)
    local_out = local_raw + alpha * (fused_local - local_raw)
    head_input = concat(global_raw, local_out)
```

验收：

1. `local` 模式下 `_fuse_features()` 输入 channel 数为 local dim。
2. 最终 `head_input` channel 数仍等于 GLACE decoder dim。
3. `global` 分支完全 bypass，不进入 fusion。
4. `decoder` 模式结果不变。

### Step 4: 替换完整 decoder residual adapter 的主路径

新增参数：

```text
--local_residual_mode none|fixed_alpha|learned_alpha|global_channel_gate
--local_residual_alpha 0.2
--local_residual_alpha_warmup_steps 0
```

第一版只实现：

```text
fixed_alpha
```

形式：

```text
local_out = local_raw + alpha * (local_fused - local_raw)
```

暂不实现 learnable gate，避免一开始 gate 学成 0。

验收：

1. `alpha=0` 等价于 no-LMC local delta。
2. `alpha>0` 时 `delta_local_ratio` 非零。
3. `fixed_alpha` 不引入额外可训练参数。

### Step 5: 拆分 freeze 开关

当前：

```text
--glace_freeze_base_network True
```

会同时冻结 encoder 和 head。

建议拆成：

```text
--glace_freeze_encoder True
--glace_freeze_head False
--glace_head_unfreeze_after_steps 500
```

兼容策略：

1. 保留 `glace_freeze_base_network`，用于旧配置。
2. 新参数显式出现时优先生效。
3. 旧参数为 True 时，默认等价于 `freeze_encoder=True, freeze_head=True`。

主方案：

```text
encoder frozen
head short warmup frozen, then trainable
```

验收：

1. encoder 始终不训练。
2. head 在 warmup 后进入 optimizer。
3. 日志清晰打印 head trainable 状态变化。

### Step 6: guard loss 后移

新增或约定配置：

```text
--glace_antiregression_weight 0.0
```

第一轮 local-only 主实验先不开 guard。

后续再试：

```text
weight = 0.01 / 0.05 / 0.1
margin = 1.0px / 3.0px
```

验收：

1. guard 关闭时 fused path 不被 vanilla base 过早压回。
2. guard 打开时记录 `guardPx/basePx/guardLoss`。

### Step 7: 诊断日志

至少新增：

```text
delta_local_ratio = ||local_out - local_raw|| / ||local_raw||
delta_head_ratio  = ||head_input - vanilla_head_input|| / ||vanilla_head_input||
```

后续可加：

```text
gate mean/std/p10/p50/p90
attention entropy
effective memory token count
token usage histogram
err_fused - err_base
```

验收：

1. 每个 S2 log interval 打印 `deltaLocal`。
2. `alpha=0` 时 `deltaLocal` 接近 0。
3. `local-only` 模式能确认 memory branch 是否真的有非零影响。

## 4. 实验矩阵

### A 组：主结构对照

```text
A0: vanilla GLACE baseline

A1: current full decoder fusion
    lmc_fusion_target=decoder
    residual=current local_delta_tanh_scalar

A2: local-only fusion + global bypass + frozen head
    lmc_fusion_target=local
    local_residual_mode=fixed_alpha
    head frozen

A3: local-only fusion + global bypass + vanilla head init + early unfreeze
    主方案

A4: no LMC + vanilla head init + same early-unfreeze schedule
    head-only fine-tune 对照

A5: local-only fusion + fixed alpha residual
    alpha sweep

A6: local-only fusion + no residual
    local_residual_mode=none
```

关键判断：

```text
A3 > A4:
    LMC 有独立贡献

A3 ≈ A4:
    提升主要来自 head fine-tune，LMC 没发挥

A2 ≈ A0 but A3 > A0:
    frozen head 限制 fusion，head co-adaptation 必要

A1 ≈ A0 but A3 > A1:
    当前 full decoder fusion / adapter 设计有问题
```

### B 组：alpha 消融

```text
alpha = 0.0
alpha = 0.1
alpha = 0.2
alpha = 0.5
```

第一版不做 learnable alpha。

### C 组：head 训练策略

```text
H0: head always frozen
H1: warmup frozen, then unfreeze
H2: head train from step 0
H3: random head init
H4: vanilla head init, no LMC, same schedule
```

主方案优先：

```text
H1
```

### D 组：fusion 位置

```text
F0: full decoder fusion
F1: local-only fusion
F2: local-only fusion + global channel gate
F3: global-only fusion
F4: local fusion + global passthrough
```

优先：

```text
F1
```

后续再考虑：

```text
F2
```

## 5. 推荐落地顺序

严格按下面顺序推进：

1. 新增 `lmc_fusion_target`，保留 decoder 默认。
2. 新增 GLACE feature helper。
3. 跑通 local-only fusion + global bypass。
4. 替换为 fixed-alpha local delta。
5. 新增 `alpha=0` sanity check。
6. 拆分 freeze encoder/head。
7. 实现 head early unfreeze。
8. 加 no-LMC head fine-tune 对照。
9. 加 delta diagnostics。
10. 再考虑 guard / global gate / token usage。

## 6. 第一轮最小实现目标

第一轮只做：

```text
--lmc_fusion_target local
--local_residual_mode fixed_alpha
--local_residual_alpha 0.2
--glace_global_bypass True
```

暂时不做：

```text
global gate
learnable alpha
spatial gate
strong guard loss
token usage histogram
```

验收实验：

```text
E0: current decoder fusion baseline
E1: local-only alpha=0.0
E2: local-only alpha=0.1
E3: local-only alpha=0.2
```

如果 `E2/E3` 相比 `E1` 没有任何收益，说明 memory local delta 本身没有提供有效信号，需要回头看 memory token / fusion attention。

如果 `E2/E3` 比 `E1` 有收益，但低于 head-unfreeze 版本，则继续推进 head co-adaptation。

