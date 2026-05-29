# GLACE-LMC 审阅地图

这份文档只做定位，不解释训练细节。目标是让你快速找到这次 GLACE-LMC / ACE-G 相关改动的代码位置、配置来源和日志证据。

## 1. 参数入口

先看这里，确认命令行参数是怎么进入训练逻辑的。

- `options_dinov2_lmc.py`
  - `--glace_init_head_path`
  - `--glace_freeze_base_network`
  - `--glace_freeze_encoder`
  - `--glace_freeze_head`
  - `--glace_antiregression_weight`
  - `--glace_antiregression_margin_px`
  - `--glace_antiregression_max_px`
  - `--lmc_fusion_target`

重点位置：

- `options_dinov2_lmc.py:157-245`

你要确认的点：

1. 这些参数是否默认符合你的预期。
2. 是否会被 `train_preset` 或 CLI 显式参数覆盖。
3. 是否存在默认值过强、导致实验过早冻结或过强约束的问题。

## 2. 训练主流程

真正影响训练行为的逻辑主要在 `trainer_dinov2_lmc.py`。

### 2.1 GLACE regressor 构建

位置：

- `trainer_dinov2_lmc.py:1206-1238`

关注点：

1. `glace_init_head_path` 是否真的加载了已训练 head。
2. 如果没提供该路径，是否回退到默认构建逻辑。
3. 这里是否影响后续 head reset / warm-start 行为。

### 2.2 Fusion target 接线

位置：

- `trainer_dinov2_lmc.py:1739-1764`
- `trainer_dinov2_lmc.py:1937-1957`
- `trainer_dinov2_lmc.py:4699-4706`

关注点：

1. `requested_lmc_fusion_target` 是命令行请求值。
2. `effective_lmc_fusion_target` 现在等于请求值；`local` 会真实切到 local-only fusion。
3. local-only forward 通过 `_fuse_lmc_features_for_head()` 统一覆盖 S1 online、S1 buffer 和 ACE-G S2。

### 2.3 Base network freeze

位置：

- `trainer_dinov2_lmc.py:1046-1065`
- `trainer_dinov2_lmc.py:1274-1310`

关注点：

1. `_glace_freeze_encoder()` 和 `_glace_freeze_head()` 是否按实验配置拆分冻结。
2. 冻结后是不是只剩 plugin 在动。
3. 是否在初始化时就把 reference regressor 固定下来，供后面的 guard loss 使用。

### 2.4 Head reset

位置：

- `trainer_dinov2_lmc.py:2806-2818`

关注点：

1. 只要传了 `glace_init_head_path`，是否就跳过 head reset。
2. 这会不会让某些 round 的初始化状态和你预期不一致。

### 2.5 S1 / S1-Buffer

位置：

- `trainer_dinov2_lmc.py:3557-3860`
- `trainer_dinov2_lmc.py:3860-4050`

关注点：

1. S1 scheduler 是怎么构造的。
2. `freeze_glace_head` 是否会把 head 排除在 optimizer 之外。
3. buffer 训练里为什么会打印 `CONVERGENCE WARNING`。

你在日志里看到的这个 warning，对应的就是这里：

- `trainer_dinov2_lmc.py:3810-3834`

### 2.6 S2 / ACE-G 分支

位置：

- `trainer_dinov2_lmc.py:4361-4557`

关注点：

1. `ace_g_fusion_in_s2` 是否让 fusion 在 S2 里继续训练。
2. `ace_g_fusion_lr_ratio` 和 `glace_residual_lr_ratio` 是否太大。
3. head、fusion、residual 的 lr group 是怎么分开的。

### 2.7 单步训练与 guard loss

位置：

- `trainer_dinov2_lmc.py:5520-5715`

关注点：

1. `ace_g` 单步 forward 的路径。
2. `gGain` 和 `dRatio` 是如何计算出来的。
3. `glace_antiregression_weight` 打开后，guard loss 是怎样和 base regressor 做比较的。
4. 最终日志里会把哪些诊断量打印出来。

你在日志里看到的这些字段，就来自这里：

- `gGain`
- `dRatio`
- `guardPx`
- `basePx`
- `guardLoss`

## 3. 训练入口与目录

位置：

- `train_ace_dinov2_lmc.py:160-305`
- `train_ace_dinov2_lmc.py:740-820`

关注点：

1. `memory_compare_ace_g_v2` 这种 preset 默认给了哪些参数。
2. preset 有没有覆盖你在命令行里想手动调的值。
3. run directory 是怎么拼出来的，为什么 best checkpoint 和 full log 会落到那个位置。

## 4. 本次 run 的证据文件

这个 run 的目录：

- `/data/xwh/ace_dinov2_lmc/04_evaluation/wayspots_glace_lmc/frozen_guard_active/memory_pooled_vs_asb/extracted/wayspots_bears/dino_ace_lmc_ace_g/20260527_112733_aceg_fS2cie_global_res476_buf3.2M_F8M_K64_it28_ep24_bs10240_spi_s1buf_sp512_onecycle_improved`

建议按这个顺序看：

1. `run_config.json`
2. `best_checkpoint_meta.json`
3. `best_K64_it28_wayspots_bears_glace_lmc_frozen_guard_active_eval_log.txt`
4. `training_full_log.txt`

你可以重点比对这些字段：

- `best_iter`
- `best_score`
- `glace_freeze_base_network`
- `glace_freeze_encoder`
- `glace_freeze_head`
- `glace_antiregression_weight`
- `s2_learning_rate_max`
- `ace_g_fusion_lr_ratio`
- `lmc_iterations`
- `lmc_train_steps`
- `s1_loss_step_mode`

## 5. 快速审阅建议

如果你只想快速判断这套配置值不值得继续跑，优先看这几件事：

1. `best_iter` 是否很早就出现并且后续不再刷新。
2. `S1-Buffer` 的 `CONVERGENCE WARNING` 是否反复出现。
3. `dRatio` 是否明显上升，但 `pct5` 没有同步提高。
4. `guardLoss` 是否太小，根本压不住漂移。
5. `freeze_base_network` 打开后，是否只剩 plugin 在动，但收益仍然不稳定。

## 6. 你可以直接跳看的位置

如果你只打算点开 6 个位置，优先这几个：

1. `options_dinov2_lmc.py:157-245`
2. `trainer_dinov2_lmc.py:1046-1065`
3. `trainer_dinov2_lmc.py:1206-1238`
4. `trainer_dinov2_lmc.py:2806-2818`
5. `trainer_dinov2_lmc.py:4361-4557`
6. `trainer_dinov2_lmc.py:5520-5715`
