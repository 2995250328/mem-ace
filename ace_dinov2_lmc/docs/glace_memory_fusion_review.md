# GLACE Memory Fusion 审阅地图

这份文档只看一条链路：

`memory -> GLACE encoder/head -> S2 融合 -> guard / diagnostics`

不包含训练入口、目录构建、preset 细节，也不解释整体 LMC 训练流程。

## 1. 先看参数入口

位置：

- [options_dinov2_lmc.py](/home/xwh/project/ace_depth/ace_dinov2_lmc/options_dinov2_lmc.py:150)

函数 / 区段：

- `get_lmc_train_parser()`

重点参数：

- `--glace_init_head_path`
- `--glace_freeze_base_network`
- `--glace_residual_mode`
- `--glace_residual_gate_init`
- `--glace_residual_lr_ratio`
- `--glace_antiregression_weight`
- `--glace_antiregression_margin_px`
- `--glace_antiregression_max_px`

你主要要确认：

1. 这些参数有没有被 preset 或 CLI 覆盖成你不想要的值。
2. `glace_freeze_base_network=True` 时，base GLACE 是否真的被冻结。
3. `glace_antiregression_weight` 是否足够大到能约束漂移。

## 2. GLACE regressor 是怎么初始化的

位置：

- [trainer_dinov2_lmc.py](/home/xwh/project/ace_depth/ace_dinov2_lmc/trainer_dinov2_lmc.py:1213)

函数：

- `TrainerACEDINOv2LMC._create_regressor()`

看这段：

- `glace_init_head_path` 存在时，会加载已训练的 vanilla GLACE head
- 否则会用 `glace_encoder_path` + dataset mean 去构建新 regressor

你主要看：

1. `glace_init_head_path` 是否真的被读到了。
2. 没有这个路径时，是否退回到默认构建。
3. 这一步是否决定了后面 head reset / warm-start 的基线。

## 3. memory 进入 GLACE 后，哪些部分被固定

位置：

- [trainer_dinov2_lmc.py](/home/xwh/project/ace_depth/ace_dinov2_lmc/trainer_dinov2_lmc.py:1260)
- [trainer_dinov2_lmc.py](/home/xwh/project/ace_depth/ace_dinov2_lmc/trainer_dinov2_lmc.py:1318)

函数：

- `TrainerACEDINOv2LMC.__init__()`

看这几件事：

1. `global_feats` 是否被加载到 `self.global_feats`
2. `glace_residual_adapter` 是否按 `glace_residual_mode` 初始化
3. `glace_freeze_base_network=True` 时，encoder/head 是否 `eval()` 且 `requires_grad_(False)`
4. `glace_reference_regressor` 是否被复制出来，专门给 guard loss 用

这部分就是“memory 怎么成为 GLACE 的一部分”的第一道门。

## 4. S2 里 memory 怎么融合进 GLACE

位置：

- [trainer_dinov2_lmc.py](/home/xwh/project/ace_depth/ace_dinov2_lmc/trainer_dinov2_lmc.py:4411)
- [trainer_dinov2_lmc.py](/home/xwh/project/ace_depth/ace_dinov2_lmc/trainer_dinov2_lmc.py:4530)

函数：

- `TrainerACEDINOv2LMC._setup_s2_optimizer_and_schedule()`
- `TrainerACEDINOv2LMC._run_s2_polish_phase()`

这里是关键：

1. `ace_g_fusion_in_s2=True` 时，fusion 会参与 S2
2. `ace_g_fusion_lr_ratio` 决定 fusion 的学习率
3. `glace_residual_lr_ratio` 决定 residual adapter 的学习率
4. `freeze_glace_head` 为真时，head 会被排除在 optimizer 之外

你审的时候，优先看：

- `trainer_dinov2_lmc.py:4437-4475`
- `trainer_dinov2_lmc.py:4530-4578`

这两段决定了“memory 融进 GLACE 后，到底哪些参数还能动”。

## 5. 真正的 on-the-fly 融合逻辑

位置：

- [trainer_dinov2_lmc.py](/home/xwh/project/ace_depth/ace_dinov2_lmc/trainer_dinov2_lmc.py:5531)

函数：

- `TrainerACEDINOv2LMC._training_step_ace_g()`

这里是 ACE-G 的核心路径：

1. raw backbone features 进入 `_fuse_features`
2. fusion 输出 `fused_bCHW`
3. `fused_features_bC` 再进入 `_mix_glace_decoder_features`
4. 最后交给 `self.regressor.get_scene_coordinates(...)`

如果你只想盯“memory 怎么混进 GLACE”，主要就是看这条链：

`features_bCHW -> _fuse_features -> fused_features_bC -> _mix_glace_decoder_features -> regressor.get_scene_coordinates`

## 6. guard / 反退化约束

位置：

- [trainer_dinov2_lmc.py](/home/xwh/project/ace_depth/ace_dinov2_lmc/trainer_dinov2_lmc.py:5629)

函数：

- `TrainerACEDINOv2LMC._training_step_ace_g()`

看这里：

1. `glace_reference_regressor` 会先算 base 路径的 reprojection error
2. fused 路径的 error 和 base 路径逐点比较
3. 超过 `margin_px` 的部分才产生 penalty
4. penalty 再乘 `glace_antiregression_weight`

你在日志里看到的：

- `gGain`
- `dRatio`
- `guardPx`
- `basePx`
- `guardLoss`

都来自这一段附近。

## 7. 你最该看的 4 个代码点

如果只看四处，优先顺序是：

1. [trainer_dinov2_lmc.py](/home/xwh/project/ace_depth/ace_dinov2_lmc/trainer_dinov2_lmc.py:1213) - `TrainerACEDINOv2LMC._create_regressor()`
2. [trainer_dinov2_lmc.py](/home/xwh/project/ace_depth/ace_dinov2_lmc/trainer_dinov2_lmc.py:1260) - `TrainerACEDINOv2LMC.__init__()`
3. [trainer_dinov2_lmc.py](/home/xwh/project/ace_depth/ace_dinov2_lmc/trainer_dinov2_lmc.py:4437) - `TrainerACEDINOv2LMC._setup_s2_optimizer_and_schedule()`
4. [trainer_dinov2_lmc.py](/home/xwh/project/ace_depth/ace_dinov2_lmc/trainer_dinov2_lmc.py:5531) - `TrainerACEDINOv2LMC._training_step_ace_g()`

## 8. 这次 run 的对照文件

如果你要对照实际运行值，直接看这个目录下的：

- `run_config.json`
- `best_checkpoint_meta.json`
- `training_full_log.txt`
- `best_K64_it28_wayspots_bears_glace_lmc_frozen_guard_active_eval_log.txt`

目录：

- `/data/xwh/ace_dinov2_lmc/04_evaluation/wayspots_glace_lmc/frozen_guard_active/memory_pooled_vs_asb/extracted/wayspots_bears/dino_ace_lmc_ace_g/20260527_112733_aceg_fS2cie_global_res476_buf3.2M_F8M_K64_it28_ep24_bs10240_spi_s1buf_sp512_onecycle_improved`

