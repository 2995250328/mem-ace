# ACE DINOv2 LMC 使用与实现说明

## 1. 文档目的

本文档对应当前 `ace_depth` 最新实现，目标是：
- 准确说明 **S1/S2 两阶段迭代训练**；
- 明确列出与 `map-anything` 对齐的策略（学习率调度、每轮评估、最优权重策略等）；
- 给出可直接复现的启动命令与调参建议；
- 说明日志与输出文件结构，方便长期实验管理。

> 说明：若代码更新，文档应以 `train_ace_dinov2_lmc.py` 与 `trainer_dinov2_lmc.py` 为准同步修改。

---

## 2. 总体训练流程

当前 LMC 训练为 **迭代式两阶段**（每轮 iteration 包含 S1 + S2）：

1. **S1（在线全监督）**
   - 使用训练集图像 batch；
   - 前向路径：`backbone -> compressor -> fusion -> head`；
   - 损失：使用 `ace_depth` 原生 reprojection 计算公式（不是 map-anything 的 `_loss_fn`）；
   - 更新参数：`compressor + fusion + head`；
   - 支持 map-anything 风格可选 LR 调度（`auto/warmup_cosine/warmup_plateau_cosine/onecycle_improved/onecycle_legacy`）。

2. **S2（离线 buffer 头部训练）**
   - 每轮先压缩一次 memory token，再填充训练 buffer（该轮内复用 token）；
   - 只训练 `head`（head-only optimizer）；
   - 使用 S2 专用学习率（`head_lr_multiplier_s2` + 首轮/后续 boost）；
   - ReproLoss step 使用“全局连续 + 局部回拨衰减”策略（`step_eff`）。

3. **每轮结束**
   - 可选执行一次 eval（默认开启）；
   - 按 `best_metric` 评估得分；
   - 默认仅保留一个最优权重文件（`output_map`）。

---

## 3. Memory 文件要求（非常关键）

只支持 **POOLED** memory 文件：

- 必需键：
  - `pooled_points`
  - `pooled_features`
- 可选键：
  - `scene_center`（缺失时自动用 `pooled_points` 均值）
  - `all_scale_tokens`
  - `layers_idx`

不支持 FULL 中间格式（含 `intermediate/final`）直接训练；请先转成 pooled。

### 维度推导规则（与 map-anything 对齐）

- `num_layers`：优先 `len(layers_idx)`，否则默认 4。
- `feature_dim`：`pooled_features_dim // num_layers`。
- GeoLMC 使用：`input_dim = compress_dim = feature_dim`。
- Fusion 使用：`query_feature_dim = 1024`，`memory_feature_dim = feature_dim`，输出仍为 1024 给 head。

---

## 4. 学习率与损失调度策略

## 4.1 S1 学习率策略（可选）

参数：`--lmc_lr_scheduler_type`

- `auto`（默认）
  - iter0: `warmup_plateau_cosine`
  - iter1+: `warmup_cosine`
- `warmup_cosine`
- `warmup_plateau_cosine`
- `onecycle_improved`
- `onecycle_legacy`

相关参数：
- `--lmc_min_lr_ratio`
- `--lmc_warmup_ratio`
- `--lmc_plateau_ratio`
- `--lmc_lr_pct_start`
- `--lmc_lr_div_factor`
- `--lmc_lr_final_div_factor`

## 4.2 S2 学习率策略

S2 为 head-only 训练，学习率由 **S2 专用** `s2_learning_rate_max` 控制（默认 1e-3，与 DINO ACE head 阶段一致），不再使用全局 `learning_rate_max`：

`head_lr = s2_learning_rate_max * head_lr_multiplier_s2 * boost`

其中：
- 第一轮 S2 boost = `s2_lr_boost_first`（默认 1.5）
- 后续轮次 boost = `s2_lr_boost_later`（默认 1.2）

并使用 OneCycleLR，支持 `s2_lr_warmup_steps` 控制预热。S1 使用 `s1_learning_rate_max`（默认 1e-4），与 S2 解耦。

## 4.3 S2 ReproLoss step 精细化（重点）

不是直接用一个全局 step，而是：

- 维护 `global_s2_step`（跨轮次连续）
- 维护 `local_s2_step`（当前轮次 S2 内计数）
- 每轮开始引入回拨：
  - 首轮回拨比例：`s2_repro_rewind_first_ratio`（默认 0.20）
  - 后续回拨比例：`s2_repro_rewind_later_ratio`（默认 0.08）
- 回拨按指数衰减：`exp(-local_s2_step / s2_repro_rewind_tau)`

最终用于损失的步数：`step_eff`。

这样可在新一轮 token 分布变化时，前期稍快收敛，同时保持全局训练稳定。

---

## 5. 迭代评估与最优权重策略

支持参数：
- `--eval_each_iteration`（默认 True）
- `--keep_best_only`（默认 True）
- `--best_metric`（`pct5/pct10_5/composite`）

行为：
- 每轮末保存临时 checkpoint；
- 跑评估打分；
- 若优于历史 best，替换为最终 `output_map`；
- 若不是 best 且 `keep_best_only=True`，删除该轮临时权重。

> 结果：训练结束后通常只剩一个最优模型文件，目录更整洁。

---

## 6. 日志与输出文件

### 6.1 输出目录布局

- **默认（`--output_layout legacy`）**：`run_dir = <experiment_root>/<run_name>`，其中 `run_name` 为时间戳+场景+参数自动生成。
- **层级布局（`--output_layout hierarchical`）**：  
  `run_dir = <experiment_root>/<dataset>/<scene>/<function>/<run_id>`  
  - `dataset`：场景父目录名（如 `pgt_7scenes`）  
  - `scene`：场景名（如 `chess`）  
  - `function`：`dino_ace_baseline` 或 `dino_ace_lmc_s1s2`  
  - `run_id`：同 run_name  
  便于按数据集/场景/功能对比实验。

### 6.2 单次 run 下的文件（均在 run_dir 内）

- **权重**：`output_map`（如 `chess_lmc.pt`）为最终保留的最优权重。
- **复现与追溯**：
  - `run_config.json`：本次运行全部参数。
  - `run_command.txt`：完整启动命令。
- **日志**：
  - `training_full_log.txt`：控制台全文日志。
  - `training_log.txt`：逐步 step 日志。
  - `<stem>_training_log.txt`：迭代摘要 CSV。
  - `<stem>_eval_log.txt`：每轮评估摘要。
- **最优 checkpoint 元信息**（LMC 且每轮评估时）：  
  - `best_checkpoint_meta.json`：`best_iter`、`best_score`、`best_checkpoint_path`、`pct5`、`median_tErr`、`median_rErr`。
- 若启用训练后评估：还会产生 test 脚本的 `test_*.txt`、`poses_*.txt` 等。

迭代摘要 CSV 字段：  
`iter,s1_steps,s2_epochs,buffer_size,is_best,score,pct5,median_tErr_cm,median_rErr_deg,elapsed_s`

---

## 7. 当前 CLI 参数（按功能分组）

## 7.1 通用训练
- `scene`（必填）
- `output_map`（必填）
- `--device`（默认 `cuda:0`）
- `--dinov2_path`
- `--freeze_backbone`
- `--num_head_blocks`
- `--use_homogeneous`
- `--training_buffer_size`（默认 `1920000`）
- `--buffer_batch_size`（默认 `10`）
- `--buffer_image_width`
- `--samples_per_image`（默认 `384`）
- `--epochs`（默认 `16`）
- `--batch_size`（默认 `3840`）
- `--learning_rate_min`（默认 `1e-4`）
- `--learning_rate_max`（默认 `1e-3`）
- `--image_resolution`（默认 `518`）
- `--use_aug` / `--aug_rotation` / `--aug_scale`
- `--use_half`

## 7.2 ReproLoss 相关
- `--repro_loss_type`（默认 `dyntanh`）
- `--repro_loss_soft_clamp`（默认 `50`）
- `--repro_loss_soft_clamp_min`（默认 `1`）
- `--repro_loss_schedule`（默认 `circle`）
- `--repro_loss_hard_clamp`（默认 `1000`）
- `--depth_min` / `--depth_max` / `--depth_target`

## 7.3 LMC 核心
- `--use_lmc`
- `--memory_path`
- `--lmc_mode`
- `--num_latent_tokens`
- `--num_attn_layers`
- `--use_scale_token`
- `--s1_learning_rate_max`（默认 `1e-4`，S1 专用）
- `--s2_learning_rate_max`（默认 `1e-3`，S2 专用，与 DINO ACE head 一致）
- `--lmc_iterations`（默认 `15`）
- `--lmc_train_steps`（默认 `600`）
- `--lmc_warmup_steps`（默认 `3000`）
- `--buffer_size_final`（默认 `3 * training_buffer_size`）
- `--head_reset_strategy`
- `--head_lr_multiplier_s2`
- `--output_layout`（`legacy` | `hierarchical`，默认 `legacy`）

## 7.4 S1 LR 策略
- `--lmc_lr_scheduler_type`
- `--lmc_min_lr_ratio`
- `--lmc_warmup_ratio`
- `--lmc_plateau_ratio`
- `--lmc_lr_pct_start`
- `--lmc_lr_div_factor`
- `--lmc_lr_final_div_factor`

## 7.5 S2 Repro/LR 精细化
- `--s2_repro_rewind_first_ratio`
- `--s2_repro_rewind_later_ratio`
- `--s2_repro_rewind_tau`
- `--s2_lr_boost_first`
- `--s2_lr_boost_later`
- `--s2_lr_warmup_steps`

## 7.6 评估与最优保存
- `--eval_each_iteration`
- `--keep_best_only`
- `--best_metric`
- `--eval_after_train`
- `--eval_session`

---

## 8. 推荐启动模板

### 8.1 推荐运行指令（标准 LMC，与 DINO ACE 对齐）

以下命令在 7Scenes/chess 上做完整 LMC 训练，S1 用较小学习率、S2 用 1e-3（与 DINO ACE head 训练一致），输出使用层级目录便于对比实验：

```bash
python train_ace_dinov2_lmc.py \
  /mnt/storage/xwh/7Scenes/pgt_7scenes_chess \
  output/chess_lmc.pt \
  --device cuda:0 \
  --use_lmc True \
  --memory_path /path/to/chess_pooled.pt \
  --lmc_mode global \
  --num_latent_tokens 64 \
  --lmc_iterations 15 \
  --buffer_batch_size 10 \
  --training_buffer_size 5760000 \
  --samples_per_image 384 \
  --batch_size 3840 \
  --image_resolution 518 \
  --epochs 16 \
  --s1_learning_rate_max 1e-4 \
  --s2_learning_rate_max 1e-3 \
  --output_layout hierarchical \
  --eval_each_iteration True \
  --keep_best_only True \
  --best_metric pct5
```

说明：
- **S1/S2 学习率**：`s1_learning_rate_max` 默认 1e-4，S2 用 `s2_learning_rate_max` 默认 1e-3，与 DINO ACE head 阶段一致，避免 S2 欠拟合。
- **buffer/epoch/batch**：与 DINO ACE 一致（5760000 / 384 / 3840 / 16），便于和 baseline 对比。
- **output_layout hierarchical**：输出到 `run_root/<dataset>/<scene>/dino_ace_lmc_s1s2/<run_id>/`，便于按数据集、场景、功能区分。

### 8.2 退化到 DINO ACE（baseline 验证）

不启用 LMC 时，行为与 `train_ace_dinov2.py` 对齐（同一套默认 buffer/epoch/LR）：

```bash
python train_ace_dinov2_lmc.py \
  /mnt/storage/xwh/7Scenes/pgt_7scenes_chess \
  output/chess_baseline.pt \
  --device cuda:0 \
  --buffer_batch_size 10 \
  --training_buffer_size 5760000 \
  --samples_per_image 384 \
  --batch_size 3840 \
  --image_resolution 518 \
  --epochs 16 \
  --use_lmc False
```

### 8.3 快速 smoke test

```bash
python train_ace_dinov2_lmc.py \
  /mnt/storage/xwh/7Scenes/pgt_7scenes_chess \
  output/smoke.pt \
  --device cuda:0 \
  --use_lmc True \
  --memory_path /path/to/chess_pooled.pt \
  --lmc_iterations 2 \
  --lmc_warmup_steps 50 \
  --lmc_train_steps 20 \
  --epochs 1 \
  --training_buffer_size 300000 \
  --eval_each_iteration True
```

### 8.4 显存吃紧时

```bash
--batch_size 2048 \
--buffer_batch_size 5 \
--training_buffer_size 1200000
```

---

## 9. 常见问题排查

### 9.1 `--use_lmc requires --memory_path`
需要提供 pooled memory 文件路径。

### 9.2 `FULL intermediate format` 报错
当前输入是 FULL 格式，请改为 pooled 文件。

### 9.3 `mat1 and mat2 shapes cannot be multiplied`
通常是 memory 维度与 `layers_idx` 不匹配。检查：
- `pooled_features.shape[-1]`
- `len(layers_idx)`
- 二者是否整除。

### 9.4 OOM
优先下调：
- `batch_size`
- `buffer_batch_size`
- `training_buffer_size`

### 9.5 训练日志好看但指标不涨
重点看：
- `*_eval_log.txt` 每轮 `pct5/median` 变化；
- S2 前几百步是否出现明显改善（受 `s2_repro_rewind_*` 和 `s2_lr_boost_*` 影响）。

---

## 10. Citation

```bibtex
@inproceedings{brachmann2023ace,
  title={Accelerated Coordinate Encoding: Learning to Relocalize in Minutes using RGB and Poses},
  author={Brachmann, Eric and Cavallari, Tommaso and Prisacariu, Victor Adrian},
  booktitle={CVPR},
  year={2023}
}
```
