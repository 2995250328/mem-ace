# ACE-DINOv2-LMC

ACE（加速坐标编码）结合 DINOv2 ViT-L/14 骨干网络与 GeoLMC（几何潜在记忆压缩）两阶段迭代训练。

## 概述

两种训练模式：

- **Vanilla**（`--use_lmc False`）：单阶段 buffer 训练，快速基线
- **LMC**（`--use_lmc True`）：两阶段迭代训练（S1 记忆对齐 + S2 重投影精化），需要预构建的 pooled memory 文件

## 文件结构

所有实现文件位于本目录：

| 文件 | 作用 |
|---|---|
| `train_ace_dinov2_lmc.py` | 训练入口 |
| `test_ace_dinov2_lmc.py` | 评估脚本 |
| `options_dinov2_lmc.py` | 完整 CLI 参数解析器 |
| `trainer_dinov2_lmc.py` | `TrainerACEDINOv2LMC` 两阶段训练器 |

共享的根目录依赖（通过 `sys.path` 访问）：

| 文件 | 作用 |
|---|---|
| `../trainer_dinov2.py` | 基础 `TrainerACEDINOv2` |
| `../ace_network_dinov2.py` | DINOv2 `Regressor` |
| `../dataset_dinov2.py` | ACE 格式数据集加载器 |
| `../dataset_wai_dinov2.py` | WAI 格式数据集加载器 |
| `../utils_lmc.py` | 共享 LMC 工具函数 |
| `../ace_compressor.py` | `GeoLMC` 压缩器 |
| `../ace_fusion.py` | `LMCFeatureFusion` |
| `../ace_loss.py` | `ReproLoss` |
| `../ace_util.py` | 通用工具函数 |

根目录的 stub 文件（`../train_ace_dinov2_lmc.py` 等）会转发到本目录，保持向后兼容。

## 训练命令

所有命令从**项目根目录**（`ace_depth/`）运行。

### ACE-G flow（主要用法）

<!-- Updated 2026-03-30: clarified memory file format and training stages -->
实验中使用的标准配置，**需要预构建的 pooled memory 文件**（由 `memory_extraction/extract_memory.sh` 输出，通常命名为 `memory_bse.pt`）。部分旧脚本中 `*_GT_patch.pt` 命名指同一格式。

**S1/S2 迭代训练说明：**
- **S1（记忆对齐）**：训练 GeoLMC 压缩器 + 特征融合，将记忆与 backbone 特征对齐
- **S2（重投影精化）**：在融合特征上训练回归头，使用重投影损失
- 交替循环 `--lmc_iterations`（默认 28）轮，每轮结束后评估并保留最优 checkpoint

**示例中未列出的常用参数：**
- `--buffer_on_cpu True`：训练缓冲区放 CPU 节省显存（默认 True）
- `--data_backend ace/wai`：数据集格式（默认 ace）
- `--use_half True`：半精度训练（默认 True）
- `--best_metric {pct5,pct10_5,composite,rt_error}`：最优 checkpoint 选择指标（默认 pct5）

**7-Scenes 示例：**
```bash
python train_ace_dinov2_lmc.py \
    /mnt/storage/xwh/7Scenes/pgt_7scenes_heads \
    heads_aceg_full_refill.pt \
    --device cuda:3 \
    --run_name heads_aceg_full_refill \
    --use_lmc True \
    --memory_path /home/xwh/project/map-anything-experiments/memory_extract/heads_train/20views/20260127_235757/7Scenes_heads_train_pooled_GT_patch.pt \
    --dinov2_path /mnt/storage/xwh/checkpoints/dinov2_vitl14_pretrain.pth \
    --lmc_flow ace_g \
    --lmc_mode global \
    --num_latent_tokens 64 \
    --lmc_iterations 28 \
    --lmc_warmup_steps 2000 \
    --lmc_train_steps 600 \
    --s1_learning_rate_max 1e-4 \
    --s2_learning_rate_max 1e-3 \
    --training_buffer_size 2560000 \
    --buffer_size_final 7680000 \
    --epochs 24 \
    --batch_size 5120 \
    --samples_per_image 384 \
    --buffer_batch_size 1 \
    --buffer_on_cpu True \
    --image_resolution 518 \
    --s1_batch_size 16 \
    --s1_use_buffer True \
    --s1_loss_mode sample_per_image \
    --s1_buffer_refill_mode full \
    --ace_g_fusion_in_s2 True \
    --ace_g_fusion_lr_ratio 0.01 \
    --ace_g_cross_iter_eval True \
    --lmc_memory_preflight True \
    --lmc_memory_preflight_strict True \
    --eval_each_iteration True \
    --eval_after_train True \
    --keep_best_only True \
    --best_metric pct5 \
    --use_half True
```

**Indoor6 示例**（大场景，额外加 `--lmc_scene_center_max_distance 4.0`）：
```bash
python train_ace_dinov2_lmc.py \
    /mnt/storage/xwh/indoor6_ace/scene3 \
    scene3_aceg_full_refill.pt \
    --device cuda:3 \
    --run_name scene3_aceg_full_refill \
    --use_lmc True \
    --memory_path /home/xwh/project/map-anything-experiments/memory_extract/scene3_train/40views/20260315_105429/Indoor6_scene3_train_pooled_GT.pt \
    --lmc_scene_center_max_distance 4.0 \
    --dinov2_path /mnt/storage/xwh/checkpoints/dinov2_vitl14_pretrain.pth \
    --lmc_flow ace_g \
    --lmc_mode global \
    --num_latent_tokens 64 \
    --lmc_iterations 28 \
    --lmc_warmup_steps 2000 \
    --lmc_train_steps 600 \
    --s1_learning_rate_max 1e-4 \
    --s2_learning_rate_max 1e-3 \
    --training_buffer_size 2560000 \
    --buffer_size_final 7680000 \
    --epochs 24 \
    --batch_size 5120 \
    --samples_per_image 384 \
    --buffer_batch_size 1 \
    --buffer_on_cpu True \
    --image_resolution 518 \
    --s1_batch_size 16 \
    --s1_use_buffer True \
    --s1_loss_mode sample_per_image \
    --s1_buffer_refill_mode full \
    --ace_g_fusion_in_s2 True \
    --ace_g_fusion_lr_ratio 0.01 \
    --ace_g_cross_iter_eval True \
    --lmc_memory_preflight True \
    --lmc_memory_preflight_strict True \
    --eval_each_iteration True \
    --eval_after_train True \
    --keep_best_only True \
    --best_metric pct5 \
    --use_half True
```

### Vanilla（不启用 LMC，快速基线）

```bash
python train_ace_dinov2_lmc.py \
    /mnt/storage/xwh/7Scenes/pgt_7scenes_chess \
    chess_vanilla.pt \
    --use_lmc False \
    --device cuda:0
```

## 测试

```bash
python test_ace_dinov2_lmc.py \
    /mnt/storage/xwh/7Scenes/pgt_7scenes_heads \
    output/.../best_K64_it28_heads_aceg_full_refill.pt \
    --device cuda:0 \
    --session lmc_test
```

## 研究阶段

| 文件夹 | 内容 |
|---|---|
| `00_ideas/` | 原始想法笔记和复用地图 |
| `01_design/` | 提案和设计文档 |
| `02_architecture/` | 系统架构 |
| `03_implementation/` | 实现笔记 |
| `04_evaluation/` | 评估计划和日志 |
