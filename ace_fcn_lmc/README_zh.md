# ACE-FCN-LMC

**ACE**（Accelerated Coordinate Encoding）在 **原始 FCN 编码器** 上接入 **GeoLMC**（Geometric Latent Memory Compression）两阶段迭代训练；在 **Indoor6** 上优于 **ACE-G** 基线，同时保持 ACE 原版推理速度（约 **30 FPS**）。

## 概览

两种训练模式：

- **Vanilla**（`--use_lmc False`）：单阶段 buffer 训练，快速基线  
- **LMC**（`--use_lmc True`）：两阶段迭代（S1 记忆对齐 + S2 重投影精化），需预先构建 **pooled 记忆文件**

## 文件结构

实现文件均在本目录 `ace_fcn_lmc/` 下：

| 文件 | 作用 |
|------|------|
| `ace_network_ace.py` | `ACEEncoder` 封装 + `RegressorACE`（FCN → LMC 接口） |
| `options_ace_lmc.py` | 完整 CLI 参数解析 |
| `trainer_ace_fcn.py` | `TrainerACEFCN` / `TrainerACEFCNLMC`（薄子类） |
| `train_ace_lmc.py` | 训练入口 |
| `test_ace_lmc.py` | 评估脚本 |

项目根目录下的桩脚本（`../train_ace_lmc.py`、`../test_ace_lmc.py`）会转发到上述文件，以保持旧路径可用。

**设计说明**：LMC 训练框架（S1/S2）与 encoder 解耦；在 FCN 上通过专用 regressor 与训练器子类接入，无需重写整套 DINOv2-LMC 管线。

## 快速对照

| 场景 | 关键参数 | 约耗时 |
|------|----------|--------|
| 快速实验 | `--epochs 8 --training_buffer_size 1280000` | ~5 min |
| 标准 Vanilla（生产常用） | `--epochs 16 --training_buffer_size 10000000` | ~15 min |
| 迭代 Vanilla | `--vanilla_iterations 3 --eval_each_iteration True` | ~45 min |
| LMC 快速实验 | `--use_lmc True --lmc_iterations 4` | ~20 min |
| LMC 满配（ACE-G 风格） | `--use_lmc True --lmc_iterations 28` | ~60 min |
| 低显存 | `--buffer_on_cpu True --batch_size 2560` | 依赖 CPU 内存 |

## 训练命令

均在 **`ace_depth/` 项目根目录** 执行。

### 示例 1 — Vanilla 快速（冒烟验证）

```bash
python train_ace_lmc.py \
    /data/xwh/7Scenes/pgt_7scenes_chess \
    chess_quick.pt \
    --encoder_path ace_encoder_pretrained.pt \
    --device cuda:0 \
    --image_resolution 480 \
    --training_buffer_size 1280000 \
    --buffer_batch_size 5 \
    --samples_per_image 256 \
    --epochs 8 \
    --batch_size 5120 \
    --eval_after_train True
```

### 示例 2 — Vanilla 标准（生产）

```bash
python train_ace_lmc.py \
    /data/xwh/7Scenes/pgt_7scenes_heads \
    heads_vanilla.pt \
    --encoder_path ace_encoder_pretrained.pt \
    --device cuda:2 \
    --run_name chess_vanilla \
    --image_resolution 480 \
    --training_buffer_size 10000000 \
    --buffer_batch_size 1 \
    --samples_per_image 512 \
    --epochs 16 \
    --batch_size 5120 \
    --eval_after_train True \
    --eval_session vanilla \
    --use_half True
```

> `--apply_baseline_contract True`（默认）会自动套用 Vanilla 默认的 `num_head_blocks`、`learning_rate_max`、`training_buffer_size`、`epochs` 等。

### 示例 3 — Vanilla 多轮迭代

```bash
python train_ace_lmc.py \
    /data/xwh/7Scenes/pgt_7scenes_chess \
    chess_iter.pt \
    --encoder_path ace_encoder_pretrained.pt \
    --device cuda:0 \
    --run_name chess_iter3 \
    --vanilla_iterations 3 \
    --apply_baseline_contract False \
    --training_buffer_size 2560000 \
    --buffer_batch_size 1 \
    --samples_per_image 512 \
    --epochs 24 \
    --batch_size 5120 \
    --learning_rate_max 0.0001 \
    --eval_each_iteration True \
    --eval_after_train True \
    --keep_best_only True \
    --best_metric pct5 \
    --use_half True
```

### 示例 4 — LMC 快速（快速跑通 LMC）

```bash
python train_ace_lmc.py \
    /data/xwh/7Scenes/pgt_7scenes_chess \
    chess_lmc_quick.pt \
    --encoder_path ace_encoder_pretrained.pt \
    --device cuda:0 \
    --run_name chess_lmc_quick \
    --use_lmc True \
    --memory_path /path/to/7Scenes_chess_train_pooled_GT.pt \
    --lmc_mode global \
    --lmc_flow iterative \
    --lmc_profile legacy \
    --num_latent_tokens 128 \
    --num_attn_layers 4 \
    --lmc_iterations 4 \
    --lmc_warmup_steps 1000 \
    --lmc_train_steps 500 \
    --s1_use_buffer False \
    --s1_loss_mode full_map \
    --s1_batch_size 16 \
    --s1_learning_rate_max 1e-4 \
    --training_buffer_size 2560000 \
    --buffer_batch_size 1 \
    --samples_per_image 768 \
    --epochs 16 \
    --batch_size 5120 \
    --s2_learning_rate_max 1e-3 \
    --image_resolution 480 \
    --eval_after_train True \
    --eval_session lmc_quick
```

### 示例 5 — LMC 满配（ACE-G 风格，生产）

```bash
python train_ace_lmc.py \
    /data/xwh/7Scenes/pgt_7scenes_heads \
    heads_lmc.pt \
    --encoder_path ace_encoder_pretrained.pt \
    --device cuda:2 \
    --run_name heads_lmc_full \
    --use_lmc True \
    --memory_path /home/xwh/project/map-anything-experiments/memory_extract/heads_train/20views/20260127_235757/7Scenes_heads_train_pooled_GT_patch.pt \
    --lmc_mode global \
    --lmc_flow ace_g \
    --lmc_profile legacy \
    --lmc_scene_center_max_distance 4.0 \
    --num_latent_tokens 64 \
    --num_attn_layers 2 \
    --lmc_iterations 28 \
    --lmc_warmup_steps 2000 \
    --lmc_train_steps 600 \
    --s1_use_buffer True \
    --s1_loss_mode sample_per_image \
    --s1_buffer_refill_mode full \
    --s1_batch_size 16 \
    --s1_learning_rate_max 1e-4 \
    --s1_lr_scale_later 1.0 \
    --training_buffer_size 2560000 \
    --buffer_size_final 7680000 \
    --buffer_batch_size 1 \
    --buffer_on_cpu True \
    --samples_per_image 384 \
    --epochs 24 \
    --batch_size 5120 \
    --s2_learning_rate_max 1e-3 \
    --s2_repro_rewind_first_ratio 0.0 \
    --s2_repro_rewind_later_ratio 0.0 \
    --s2_lr_boost_first 1.0 \
    --s2_lr_boost_later 1.0 \
    --image_resolution 480 \
    --freeze_backbone True \
    --use_half True \
    --lmc_memory_preflight True \
    --lmc_memory_preflight_strict True \
    --lmc_lr_scheduler_type onecycle_improved \
    --eval_each_iteration True \
    --eval_after_train True \
    --keep_best_only True \
    --best_metric pct5
```

### 示例 6 — 独立评估

```bash
python test_ace_lmc.py \
    /data/xwh/7Scenes/pgt_7scenes_chess \
    output/7Scenes/pgt_7scenes_chess/ace_fcn_lmc_aceg/.../best_K64_it28_scene3_lmc.pt \
    --encoder_path ace_encoder_pretrained.pt \
    --device cuda:0 \
    --image_resolution 480 \
    --session lmc_eval \
    --hypotheses 64 \
    --threshold 10
```

## 参数参考

### 核心（必填）

| 参数 | 说明 |
|------|------|
| `scene` | 场景目录路径（须含 `train/`） |
| `output_map` | 输出权重文件名后缀（如 `chess.pt`）；实际写在 `output/` 下 |
| `--encoder_path` | `ace_encoder_pretrained.pt` 路径 |
| `--device` | CUDA 设备，如 `cuda:0` |

### Buffer

| 参数 | 默认 | 说明 |
|------|------|------|
| `--training_buffer_size` | 2560000 | buffer 中特征样本数 |
| `--buffer_size_final` | `3×training_buffer_size` | LMC 最后一轮迭代用的最终 buffer 大小 |
| `--s1_last_iter_use_final_buffer` | False | 最后一轮 S1 是否使用 `buffer_size_final`（True）；否则保持 `training_buffer_size`（False） |
| `--buffer_batch_size` | 10 | 填充 buffer 时每步前向的图像数 |
| `--samples_per_image` | 512 | 每图采样特征数 |
| `--buffer_on_cpu` | False | buffer 放 CPU（省显存） |
| `--buffer_sampling_replacement` | True | 允许重复采样 |

显存建议：

| GPU | `training_buffer_size` | `buffer_batch_size` | `buffer_on_cpu` |
|-----|------------------------|---------------------|-----------------|
| 24 GB | 2560000 | 10 | False |
| 16 GB | 1280000 | 5 | False |
| 12 GB | 640000 | 3 | True |

### 训练

| 参数 | 默认 | 说明 |
|------|------|------|
| `--epochs` | 24 | 每轮迭代中 S2 训练 epoch 数 |
| `--batch_size` | 5120 | S2 batch 大小 |
| `--learning_rate_max` | 0.0001 | Vanilla 最大学习率 |
| `--s1_learning_rate_max` | 0.0001 | LMC S1 最大学习率 |
| `--s2_learning_rate_max` | 0.001 | LMC S2 最大学习率 |
| `--use_half` | False | FP16 混合精度（checkpoint 约减半） |
| `--num_head_blocks` | 4 | 回归头深度 |
| `--freeze_backbone` | True | 冻结 FCN encoder |
| `--image_resolution` | 480 | 输入图像高度（无 patch 倍数约束） |

### Vanilla 迭代

| 参数 | 默认 | 说明 |
|------|------|------|
| `--vanilla_iterations` | 1 | 迭代轮数 |
| `--apply_baseline_contract` | True | 自动套用 Vanilla 默认（LR、buffer、epochs 等） |
| `--reset_optimizer_each_iter` | False | 每轮是否重置优化器状态 |

### LMC

| 参数 | 默认 | 说明 |
|------|------|------|
| `--use_lmc` | False | 开启 LMC |
| `--memory_path` | — | POOLED `.pt` 记忆路径（`use_lmc=True` 时必填） |
| `--lmc_mode` | `global` | `global` 或 `local` |
| `--lmc_flow` | `iterative` | `iterative`（经典 S1+S2）或 `ace_g`（ACE-G 风格） |
| `--lmc_profile` | `legacy` | `legacy` 或 `mapany_flow_v1` |
| `--lmc_iterations` | 4 | S1+S2 轮数 |
| `--num_latent_tokens` | 256 | 潜变量 token 数 K |
| `--num_attn_layers` | 2 | 压缩器中 attention 层数 |
| `--lmc_lr_scheduler_type` | `onecycle` | `onecycle` 或 `cosine` |

### 阶段 1（S1）

| 参数 | 默认 | 说明 |
|------|------|------|
| `--lmc_warmup_steps` | 1000 | 首轮迭代 S1 步数 |
| `--lmc_train_steps` | 500 | 后续迭代 S1 步数 |
| `--s1_batch_size` | 16 | S1 batch |
| `--s1_use_buffer` | True | 预填原始 buffer；`False` 为在线 encoder 前向 |
| `--s1_loss_mode` | `sample_per_image` | `full_map` / `sample_per_image` / `sample_pooled` |
| `--s1_buffer_refill_mode` | `full` | `full` 或 `partial`（仅当 `s1_use_buffer=True`） |
| `--s1_lr_scale_later` | 0.2 | 第 2 轮及以后 S1 学习率缩放 |
| `--s1_early_stop` | False | S1 是否早停 |

### 阶段 2（S2）

| 参数 | 默认 | 说明 |
|------|------|------|
| `--s2_repro_rewind_first_ratio` | 0.20 | 首轮迭代重绕比例 |
| `--s2_repro_rewind_later_ratio` | 0.08 | 后续迭代重绕比例 |
| `--s2_lr_boost_first` | 1.2 | 首轮迭代 head LR 放大倍数 |
| `--s2_lr_boost_later` | 1.0 | 后续迭代 head LR 放大倍数 |
| `--s2_lr_warmup_steps` | 100 | S2 学习率 warmup 步数 |

### ACE-G 专用（`--lmc_flow ace_g`）

| 参数 | 默认 | 说明 |
|------|------|------|
| `--ace_g_fusion_in_s2` | False | 在 S2 训练 fusion（R2 路径） |
| `--ace_g_fusion_lr_ratio` | 0.01 | fusion 相对 head 的 LR 比例 |
| `--ace_g_cross_iter_eval` | False | 每轮迭代内在每次 S1 后做评估 |

### 记忆文件校验

| 参数 | 默认 | 说明 |
|------|------|------|
| `--lmc_memory_preflight` | True | 训练前校验记忆文件 |
| `--lmc_memory_preflight_strict` | False | 严格校验（任一不匹配即中止） |
| `--lmc_scene_center_max_distance` | 10.0 | 记忆与场景质心最大允许距离（米） |
| `--lmc_strict_scene_check` | False | 严格场景名检查 |
| `--lmc_strict_center_check` | False | 严格质心距离检查 |

### 评估与输出

| 参数 | 默认 | 说明 |
|------|------|------|
| `--eval_after_train` | True | 训练结束后跑评估 |
| `--eval_each_iteration` | False | 每轮 LMC/Vanilla 迭代后评估 |
| `--keep_best_only` | False | 仅保留最优轮次 checkpoint |
| `--best_metric` | `pct5` | 选优指标：`pct5`、`pct25_5`、`pct10_5`、`pct2`、`median` |
| `--eval_session` | `post_train` | 输出文件会话名后缀 |
| `--post_train_eval_device` | `cuda:0` | 训后评估所用设备 |
| `--run_name` | `auto` | 运行目录名（`auto` = 时间戳 + 配置标签） |
| `--experiment_root` | `output` | 所有输出根目录 |
| `--output_layout` | `hierarchical` | `hierarchical`（默认）或 `legacy` |

## 输出目录结构

```
output/
└── {dataset}/
    └── {scene}/
        ├── ace_fcn_vanilla/
        │   └── {timestamp}_{config_tag}/
        │       ├── run_config.json          # 完整配置快照
        │       ├── run_command.txt          # 实际执行命令
        │       ├── run_metadata.json        # ResultManager 元数据
        │       ├── training_full_log.txt    # 完整训练日志
        │       ├── training_summary.json    # 训练统计
        │       ├── best_*.pt               # 最优 checkpoint
        │       ├── post_train_eval.txt      # 制表符分隔的评估指标
        │       └── eval_results/
        │           ├── {ts}_post_train_eval.json
        │           └── {ts}_post_train_eval.txt
        ├── ace_fcn_lmc_iter/               # lmc_flow=iterative
        │   └── {timestamp}_{config_tag}/
        └── ace_fcn_lmc_aceg/              # lmc_flow=ace_g
            └── {timestamp}_{config_tag}/
                └── iteration_results/     # 每轮迭代评估结果
```

`config_tag` 示例：

- vanilla：`vanilla_buf2.6M_bs5120_ep16`
- iterative：`iter_global_res480_buf2.6M_F7.7M_K128_it4_ep16_bs5120_fm_s1enc_sp768_onecycle`
- ace_g：`aceg_fS2cie_global_res480_buf2.6M_F7.7M_K64_it28_ep24_bs5120_spi_s1buf_sp384_cosine`

## 记忆文件格式

Pooled 记忆 `.pt` 须为字典：

```python
{
    'pooled_points':   torch.Tensor,  # (N, 3)   三维点
    'pooled_features': torch.Tensor,  # (N, 512) FCN 特征
    'scene_center':    torch.Tensor,  # (3,)     场景质心
}
```

## 性能（7-Scenes Chess 典型量级）

| 模式 | 训练时间 | 显存 | 5cm/5deg |
|------|----------|------|----------|
| Vanilla（1×） | ~5 min | ~6 GB | ~45% |
| Vanilla（3×） | ~15 min | ~6 GB | ~50% |
| LMC（4 iter） | ~20 min | ~8 GB | ~52% |
| LMC（28 iter） | ~60 min | ~8 GB | ~55%+ |

## 常见问题

**Q：训练 OOM？**  
1. 减小 `--batch_size`（如 2560）  
2. 减小 `--training_buffer_size`  
3. 设 `--buffer_on_cpu True`  
4. 减小 `--buffer_batch_size`

**Q：训后评估 OOM？**  
加 `--post_train_eval_device cpu`。

**Q：Vanilla 还是 LMC？**  
- Vanilla：快、无需记忆文件  
- LMC：精度更高，需预先构建 pooled 记忆

**Q：迭代何时结束？**  
- 固定轮数：`--vanilla_iterations N`  
- 按轮次选最优 checkpoint：`--eval_each_iteration True --keep_best_only True`

**Q：与 DINOv2-LMC 区别？**

| | ACE FCN | DINOv2 |
|---|---------|--------|
| Encoder | FCN（512 维，8× 下采样） | ViT-L/14（1024 维，14×） |
| 输入 | 灰度（内部 RGB→灰） | RGB |
| 分辨率 | 无 patch 约束 | 须为 14 的倍数 |
| 推理速度 | ~30 FPS | ~10–15 FPS |
| 显存 | 较低 | 较高 |

## 研究工作流状态

| 阶段 | 状态 |
|------|------|
| 1. 提案 | 完成 |
| 2. 评审 | 完成 |
| 3. 架构 | 完成 |
| 4. 骨架 | 完成 |
| 5. 实现 | 完成 |
| 6. 评估 | 待完成 |

触发阶段 6：将评估日志拷贝到 `04_evaluation/logs/`（可用 `04_evaluation/collect_logs.py`），再运行 `/research-eval ace_fcn_lmc`。

## 更多文档

- `00_ideas/idea.md` — 想法与动机  
- `00_ideas/reuse_map.md` — 复用契约  
- `01_design/proposal.md` — 提案  
- `02_architecture/system_design.md` — 系统设计  
- `03_implementation/implementation_notes.md` — 实现说明  
- `04_evaluation/eval_plan.md` — 评估计划  
- **[README.md](README.md)** — 英文版（与本文档同步；命令与默认值以代码为准）
