# ACE FCN + LMC 实现文档

## 概述

本文档介绍 ACE FCN 与 Latent Memory Compression (LMC) 的集成实现。该实现将原始 ACE 的 FCN encoder 与 LMC 两阶段训练框架结合，支持 vanilla 迭代训练和 LMC 压缩训练两种模式。

## 核心改动

### 1. `ace_network_ace.py` - ACE Encoder 适配器

**目的**：为 ACE FCN encoder 提供统一的接口，使其与 LMC 框架兼容。

**关键类**：

```python
class ACEEncoder(nn.Module):
    """ACE FCN encoder 的包装器，提供与 DINOv2 一致的接口"""
    def __init__(self, encoder_path, device='cuda:0'):
        self.encoder = load_ace_encoder(encoder_path)
        self.feature_dim = 512  # ACE FCN 特征维度
        self.downsampling = 8   # 8x 下采样

    def forward(self, x):
        # x: (B, 1, H, W) 灰度图
        # 返回 (B, 512, H/8, W/8)
        return self.encoder(x)
```

**特点**：
- 支持灰度图输入（1 通道）
- 输出 512 维特征，8x 下采样
- 与 DINOv2 接口一致

### 2. `options_ace_lmc.py` - 参数解析器

**目的**：定义 ACE FCN LMC 训练的所有命令行参数。

**关键参数**：

```python
# Encoder 配置
--encoder_path          # ACE FCN encoder 权重路径（必需）
--image_resolution      # 输入分辨率（默认 480，无 patch-size 约束）

# LMC 配置
--use_lmc               # 启用 LMC（默认 False）
--memory_path           # POOLED 格式内存文件路径
--num_latent_tokens     # 潜在令牌数（默认 256）
--lmc_iterations        # LMC 迭代次数（默认 4）

# Vanilla 迭代配置
--vanilla_iterations    # Vanilla 迭代次数（默认 1）
--apply_baseline_contract  # 应用 vanilla 默认值（默认 True）

# 训练配置
--batch_size            # 批大小（默认 5120）
--training_buffer_size  # 训练缓冲大小（默认 2560000）
--epochs                # 训练轮数（默认 24）
--learning_rate_max     # 最大学习率（默认 0.0001）
```

### 3. `trainer_ace_fcn.py` - 训练器实现

**目的**：实现 ACE FCN 的训练逻辑，包括 vanilla 和 LMC 两种模式。

**核心类**：

```python
class TrainerACEFCN:
    """ACE FCN vanilla 训练器"""
    def __init__(self, args):
        self.encoder = ACEEncoder(args.encoder_path)
        self.regressor = RegressorACE(...)  # 坐标回归头
        self.training_buffer = TrainingBuffer(...)

    def create_training_buffer(self, buffer_size):
        """从训练图像填充缓冲"""
        # 编码所有训练图像
        # 采样特征并存储到 GPU 缓冲

    def run_epoch(self):
        """运行单个训练轮"""
        # 从缓冲采样批次
        # 前向传播 + 损失计算 + 反向传播

    def save_model(self, path):
        """保存模型检查点"""

class TrainerACEFCNLMC(TrainerACEFCN):
    """ACE FCN LMC 训练器（两阶段）"""
    def train(self):
        """执行 LMC 两阶段训练"""
        # Stage 1: 从内存填充缓冲 + 训练回归头
        # Stage 2: 迭代优化（可选）
```

**关键方法**：

- `create_training_buffer(buffer_size)` - 从训练图像填充缓冲
- `run_epoch()` - 单轮训练
- `reset_optimizer_scheduler()` - 重置优化器（用于迭代）
- `save_model(path)` - 保存检查点

### 4. `test_ace_lmc.py` - 评估脚本

**目的**：评估训练好的 ACE FCN 模型。

**核心函数**：

```python
def run_evaluation(args):
    """Vanilla 评估"""
    # 加载模型
    # 对每个测试图像预测坐标
    # 使用 RANSAC 估计姿态
    # 计算误差指标

def run_evaluation_lmc(args):
    """LMC 评估（支持内存）"""
    # 与 vanilla 相同，但支持 LMC 特定配置
```

**输出指标**：
- `median_rErr` - 中位数旋转误差（度）
- `median_tErr` - 中位数平移误差（cm）
- `pct5` - 5cm/5deg 精度百分比
- `pct25_5` - 25cm/5deg 精度百分比
- `avg_time` - 平均推理时间（秒）

### 5. `train_ace_lmc.py` - 训练入口

**目的**：主训练脚本，支持 vanilla 和 LMC 两种模式。

**核心函数**：

```python
def run_vanilla_iterative_baseline(args):
    """运行 vanilla 迭代训练"""
    # 多轮迭代，每轮填充缓冲 + 训练
    # 支持每轮评估和最佳检查点选择

def _apply_lmc_profile(args):
    """应用 LMC 配置文件默认值"""

def _build_run_dir(args):
    """构建输出目录结构"""
    # 层级结构：dataset/scene/function/run_id/

def setup_experiment(args):
    """设置实验（验证、配置、日志）"""

def run_post_train_eval(args, trainer):
    """训练后评估"""

def main():
    """主函数"""
```

## 使用方法

### 完整命令示例对比

  python train_ace_lmc.py \                                                                
      /data/xwh/7Scenes/pgt_7scenes_chess \                                                
      output/chess_lmc_simple.pt \                                                         
      --encoder_path ace_encoder_pretrained.pt \                                           
      --device cuda:0 \
      --image_resolution 480 \
      \
      # ── LMC 开关 ──────────────────────────────────────────────
      --use_lmc True \
      --memory_path /home/xwh/project/map-anything-experiments/memory_extract/chess_train/2
  0views/20260114_164907/7Scenes_chess_train_pooled_GT.pt \
      --lmc_mode global \
      --lmc_flow iterative \          # iterative | ace_g
      --lmc_profile legacy \          # legacy | mapany_flow_v1
      \
      # ── 迭代规模 ──────────────────────────────────────────────
      --lmc_iterations 4 \
      --lmc_warmup_steps 1000 \       # 第1轮 S1 步数
      --lmc_train_steps 500 \         # 后续轮 S1 步数
      \
      # ── S1 配置 ───────────────────────────────────────────────
      --s1_use_buffer False \         # True=预填 raw buffer；False=online encoder 前向
      --s1_loss_mode full_map \       # full_map | sample_per_image | sample_pooled
      --s1_batch_size 16 \
      --s1_buffer_refill_mode full \  # full | partial（仅 s1_use_buffer=True 时生效）
      --s1_learning_rate_max 1e-4 \
      --s1_early_stop True \
      \
      # ── S2 配置 ───────────────────────────────────────────────
      --training_buffer_size 2560000 \
      --buffer_size_final None \      # None=3x training_buffer_size；或显式指定
      --buffer_batch_size 1 \
      --buffer_on_cpu True \
      --samples_per_image 768 \
      --epochs 16 \
      --batch_size 5120 \
      --s2_learning_rate_max 1e-3 \
      --head_reset_strategy first_only \  # first_only | every | output_only | none
      \
      # ── ACE-G 专属（lmc_flow=ace_g 时生效）────────────────────
      --ace_g_fusion_in_s2 False \    # True=S2 中训练 fusion（R2路径）
      --ace_g_fusion_lr_ratio 0.01 \
      --ace_g_cross_iter_eval False \
      \
      # ── 网络结构 ──────────────────────────────────────────────
      --num_head_blocks 4 \
      --num_latent_tokens 128 \
      --num_attn_layers 4 \
      --freeze_backbone True \
      --use_half False \
      \
      # ── Memory 校验 ───────────────────────────────────────────
      --lmc_memory_preflight True \
      --lmc_memory_preflight_strict False \
      --lmc_strict_scene_check True \
      --lmc_strict_center_check True \
      --lmc_scene_center_max_distance 3.0 \
      \
      # ── 评估与输出 ────────────────────────────────────────────
      --run_name chess_lmc_simple \
      --eval_each_iteration True \
      --eval_after_train True \
      --eval_session lmc_simple \
      --keep_best_only True \
      --best_metric pct5

#### 示例 1：Vanilla 基础训练（快速实验）

```bash
./train_ace_lmc.py \
    /data/xwh/7scenes_chess \
    output/chess_vanilla_quick.pt \
    --encoder_path ace_encoder_pretrained.pt \
    --device cuda:0 \
    --image_resolution 480 \
    --batch_size 5120 \
    --training_buffer_size 1280000 \
    --buffer_batch_size 5 \
    --samples_per_image 256 \
    --epochs 8 \
    --learning_rate_max 0.0001 \
    --eval_after_train True
```

**特点**：快速、低显存、用于快速验证

---

#### 示例 2：Vanilla 完整训练（生产级）

```bash
./train_ace_lmc.py \
    /data/xwh/7Scenes/pgt_7scenes_chess \
    output/chess_vanilla_full.pt \
    --encoder_path ace_encoder_pretrained.pt \
    --device cuda:0 \
    --run_name chess_vanilla_full \
    --image_resolution 480 \
    --batch_size 5120 \
    --training_buffer_size 8000000 \
    --buffer_batch_size 10 \
    --samples_per_image 512 \
    --epochs 24 \
    --learning_rate_max 0.0001 \
    --eval_after_train True \
    --eval_session vanilla_full \
    --use_half True
```

**特点**：完整训练、高精度、标准配置

---

#### 示例 3：Vanilla 迭代训练（多轮优化）

```bash
./train_ace_lmc.py \
    /data/xwh/7scenes_chess \
    output/chess_vanilla_iter.pt \
    --encoder_path ace_encoder_pretrained.pt \
    --device cuda:0 \
    --run_name chess_vanilla_iter \
    --vanilla_iterations 3 \
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

**特点**：多轮迭代、自动选择最佳、逐步优化

---

#### 示例 4：LMC 简化训练（快速 LMC 实验）

```bash
python train_ace_lmc.py \
    /data/xwh/7Scenes/pgt_7scenes_chess \
    output/chess_lmc_simple.pt \
    --encoder_path ace_encoder_pretrained.pt \
    --device cuda:0 \
    --use_lmc True \
    --memory_path /home/xwh/project/map-anything-experiments/memory_extract/chess_train/20views/20260114_164907/7Scenes_chess_train_pooled_GT.pt \
    --run_name chess_lmc_simple \
    --image_resolution 480 \
    --num_latent_tokens 128 \
    --lmc_iterations 4 \
    --lmc_warmup_steps 1000 \
    --lmc_train_steps 500 \
    --training_buffer_size 2560000 \
    --buffer_batch_size 1 \
    --samples_per_image 768 \
    --batch_size 5120 \
    --epochs 16 \
    --s1_learning_rate_max 1e-4 \
    --s2_learning_rate_max 1e-3 \
    --eval_after_train True \
    --eval_session lmc_simple
```

**特点**：LMC 快速实验、较低显存需求

---

#### 示例 5：LMC 完整训练（生产级，类似 ACE-G）

```bash
./train_ace_lmc.py \
    /data/xwh/indoor6_ace/scene3 \
    scene3_lmc_full.pt \
    --encoder_path ace_encoder_pretrained.pt \
    --device cuda:1 \
    --run_name scene3_lmc_full \
    --use_lmc True \
    --memory_path /home/xwh/project/memory_extract/scene3_train/40views/Indoor6_scene3_train_pooled_GT.pt \
    --lmc_scene_center_max_distance 4.0 \
    --lmc_mode global \
    --num_latent_tokens 64 \
    --num_attn_layers 2 \
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
    --image_resolution 480 \
    --s1_batch_size 16 \
    --eval_each_iteration True \
    --eval_after_train True \
    --keep_best_only True \
    --best_metric pct5 \
    --use_half True \
    --lmc_memory_preflight True \
    --lmc_memory_preflight_strict True \
    --s1_use_buffer True \
    --s1_loss_mode sample_per_image \
    --s1_buffer_refill_mode full \
    --s1_lr_scale_later 1.0 \
    --s2_repro_rewind_first_ratio 0.0 \
    --s2_repro_rewind_later_ratio 0.0 \
    --s2_lr_boost_first 1.0 \
    --s2_lr_boost_later 1.0 \
    --lmc_lr_scheduler_type cosine \
    --lmc_profile legacy
```

**特点**：完整 LMC 训练、高精度、所有特性启用

---

#### 示例 6：评估已训练模型

```bash
./test_ace_lmc.py \
    /data/xwh/7scenes_chess \
    output/chess_lmc_full.pt \
    --encoder_path ace_encoder_pretrained.pt \
    --device cuda:0 \
    --image_resolution 480 \
    --session lmc_eval \
    --hypotheses 64 \
    --threshold 10
```

**特点**：独立评估、可指定 RANSAC 参数

---

## 命令参数速查表

| 场景 | 关键参数 | 推荐值 |
|------|---------|--------|
| 快速实验 | `--epochs 8 --training_buffer_size 1280000` | 5 分钟 |
| 标准训练 | `--epochs 24 --training_buffer_size 2560000` | 15 分钟 |
| 迭代优化 | `--vanilla_iterations 3 --eval_each_iteration True` | 45 分钟 |
| LMC 快速 | `--use_lmc True --lmc_iterations 4` | 20 分钟 |
| LMC 完整 | `--use_lmc True --lmc_iterations 28` | 60 分钟 |
| 低显存 | `--buffer_on_cpu True --batch_size 2560` | 需要 CPU 内存 |
| 高精度 | `--use_half False --epochs 32` | 更多显存 |

## 架构对比

| 特性 | ACE Vanilla | ACE + LMC |
|------|------------|-----------|
| Encoder | FCN (512-dim, 8x) | FCN (512-dim, 8x) |
| 输入 | 灰度图 | 灰度图 |
| 训练模式 | 单阶段 | 两阶段 |
| 内存需求 | 低 | 中等（可压缩） |
| 训练速度 | 快 | 中等 |
| 精度 | 基准 | 可能更高 |

### 缓冲配置

```python
--training_buffer_size      # 训练缓冲大小（样本数，默认 2560000）
--buffer_size_final         # 最终缓冲大小（用于 LMC，默认 training_buffer_size * 3）
--buffer_batch_size         # 填充缓冲时的批大小（默认 10）
--samples_per_image         # 每张图像采样的特征数（默认 512）
--buffer_on_cpu             # 缓冲存储在 CPU（节省 GPU 显存，默认 False）
--buffer_sampling_replacement  # 采样时是否允许重复（默认 True）
```

**推荐值**：
- 24GB GPU: `--training_buffer_size 2560000 --buffer_batch_size 10 --buffer_on_cpu False`
- 16GB GPU: `--training_buffer_size 1280000 --buffer_batch_size 5 --buffer_on_cpu False`
- 12GB GPU: `--training_buffer_size 640000 --buffer_batch_size 3 --buffer_on_cpu True`

### 训练配置

```python
--epochs                    # 训练轮数（默认 24）
--batch_size                # 批大小（默认 5120）
--learning_rate_max         # Vanilla 最大学习率（默认 0.0001）
--s1_learning_rate_max      # LMC Stage 1 最大学习率（默认 0.0001）
--s2_learning_rate_max      # LMC Stage 2 最大学习率（默认 0.001）
--use_half                  # 使用 FP16 混合精度（默认 False）
```

### LMC 特定参数

```python
--use_lmc                   # 启用 LMC（默认 False）
--memory_path               # POOLED 格式内存文件路径（必需，当 use_lmc=True）
--num_latent_tokens         # 潜在令牌数（默认 256）
--num_attn_layers           # 注意力层数（默认 2）
--lmc_iterations            # LMC 迭代次数（默认 4）
--lmc_warmup_steps          # Stage 1 预热步数（默认 1000）
--lmc_train_steps           # Stage 1 训练步数（默认 500）
--lmc_mode                  # LMC 模式：'global' 或 'local'（默认 'global'）
--lmc_lr_scheduler_type     # 学习率调度器：'onecycle' 或 'cosine'（默认 'onecycle'）
--lmc_profile               # 配置文件：'legacy' 或 'mapany_flow_v1'（默认 'legacy'）
```

### Vanilla 迭代参数

```python
--vanilla_iterations        # Vanilla 迭代次数（默认 1）
--apply_baseline_contract   # 应用 vanilla 默认值（默认 True）
--reset_optimizer_each_iter # 每轮迭代重置优化器（默认 False）
```

### 评估参数

```python
--eval_each_iteration       # 每轮迭代后评估（默认 False）
--eval_after_train          # 训练后评估（默认 True）
--keep_best_only            # 仅保留最佳检查点（默认 False）
--best_metric               # 最佳指标：'pct5', 'pct25_5', 'median' 等（默认 'pct5'）
--eval_session              # 评估 session 名称（默认 'post_train'）
--post_train_eval_device    # 评估设备（默认 'cuda:0'）
```

### 内存检查参数

```python
--lmc_memory_preflight      # 启用内存预检查（默认 True）
--lmc_memory_preflight_strict  # 严格内存检查（默认 False）
--lmc_scene_center_max_distance  # 场景中心最大距离（默认 10.0 m）
--lmc_strict_scene_check    # 严格场景检查（默认 False）
--lmc_strict_center_check   # 严格中心检查（默认 False）
```

### Stage 1 参数

```python
--s1_batch_size             # Stage 1 批大小（默认 16）
--s1_use_buffer             # Stage 1 使用缓冲（默认 True）
--s1_loss_mode              # Stage 1 损失模式：'full_map' 或 'sample_per_image'（默认 'sample_per_image'）
--s1_buffer_refill_mode     # 缓冲补充模式：'full' 或 'partial'（默认 'full'）
--s1_lr_scale_later         # Stage 1 后期学习率缩放（默认 0.2）
```

### Stage 2 参数

```python
--s2_repro_rewind_first_ratio   # Stage 2 前期回退比例（默认 0.20）
--s2_repro_rewind_later_ratio   # Stage 2 后期回退比例（默认 0.08）
--s2_lr_boost_first             # Stage 2 前期学习率提升（默认 1.2）
--s2_lr_boost_later             # Stage 2 后期学习率提升（默认 1.0）
--s2_lr_warmup_steps            # Stage 2 预热步数（默认 100）
```

### 输出配置

```python
--experiment_root           # 实验根目录（默认 'output'）
--output_layout             # 输出布局：'hierarchical' 或 'flat'（默认 'hierarchical'）
--overwrite_run_dir         # 覆盖已存在的运行目录（默认 True）
```

## 输出结构

### 层级化目录组织

```
output/
├── {dataset}/
│   └── {scene}/
│       ├── ace_fcn_vanilla/
│       │   └── {timestamp}_{config_tag}/
│       │       ├── run_metadata.json          # 运行配置元数据
│       │       ├── training_summary.json      # 训练摘要（JSON）
│       │       ├── training_summary.txt       # 训练摘要（易读文本）
│       │       ├── best_*.pt                  # 最佳检查点
│       │       ├── training_full_log.txt      # 完整训练日志
│       │       ├── post_train_eval.txt        # 训练后评估结果
│       │       └── eval_results/
│       │           ├── {timestamp}_post_train_eval.json
│       │           ├── {timestamp}_post_train_eval.txt
│       │           └── ...
│       ├── ace_fcn_lmc_iter/          # lmc_flow=iterative（经典 S1+S2 迭代）
│       │   └── {timestamp}_{config_tag}/
│       └── ace_fcn_lmc_aceg/          # lmc_flow=ace_g（含 fS2/fS1、cie 等在 config_tag）
│           └── {timestamp}_{config_tag}/
│               ├── run_metadata.json
│               ├── training_summary.json
│               ├── training_summary.txt
│               ├── best_*.pt
│               ├── training_full_log.txt
│               ├── post_train_eval.txt
│               ├── eval_results/
│               └── iteration_results/
└── results_index.json                         # 全局结果索引
```

**`config_tag`**（`utils_lmc.build_lmc_run_folder_config_tag`）示例：

- **iter**：`iter_global_res480_buf2.6M_F7.7M_K128_it4_ep16_bs5120_fm_s1enc_sp768_onecycle_improved`
- **ace_g（S2 训 fusion）**：`aceg_fS2cie_global_res518_buf2.6M_F7.7M_K64_it28_ep24_bs5120_spi_s1buf_sp384_onecycle_improved`

字段含义：`aceg`/`iter`；`fS2`=ACE-G 在 S2 训练 fusion，`fS1`=仅 S1 相关；`cie`=cross-iter eval；`fm`/`spi`/`spool`=S1 loss；`s1buf`/`s1enc`；`buf`/`F`=S2 首末 buffer 规模（M）；`res`/`K`/`it`/`ep`/`bs`/`sp`=分辨率、latent 数、迭代轮、S2 epoch、batch、每图采样数。

### 文件说明

**run_metadata.json** - 运行配置元数据
```json
{
  "timestamp": "2026-03-15T10:30:45.123456",
  "algorithm": "ace_fcn_lmc",
  "dataset": "7scenes",
  "scene": "chess",
  "scene_path": "/data/xwh/7scenes_chess",
  "encoder_path": "ace_encoder_pretrained.pt",
  "device": "cuda:0",
  "use_lmc": true,
  "image_resolution": 480,
  "training_buffer_size": 2560000,
  "batch_size": 5120,
  "epochs": 24,
  "learning_rate_max": 0.0001,
  "memory_path": "/path/to/memory_pooled.pt",
  "num_latent_tokens": 64,
  "lmc_iterations": 28,
  "lmc_mode": "global",
  "s1_learning_rate_max": 0.0001,
  "s2_learning_rate_max": 0.001
}
```

**training_summary.json** - 训练摘要
```json
{
  "timestamp": "2026-03-15T10:45:30.654321",
  "training_stats": {
    "total_time": 3600.5,
    "best_iter": 12,
    "best_score": 99.8,
    "best_metric": "pct5",
    "num_iterations": 28,
    "final_checkpoint": "/path/to/best_K64_it28_chess_lmc.pt"
  }
}
```

**training_summary.txt** - 易读的训练摘要
```
================================================================================
TRAINING SUMMARY
================================================================================
Timestamp: 2026-03-15T10:45:30.654321

total_time............................ 3600.5000
best_iter............................ 12
best_score........................... 99.8000
best_metric.......................... pct5
num_iterations....................... 28
final_checkpoint..................... /path/to/best_K64_it28_chess_lmc.pt
================================================================================
```

**eval_results/{timestamp}_{session}_eval.json** - 评估结果（JSON）
```json
{
  "timestamp": "2026-03-15T10:50:00.123456",
  "session": "post_train",
  "results": {
    "median_rErr": 0.52,
    "median_tErr": 1.67,
    "pct5": 99.8,
    "pct25_5": 100.0,
    "pct10_5": 100.0,
    "pct2": 95.2,
    "pct1": 45.3,
    "avg_time": 0.045,
    "total_frames": 1000
  }
}
```

**eval_results/{timestamp}_{session}_eval.txt** - 评估结果（易读文本）
```
================================================================================
EVALUATION RESULTS (post_train)
================================================================================
Timestamp: 2026-03-15T10:50:00.123456

Accuracy Metrics:
  25cm/5deg:  100.00%
  10cm/5deg:  100.00%
   5cm/5deg:   99.80%
   2cm/2deg:   95.20%
   1cm/1deg:   45.30%

Error Metrics:
  Median Rotation Error:     0.5219°
  Median Translation Error:     1.6714 cm

Performance:
  Avg Time per Frame:    45.00 ms
  Total Frames:       1000
================================================================================
```

**results_index.json** - 全局结果索引
```json
{
  "7scenes": {
    "chess": {
      "ace_fcn_vanilla": [
        {
          "run_name": "20260315_103045_vanilla_buf2.6M_bs5120_ep24",
          "run_path": "/path/to/output/7scenes/chess/ace_fcn_vanilla/...",
          "has_metadata": true,
          "has_summary": true,
          "has_eval": true,
          "timestamp": "2026-03-15T10:30:45.123456",
          "use_lmc": false,
          "best_score": 95.5
        }
      ],
      "ace_fcn_lmc": [
        {
          "run_name": "20260315_104530_global_buf2.6M_K64_it28_bs5120_onecycle",
          "run_path": "/path/to/output/7scenes/chess/ace_fcn_lmc/...",
          "has_metadata": true,
          "has_summary": true,
          "has_eval": true,
          "timestamp": "2026-03-15T10:45:30.654321",
          "use_lmc": true,
          "best_score": 99.8
        }
      ]
    }
  }
}
```

### 结果查询

使用 `ResultManager` 快速查询结果：

```python
from result_manager import ResultManager

mgr = ResultManager("output")

# 创建结果索引
mgr.create_results_index()

# 打印结果摘要
mgr.print_results_summary()
```

输出示例：
```
================================================================================
RESULTS SUMMARY
================================================================================

Dataset: 7scenes
  Scene: chess
    Algorithm: ace_fcn_vanilla (5 runs)
      - 20260315_103045_vanilla_buf2.6M_bs5120_ep24 (score: 95.5, time: 2026-03-15T10:30:45.123456)
      - 20260314_180000_vanilla_buf2.6M_bs5120_ep24 (score: 94.2, time: 2026-03-14T18:00:00.000000)
      - 20260313_150000_vanilla_buf2.6M_bs5120_ep24 (score: 93.8, time: 2026-03-13T15:00:00.000000)
    Algorithm: ace_fcn_lmc (3 runs)
      - 20260315_104530_global_buf2.6M_K64_it28_bs5120_onecycle (score: 99.8, time: 2026-03-15T10:45:30.654321)
      - 20260314_190000_global_buf2.6M_K64_it28_bs5120_onecycle (score: 98.5, time: 2026-03-14T19:00:00.000000)
      - 20260313_160000_global_buf2.6M_K64_it28_bs5120_onecycle (score: 97.2, time: 2026-03-13T16:00:00.000000)

================================================================================
```

## 结果管理系统

### ResultManager 类

`result_manager.py` 提供结构化的结果管理功能。

**主要方法**：

```python
from result_manager import ResultManager

mgr = ResultManager(experiment_root="output")

# 1. 解析场景信息
scene_info = mgr.parse_scene_info("/data/xwh/7scenes_chess")
# 返回: {'dataset': '7scenes', 'scene': 'chess'}

# 2. 构建层级化目录
run_dir = mgr.build_hierarchical_path(
    scene_path="/data/xwh/7scenes_chess",
    algorithm="ace_fcn_lmc",
    config_tag="global_buf2.6M_K64_it28_bs5120",
    timestamp=None  # 使用当前时间戳
)
# 返回: output/7scenes/chess/ace_fcn_lmc/20260315_103045_global_buf2.6M_K64_it28_bs5120/

# 3. 保存运行元数据
mgr.save_run_metadata(run_dir, args, algorithm="ace_fcn_lmc", scene_info=scene_info)

# 4. 保存训练摘要
training_stats = {
    'total_time': 3600.5,
    'best_iter': 12,
    'best_score': 99.8,
    'best_metric': 'pct5',
    'num_iterations': 28,
    'final_checkpoint': str(run_dir / 'best_K64_it28_chess_lmc.pt'),
}
mgr.save_training_summary(run_dir, training_stats)

# 5. 保存评估结果
eval_results = {
    'median_rErr': 0.52,
    'median_tErr': 1.67,
    'pct5': 99.8,
    'pct25_5': 100.0,
    'pct10_5': 100.0,
    'pct2': 95.2,
    'pct1': 45.3,
    'avg_time': 0.045,
    'total_frames': 1000,
}
mgr.save_evaluation_results(run_dir, eval_results, session="post_train")

# 6. 保存迭代结果
mgr.save_iteration_results(run_dir, iteration=1, eval_results=eval_results)

# 7. 创建全局结果索引
mgr.create_results_index()

# 8. 打印结果摘要
mgr.print_results_summary()
```

### 自动集成

训练脚本已自动集成 ResultManager：

```bash
./train_ace_lmc.py \
    /data/xwh/7scenes_chess \
    output/chess_lmc.pt \
    --encoder_path ace_encoder_pretrained.pt \
    --device cuda:0 \
    --use_lmc True \
    --memory_path /path/to/memory_pooled.pt
```

训练完成后，自动生成：
- ✅ `run_metadata.json` - 运行配置
- ✅ `training_summary.json` + `training_summary.txt` - 训练摘要
- ✅ `eval_results/` - 评估结果（JSON + 易读文本）
- ✅ `results_index.json` - 全局索引

### 快速查询

```bash
# 查看所有训练结果
python3 << 'EOF'
from result_manager import ResultManager
mgr = ResultManager("output")
mgr.create_results_index()
mgr.print_results_summary()
EOF
```

### 结果对比

```python
import json
from pathlib import Path

# 加载结果索引
with open("output/results_index.json") as f:
    index = json.load(f)

# 查找特定数据集和场景的所有运行
dataset = "7scenes"
scene = "chess"
runs = index[dataset][scene]

# 对比 vanilla 和 LMC
vanilla_runs = runs.get("ace_fcn_vanilla", [])
lmc_runs = runs.get("ace_fcn_lmc", [])

print(f"Vanilla best score: {max(r['best_score'] for r in vanilla_runs)}")
print(f"LMC best score: {max(r['best_score'] for r in lmc_runs)}")
```



### Q: 如何选择 vanilla 还是 LMC？

**A**:
- **Vanilla**: 快速实验、基准测试
- **LMC**: 需要更好精度、有内存文件可用

### Q: 内存文件格式要求？

**A**: 必须是 POOLED 格式，包含：
```python
{
    'pooled_points': torch.Tensor,      # (N, 3) 3D 点
    'pooled_features': torch.Tensor,    # (N, 512) 特征
    'scene_center': torch.Tensor,       # (3,) 场景中心
}
```

### Q: 如何处理 OOM？

**A**:
1. 减少 `--batch_size`
2. 减少 `--training_buffer_size`
3. 增加 `--buffer_batch_size`（如果内存允许）

### Q: 迭代训练何时停止？

**A**:
- 固定轮数：`--vanilla_iterations N`
- 自动停止：`--eval_each_iteration True` + `--keep_best_only True`

## 性能指标

**典型性能**（7-Scenes Chess）：

| 模式 | 训练时间 | 内存 | 5cm/5deg |
|------|---------|------|----------|
| Vanilla | ~5 min | 6GB | ~45% |
| Vanilla x3 | ~15 min | 6GB | ~50% |
| LMC | ~10 min | 8GB | ~52% |

## 文件清单

| 文件 | 行数 | 功能 |
|------|------|------|
| `ace_network_ace.py` | ~150 | ACE encoder 适配器 |
| `options_ace_lmc.py` | ~400 | 参数解析器 |
| `trainer_ace_fcn.py` | ~800 | 训练器实现 |
| `test_ace_lmc.py` | ~600 | 评估脚本 |
| `train_ace_lmc.py` | ~550 | 训练入口 |

## 与 DINOv2 版本的区别

| 方面 | ACE FCN | DINOv2 |
|------|---------|--------|
| Encoder | FCN (512-dim) | ViT-L/14 (1024-dim) |
| 输入 | 灰度图 | RGB 图 |
| 下采样 | 8x | 14x |
| 分辨率约束 | 无 | 必须是 14 的倍数 |
| 推理速度 | ~30 FPS | ~10-15 FPS |
| 内存需求 | 低 | 高 |

## 扩展建议

1. **多场景训练**：使用 `--num_clusters` 进行空间聚类
2. **集成评估**：训练多个头网络，合并结果
3. **自定义 encoder**：替换 `ACEEncoder` 实现
4. **特征融合**：集成深度、SuperPoint 等模态

## 训练后自动评估

### 功能说明

训练完成后，系统会自动运行评估（可通过 `--eval_after_train` 控制）。

**流程**：
1. 训练完成，保存最佳检查点
2. 释放 trainer 占用的 GPU 显存
3. 加载检查点进行评估
4. 输出评估指标到 `post_train_eval.txt`

### 参数配置

```python
--eval_after_train          # 是否训练后评估（默认 True）
--eval_session              # 评估 session 名称（默认 'post_train'）
--post_train_eval_device    # 评估使用的设备（默认 'cuda:0'）
```

### 工作流程

**Vanilla 模式**：
```
训练 → 保存最佳检查点 → 评估 → 输出结果
```

**LMC 模式**：
```
Stage 1 + Stage 2 → 保存检查点 → 评估 → 输出结果
```

### 输出文件

评估完成后在 `run_dir` 下生成 `post_train_eval.txt`：

```
median_rotation_deg         5.23
median_translation_cm       12.45
accuracy_25cm5deg_pct       85.50
accuracy_10cm5deg_pct       72.30
accuracy_5cm5deg_pct        45.20
accuracy_2cm2deg_pct        15.80
accuracy_1cm1deg_pct        3.20
avg_time_per_frame_ms       45.67
total_frames                1000
```

### 禁用评估

```bash
./train_ace_lmc.py \
    datasets/7scenes_chess \
    output/chess.pt \
    --encoder_path ace_encoder_pretrained.pt \
    --eval_after_train False
```

### 显存不足处理

如果评估时 OOM，可指定 CPU 评估：

```bash
./train_ace_lmc.py \
    datasets/7scenes_chess \
    output/chess.pt \
    --encoder_path ace_encoder_pretrained.pt \
    --post_train_eval_device cpu
```

## 训练后立即测试的实现细节

### 核心逻辑

`train_ace_lmc.py` 中的 `main()` 函数：

```python
def main():
    parser = get_lmc_train_parser()
    args = parser.parse_args()
    setup_experiment(args)

    if not args.use_lmc:
        # Vanilla 模式
        trainer = run_vanilla_iterative_baseline(args)
        _logger.info("Training completed.")
        run_post_train_eval(args, trainer)  # ← 训练后评估
        return

    # LMC 模式
    trainer = TrainerACEFCNLMC(args)
    trainer.train()
    _logger.info("Training completed.")
    run_post_train_eval(args, trainer)  # ← 训练后评估
```

### 两种评估方式

**1. Vanilla 内置评估**（`run_vanilla_iterative_baseline` 中）：
```python
if vanilla_args.eval_after_train:
    result = _run_standard_eval(vanilla_args, vanilla_args.output_map, vanilla_args.eval_session)
    _write_vanilla_eval_summary(vanilla_args, result)
```

**2. 统一后处理评估**（`run_post_train_eval` 中）：
```python
def run_post_train_eval(args, trainer):
    if not args.eval_after_train:
        return
    # 释放 trainer 显存
    del trainer
    gc.collect()
    torch.cuda.empty_cache()

    # 运行评估
    result = run_evaluation_lmc(eval_opt)
    # 保存结果
```

### 显存管理

评估前自动释放训练器占用的显存：

```python
del trainer              # 删除 trainer 对象
gc.collect()            # 触发垃圾回收
torch.cuda.empty_cache()  # 清空 CUDA 缓存
torch.cuda.synchronize()  # 同步 GPU
```

这确保评估有足够的显存，即使训练时已接近 OOM。

## 参考

- 原始 ACE 论文：CVPR 2023
- LMC 框架：见 `train_ace_dinov2_lmc.py`
- DSAC* 姿态估计：`dsacstar/` 模块



