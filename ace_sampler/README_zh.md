# ACE Sampler

面向 ACE 训练的可学习缓冲区采样模块。将均匀随机采样替换为基于置信度引导的策略，该策略由重投影误差监督训练，并可选地结合 MC Dropout 不确定性估计进行增强。

## 概述

标准 ACE 通过对每张图像随机采样像素来填充训练缓冲区。本模块训练一个轻量级 `SamplerNet`，从编码器特征中预测逐像素置信度，再利用置信度将采样偏向当前 ACE 模型定位效果较好的点。

**两阶段工作流：**

- **第 1 阶段** — 以冻结 ACE 模型的重投影误差（以及可选的 MC Dropout 不确定性）作为监督信号，端到端训练 `SamplerNet`。
- **第 2 阶段** — 将训练好的 `SamplerNet` 插入 `ace_trainer.py` 的缓冲区填充流程，作为 `torch.multinomial` 的即插即用替代。

## 架构

### SamplerNet（可配置容量）

通过 `--mid_channels` 和 `--num_branches` 提供两种容量模式：

```
small（mid_channels=128，num_branches=2）— 约 107K 参数，单场景训练
─────────────────────────────────────────────────────────────────────────────
Stage 1: DW-Conv(512,k=3) → GN(32) → PW-Conv(128) → GN(32) → SiLU
Stage 2: 2 分支 Lite-ASPP（dilation 1,3），各 → 32ch，concat → 64ch
Stage 3: ECA-lite(64)
Stage 4: Conv1×1(64→1) → Sigmoid

large（mid_channels=256，num_branches=3）— 约 366K 参数，多场景训练
─────────────────────────────────────────────────────────────────────────────
Stage 1: DW-Conv(512,k=3) → GN(32) → PW-Conv(256) → GN(32) → SiLU
Stage 2: 3 分支 Lite-ASPP（dilation 1,3,5），各 → 64ch，concat → 192ch
         + residual_proj(256→192)
Stage 3: ECA-lite(192)
Stage 4: Conv1×1(192→64) → GN(32) → SiLU → Conv1×1(64→1) → Sigmoid
```

两种模式均使用 GroupNorm（非 BatchNorm），batch_size=1 时安全。

### UncertaintyHead（可选，MC Dropout）

带 Dropout 的轻量级坐标预测器，用于不确定性估计：

```
Input:  (B, 512, H/8, W/8)  — 冻结编码器特征
        Conv1x1(512 → 256) → Dropout2d(p) → ReLU → Conv1x1(256 → 3)
Output: (B, 3, H/8, W/8)  — 场景坐标预测
```

在 `train()` 模式下运行 T 次 → 重投影后 2D 位置的方差 = 逐像素不确定性。

**结合 MC Dropout 的置信度目标：**
```
conf = exp(-(α × repro_error + β × mc_variance))
```

低重投影误差且预测方差低 → 高置信度 → 优先采样。

## 快速测试（使用预训练权重）

跳过训练，直接评估已下载的 head：

```bash
python test_ace.py \
    /data/xwh/7Scenes/pgt_7scenes_chess \
    output/ace_models/7Scenes_pgt/pgt_7scenes_chess.pt \
    --device cuda:0
```

`output/ace_models/7Scenes_pgt/` 中可用的预训练 head：

| 场景 | 文件 |
|---|---|
| chess | `pgt_7scenes_chess.pt` |
| fire | `pgt_7scenes_fire.pt` |
| heads | `pgt_7scenes_heads.pt` |
| office | `pgt_7scenes_office.pt` |
| pumpkin | `pgt_7scenes_pumpkin.pt` |
| redkitchen | `pgt_7scenes_redkitchen.pt` |
| stairs | `pgt_7scenes_stairs.pt` |

结果保存在 `.pt` 文件同目录下：
- `test_<scene>_.txt` — 旋转/平移误差中位数及平均推理时间
- `poses_<scene>_.txt` — 逐帧位姿估计及误差

## 第 1 阶段：训练 SamplerNet

> **`--encoder_path` 说明**：ACE 写入输出 `.pt` 的主要是场景相关 head 权重（约 4 MB）；编码器单独保存在 `ace_encoder_pretrained.pt`。`train_ace_sampler.py` 会分别加载二者并在运行时合并。默认 `--encoder_path ace_encoder_pretrained.pt` 相对于项目根解析，因此在项目根执行时一般无需再写该参数。

> **输出路径说明**：`sampler_output` 仅作为**文件名主干**的提示。脚本会把实际保存位置重定向到  
> `ace_sampler/04_evaluation/{dataset}/{scene}/{timestamp}_{stem}.pt`。启动时会打印解析后的完整路径。

### 基础（仅重投影误差）

```bash
python train_ace_sampler.py \
    /data/xwh/7Scenes/pgt_7scenes_chess \
    output/ace_models/7Scenes_pgt/pgt_7scenes_chess.pt \
    chess_sampler.pt \
    --device cuda:0
# 保存至：ace_sampler/04_evaluation/7Scenes_pgt/pgt_7scenes_chess/<timestamp>_chess_sampler.pt
```

### 启用 MC Dropout 不确定性

```bash
python train_ace_sampler.py \
    /data/xwh/7Scenes/pgt_7scenes_chess \
    output/ace_models/7Scenes_pgt/pgt_7scenes_chess.pt \
    chess_sampler_mc.pt \
    --device cuda:0 \
    --use_mc_dropout True \
    --mc_samples 10 \
    --mc_dropout_p 0.1 \
    --sampler_beta 0.01
# 保存至：ace_sampler/04_evaluation/7Scenes_pgt/pgt_7scenes_chess/<timestamp>_chess_sampler_mc.pt
```

## 第 2 阶段：用引导采样训练 ACE

训练结束后会自动在测试划分上评估；若需跳过，传入 `--eval_after_train False`。

```bash
python train_ace.py \
    /data/xwh/7Scenes/pgt_7scenes_chess \
    output/chess_guided.pt \
    --sampler_path output/chess_sampler_mc.pt \
    --sampler_ratio 0.7
```

结果与输出 `.pt` 同目录：
- `test_<scene>_.txt` — 旋转/平移误差中位数及平均推理时间
- `poses_<scene>_.txt` — 逐帧位姿估计及误差

未指定 `--sampler_path` 时，`train_ace.py` 行为与原先一致（随机采样 + 训练后自动评估）。

## 完整参数说明

### 第 1 阶段 — `train_ace_sampler.py`

| 参数 | 默认值 | 说明 |
|---|---|---|
| `scene` | — | 场景根目录（须含 `train/`） |
| `ace_head` | — | 预训练 ACE head `.pt`（第 1 阶段冻结） |
| `sampler_output` | — | 训练得到的 SamplerNet `.pt` 输出路径 |
| `--encoder_path` | `ace_encoder_pretrained.pt` | 预训练 FCN 编码器权重 |
| `--num_head_blocks` | 1 | ACE head 深度（须与加载的 head 一致） |
| `--use_homogeneous` | `False` | 加载的 head 是否使用齐次坐标 |
| `--image_resolution` | 480 | 输入图像高度（像素） |
| `--use_aug` | `True` | 是否数据增强（旋转 + 缩放） |
| `--aug_rotation` | 15 | 最大平面内旋转角（度） |
| `--aug_scale` | 1.5 | 最大缩放因子（范围 `[1/aug_scale, aug_scale]`） |
| `--sampler_epochs` | 5 | SamplerNet 训练轮数 |
| `--sampler_lr` | 1e-3 | SamplerNet 学习率（AdamW） |
| `--sampler_alpha` | 0.1 | 重投影误差权重：`exp(-α × error_px)`。α 越大目标越“尖锐”（对大误差惩罚更重） |
| `--use_half` | `True` | 是否 FP16 混合精度 |
| `--device` | `cuda` | GPU 设备（如 `cuda:0`、`cuda:1`） |
| `--use_mc_dropout` | `False` | 是否启用 MC Dropout 不确定性 |
| `--mc_samples` | 10 | 每张图的随机前向次数 T |
| `--mc_dropout_p` | 0.1 | UncertaintyHead 的 Dropout 概率 |
| `--sampler_beta` | 0.01 | 不确定性权重：`exp(-β × mc_variance_px²)`。β 越大对不确定性的惩罚越强 |
| `--uncertainty_head_path` | `None` | 预训练 UncertaintyHead `.pt` 路径。若为 `None` 且 `--use_mc_dropout True`，则在第 1 阶段主训练前用 2 轮蒸馏从头训练该 head |

## 多场景通用 Sampler（推荐）

单场景训练容易使采样器过拟合该场景的统计特性；在多样场景上联合训练可学到与场景无关的可定位性线索（无纹理区域、重复纹理、高光等）。

### Cambridge：聚类 head 与单场景 head

Cambridge Landmarks 使用空间聚类（ACE Poker）：每个场景拆成 N 个空间簇，每簇有专用 head。用**聚类 head** 训练采样器时，重投影误差图比单场景 head 更准确。

场景列表格式支持可选后缀 `num_clusters cluster_idx`：

```
# 完整场景（7Scenes、12Scenes）
scene_path  head_path

# 空间聚类（Cambridge 集成）
scene_path  head_path  num_clusters  cluster_idx
```

`configs/sampler_all.txt` 对 Cambridge 使用 4 簇 head（共 35 条）。  
`configs/sampler_cambridge_ensemble.txt` 仅 Cambridge（20 条聚类项）。

### 步骤 1：准备场景列表

`configs/` 中可用配置：

| 文件 | 场景 | 条数 |
|---|---|---|
| `sampler_7scenes_pgt.txt` | 7Scenes（当前 3 个可用） | 3 |
| `sampler_cambridge_ensemble.txt` | Cambridge（5 场景 × 4 簇） | 20 |
| `sampler_all.txt` | 7Scenes + 12Scenes + Cambridge 簇 | 35 |

### 步骤 2：训练通用 SamplerNet

> **输出路径说明**：与单场景相同，脚本会将输出重定向到  
> `ace_sampler/04_evaluation/universal/{timestamp}_{stem}.pt`。

```bash
# 快速：3 个场景
python train_ace_sampler_multi.py \
    ace_sampler/configs/sampler_7scenes_pgt.txt \
    universal_sampler.pt \
    --device cuda:0

# 完整：35 条（7S + 12S + Cambridge 簇）
python train_ace_sampler_multi.py \
    ace_sampler/configs/sampler_all.txt \
    universal_sampler_all.pt \
    --device cuda:0 \
    --sampler_epochs 10
# 保存至：ace_sampler/04_evaluation/universal/<timestamp>_universal_sampler_all.pt
```

### 步骤 3：在第 2 阶段使用（与单场景采样器相同）

```bash
python train_ace.py \
    /data/xwh/Cambridge/Cambridge_GreatCourt \
    output/GreatCourt_guided.pt \
    --device cuda:0 \
    --sampler_path ace_sampler/04_evaluation/universal/20260323_223249_universal_sampler_all.pt \
    --sampler_ratio 0.7
```

### 多场景参数说明

| 参数 | 默认值 | 说明 |
|---|---|---|
| `scene_list` | — | 文本文件：每行 `scene head [num_clusters cluster_idx]` |
| `sampler_output` | — | 输出 `.pt`（会带时间戳重写到 eval 目录） |
| `--encoder_path` | `ace_encoder_pretrained.pt` | 共享 FCN 编码器权重 |
| `--sampler_epochs` | 10 | 训练轮数 |
| `--sampler_lr` | 5e-4 | 学习率（数据更大，低于单场景默认） |
| `--sampler_alpha` | 0.1 | 置信度目标锐度 |
| `--mid_channels` | 256 | SamplerNet 宽度：128=small(81K)，256=large(254K) |
| `--num_branches` | 3 | Lite-ASPP 分支数：2 或 3 |
| `--device` | `cuda` | GPU 设备 |

### 第 2 阶段 — `train_ace.py`（新增参数）

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--sampler_path` | `None` | 训练好的 SamplerNet `.pt`。为 `None` 时使用原始随机采样（行为与旧版一致） |
| `--sampler_ratio` | 0.7 | `samples_per_image` 中由**高置信度 top-k** 采样的比例（Part A）；其余 `1 - ratio` 为均匀随机（Part B） |
| `--use_neighbors` | `False` | 是否在特征分辨率上将每个 top-k 选择扩展为其 4 邻域 |
| `--eval_after_train` | `True` | 训练结束后是否立即在测试划分上评估 |

## 采样策略

对每张图像，`fill_buffer_with_sampler` 共采 `samples_per_image` 个点：

- **Part A**（`ratio × spi` 个点）— 按 SamplerNet 置信度取 top-k，并由 `image_mask` 约束
- **Part B**（`(1-ratio) × spi` 个点）— 在 `image_mask` 内均匀随机（与原始 ACE 相同）

Part B 保证当前模型尚未学好的区域仍有机会被采到（冷启动稳健性）。

## MC Dropout 流程

当 `--use_mc_dropout True` 时，直接对冻结的 ACE head 做 MC 推理，无需额外网络或预训练：

1. **注入 Dropout**：通过 forward hook 在 head 每个 Conv2d 后插入 `Dropout2d`，不修改原始网络权重。
2. **T 次随机前向**：每次 forward 随机丢弃不同通道，得到 T 组场景坐标 `(T, B, 3, Hf, Wf)`。
3. **重投影方差**：将 T 组坐标分别投影到 2D，对 T 求方差得到逐像素不确定性图。
4. **联合目标**：`conf = exp(-(α × repro_error + β × variance))`。两项均为像素尺度（误差为 px，方差为 px²），β 通常远小于 α（默认 α=0.1，β=0.01）。

## 可视化

通过将置信度图和采样点叠加到测试图像上，直观检查 SamplerNet 学到了什么。

```bash
python ace_sampler/visualize_sampler.py \
    /data/xwh/7Scenes/pgt_7scenes_chess \
    output/ace_models/7Scenes_pgt/pgt_7scenes_chess.pt \
    ace_sampler/04_evaluation/universal/20260323_223249_universal_sampler_all.pt \
    --out_dir ace_sampler/04_evaluation/vis_chess \
    --n_images 20 \
    --device cuda:0
```

每张图输出一个 `<stem>_side.png`，三栏并排：

| 栏 | 内容 |
|---|---|
| RGB | 原始图像 |
| Confidence | SamplerNet 置信度热图（红=高，蓝=低） |
| Sampled | 绿点 = top-k 高置信度采样（Part A），红点 = 随机补充（Part B） |

**参数说明：**

| 参数 | 默认值 | 说明 |
|---|---|---|
| `scene` | — | 场景根目录（须含 `test/` 或 `train/`） |
| `ace_head` | — | 预训练 ACE head `.pt` |
| `sampler_path` | — | 训练好的 SamplerNet `.pt` |
| `--encoder_path` | `ace_encoder_pretrained.pt` | FCN 编码器权重 |
| `--out_dir` | `ace_sampler/04_evaluation/vis` | 输出目录 |
| `--split` | `test` | 可视化哪个划分（`train` 或 `test`） |
| `--n_images` | 20 | 处理的图像数量 |
| `--samples_per_image` | 1024 | 每张图绘制的总点数 |
| `--sampler_ratio` | 0.7 | 绿点（top-k）与红点（随机）的比例 |
| `--image_resolution` | 480 | 输入图像高度 |
| `--device` | `cuda:0` | GPU 设备 |

## 模块结构

```
ace_sampler/
├── __init__.py            # 导出 SamplerNet、SamplerTrainer、fill_buffer_with_sampler
├── model.py               # SamplerNet 定义与存取
├── uncertainty.py         # MCDropoutRegressor — 特征层 MC 不确定性
├── options.py             # add_sampler_train_args、add_sampler_buffer_args、add_multi_sampler_train_args
├── trainer.py             # 第 1 阶段：SamplerTrainer（单场景，可选 MC Dropout）
├── multi_trainer.py       # 第 1 阶段：MultiSceneSamplerTrainer（通用多场景）
├── multi_dataset.py       # MultiSceneDataset 与 SceneEntry 数据类
├── buffer_sampler.py      # 第 2 阶段：fill_buffer_with_sampler
└── visualize_sampler.py   # 置信度图 + 采样点可视化
train_ace_sampler.py       # 第 1 阶段入口 — 单场景（项目根）
train_ace_sampler_multi.py # 第 1 阶段入口 — 多场景通用（项目根）
ace_eval.py                # train_ace.py 训练后评估共用的评估函数
```

## 说明

- 传入第 1 阶段的 ACE head 应已具备一定训练质量；即便只训少数几轮，也能提供有意义的重投影误差信号。
- 第 2 阶段中 `SamplerNet` 冻结，只影响采哪些像素，不参与 ACE 损失反传。
- `fill_buffer_with_sampler` 产出的缓冲区格式与 `ace_trainer.py` 一致：`features / target_px / gt_poses_inv / intrinsics / intrinsics_inv`。
- 未修改 `ace_network.py`；UncertaintyHead 为独立模块，与冻结的 ACE 回归器并行运行。
