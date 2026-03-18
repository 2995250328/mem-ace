# ACE Sampler

面向 ACE 训练的可学习缓冲区采样模块。将均匀随机采样替换为基于置信度引导的策略，该策略由重投影误差监督训练，并可选地结合 MC Dropout 不确定性估计进行增强。

## 概述

标准 ACE 通过对每张图像随机采样像素来填充训练缓冲区。本模块训练一个轻量级 `SamplerNet`，从编码器特征中预测逐像素置信度，再利用置信度将采样偏向当前 ACE 模型定位效果较好的点。

**两阶段工作流：**

- **第 1 阶段** — 以冻结 ACE 模型的重投影误差（以及可选的 MC Dropout 不确定性）作为监督信号，端到端训练 `SamplerNet`。
- **第 2 阶段** — 将训练好的 `SamplerNet` 插入 `ace_trainer.py` 的缓冲区填充流程，作为 `torch.multinomial` 的即插即用替代。

## 架构

### SamplerNet

轻量级深度可分离卷积置信度预测网络：

```
Input:  (B, 512, H/8, W/8)  — ACE 编码器特征
Block1: DW-Conv(512) → PW-Conv(128) → ReLU
Block2: DW-Conv(128) → PW-Conv(64)  → ReLU
Head:   Conv(64 → 1) → Sigmoid
Output: (B, 1, H/8, W/8)  — 置信度图，取值范围 [0, 1]
```

约 80K 参数，在特征分辨率（8× 下采样）下运行。

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

## 第 1 阶段：训练 SamplerNet

### 基础用法（仅重投影误差）

```bash
python train_ace_sampler.py \
    datasets/7scenes_chess \
    output/chess_baseline.pt \
    output/chess_sampler.pt \
    --device cuda:0
```

### 结合 MC Dropout 不确定性

```bash
python train_ace_sampler.py \
    datasets/7scenes_chess \
    output/chess_baseline.pt \
    output/chess_sampler_mc.pt \
    --device cuda:0 \
    --use_mc_dropout True \
    --mc_samples 10 \
    --mc_dropout_p 0.1 \
    --sampler_beta 0.01
```

## 第 2 阶段：使用引导采样训练 ACE

```bash
python train_ace.py \
    datasets/7scenes_chess \
    output/chess_guided.pt \
    --device cuda:0 \
    --sampler_path output/chess_sampler_mc.pt \
    --sampler_ratio 0.7
```

不指定 `--sampler_path` 时，`train_ace.py` 的行为与原版完全一致。

## 完整参数说明

### 第 1 阶段 — `train_ace_sampler.py`

| 参数 | 默认值 | 说明 |
|---|---|---|
| `scene` | — | 场景根目录（须包含 `train/`） |
| `ace_head` | — | 预训练 ACE head `.pt`（第 1 阶段冻结） |
| `sampler_output` | — | 训练好的 SamplerNet 输出路径 `.pt` |
| `--encoder_path` | `ace_encoder_pretrained.pt` | 预训练 FCN 编码器权重 |
| `--num_head_blocks` | 1 | ACE head 深度（须与加载的 head 匹配） |
| `--use_homogeneous` | `False` | 加载的 head 是否使用齐次坐标 |
| `--image_resolution` | 480 | 输入图像高度（像素） |
| `--use_aug` | `True` | 启用数据增强（旋转 + 缩放） |
| `--aug_rotation` | 15 | 最大平面内旋转角度（度） |
| `--aug_scale` | 1.5 | 最大缩放系数（范围：`[1/aug_scale, aug_scale]`） |
| `--sampler_epochs` | 5 | SamplerNet 训练轮数 |
| `--sampler_lr` | 1e-3 | SamplerNet 学习率（AdamW） |
| `--sampler_alpha` | 0.1 | 重投影误差权重：`exp(-α × error_px)`。α 越大，目标越尖锐（对大误差惩罚越重） |
| `--use_half` | `True` | 使用 FP16 混合精度 |
| `--device` | `cuda` | GPU 设备（如 `cuda:0`、`cuda:1`） |
| `--use_mc_dropout` | `False` | 启用 MC Dropout 不确定性估计 |
| `--mc_samples` | 10 | 每张图像的随机前向传播次数（T） |
| `--mc_dropout_p` | 0.1 | UncertaintyHead 的 Dropout 概率 |
| `--sampler_beta` | 0.01 | 不确定性权重：`exp(-β × mc_variance_px²)`。β 越大，不确定性惩罚越激进 |
| `--uncertainty_head_path` | `None` | 预训练 UncertaintyHead `.pt` 路径。若为 `None` 且 `--use_mc_dropout True`，则在第 1 阶段开始前通过 2 轮蒸馏从头训练该 head |

### 第 2 阶段 — `train_ace.py`（新增参数）

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--sampler_path` | `None` | 训练好的 SamplerNet `.pt`。若为 `None`，则使用原始随机采样（行为不变） |
| `--sampler_ratio` | 0.7 | `samples_per_image` 中从 top-k 置信度采样的比例（A 部分）。剩余 `1 - ratio` 随机采样（B 部分） |
| `--use_neighbors` | `False` | 将每个 top-k 选点扩展至特征分辨率下的 4 邻域 |

## 采样策略

对于每张图像，`fill_buffer_with_sampler` 采样 `samples_per_image` 个点：

- **A 部分**（`ratio × spi` 个点）— 由 SamplerNet 置信度排名 top-k 的像素，受 `image_mask` 约束
- **B 部分**（`(1-ratio) × spi` 个点）— 从 `image_mask` 中均匀随机采样（与原始 ACE 相同）

B 部分确保对当前模型尚未学习区域的覆盖（冷启动鲁棒性）。

## MC Dropout 工作流

当 `--use_mc_dropout True` 时：

1. **UncertaintyHead 预训练**（2 轮，在 SamplerNet 训练之前）：通过 MSE 损失将冻结 ACE Head 的预测蒸馏到 UncertaintyHead 中，确保 MC 方差具有实际意义而非随机噪声。
2. **逐图像不确定性**：对每张训练图像，UncertaintyHead 运行 T 次随机前向传播 → T 组场景坐标 → T 个重投影 2D 位置 → T 次结果的方差 = 不确定性图。
3. **联合目标**：`conf = exp(-(α × repro_error + β × variance))`。两项均以像素为单位（误差单位 px，方差单位 px²），因此 β 应远小于 α（默认：α=0.1，β=0.01）。

## 模块结构

```
ace_sampler/
├── __init__.py          # 导出 SamplerNet、SamplerTrainer、fill_buffer_with_sampler
├── model.py             # SamplerNet 定义 + 保存/加载
├── uncertainty.py       # 带 MC Dropout 的 UncertaintyHead + 保存/加载
├── options.py           # add_sampler_train_args、add_sampler_buffer_args
├── trainer.py           # 第 1 阶段：SamplerTrainer（含可选 MC Dropout）
└── buffer_sampler.py    # 第 2 阶段：fill_buffer_with_sampler
train_ace_sampler.py     # 第 1 阶段入口脚本（项目根目录）
```

## 注意事项

- 传入第 1 阶段的 ACE head 应已经过合理训练。即使只训练了几轮的 head 也能提供有意义的重投影误差信号。
- `SamplerNet` 在第 2 阶段保持冻结——它只影响采样哪些像素，不参与 ACE 损失计算。
- `fill_buffer_with_sampler` 生成的缓冲区格式与 `ace_trainer.py` 完全一致：`features / target_px / gt_poses_inv / intrinsics / intrinsics_inv`。
- `ace_network.py` 未作任何修改。UncertaintyHead 是独立模块，与冻结的 ACE 回归器并行运行。
