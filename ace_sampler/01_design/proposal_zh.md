# 标题：基于置信度引导的缓冲区采样用于场景坐标回归

## 1. 问题描述

**任务**：给定一组带有已知位姿 $\{T_i\}$ 的训练图像 $\{I_i\}$，以及一个冻结的
ACE 编码器 $f_\theta$，学习一个采样策略 $\pi: \mathbb{R}^{C \times H_f \times W_f} \to [0,1]^{H_f \times W_f}$，
用于选择哪些像素级特征存入训练缓冲区。

**输入**：
- 编码器特征 $F_i = f_\theta(I_i) \in \mathbb{R}^{512 \times H_f \times W_f}$
- 真值位姿 $T_i \in SE(3)$，相机内参 $K_i \in \mathbb{R}^{3 \times 3}$
- 预训练 ACE 头网络 $g_\phi$（在第一阶段冻结）

**目标**：最大化在引导缓冲区上训练的 ACE 头网络的定位精度，
使其优于在相同大小的均匀采样缓冲区上训练的头网络。

**置信度目标**：
$$c_{ij} = \exp\!\left(-\alpha \cdot e_{ij} - \beta \cdot \sigma^2_{ij}\right)$$
其中 $e_{ij}$ 是冻结头网络下图像 $i$ 中像素 $j$ 的重投影误差（像素），
$\sigma^2_{ij}$ 是 MC Dropout 重投影方差（像素²）。$\beta = 0$ 时禁用不确定性估计。

## 2. 文献综述与研究动机

**ACE（Brachmann 等，CVPR 2023）**：在 GPU 缓冲区上训练场景特定头网络，
缓冲区由随机采样的编码器特征构成。随机采样是我们改进的基线方法。

**主动学习 / 课程学习**：Bengio 等（2009）表明，按难度排序训练样本可以改善收敛性。
我们的方法是其空间类比：优先选择当前模型不确定或预测错误的像素。

**MC Dropout（Gal & Ghahramani，ICML 2016）**：通过随机前向传播实现不确定性估计。
我们将其作为辅助信号，用于识别结构上存在歧义的区域。

**研究空白**：ACE 的缓冲区填充是唯一仍保持纯随机的组件。
用可学习策略替换它是一个自然的扩展方向，目前尚未被探索。

## 3. 设计空间与所选架构

**方案 A — 在线难例挖掘**：在训练过程中按损失对缓冲区样本重新加权。
需要修改训练循环，且会引入不稳定性。

**方案 B — 离线置信度图（已选）**：训练一个独立的轻量网络，
从编码器特征预测置信度，作为预处理步骤插入缓冲区填充流程。
关注点分离清晰；无需修改 ACE 训练循环。

**方案 C — 基于注意力的采样**：对特征图使用自注意力机制。
在缓冲区填充的逐图推理中开销过大。

**已选方案**：方案 B。SamplerNet 是一个深度可分离卷积网络（约 80K 参数），
从编码器特征预测置信度图，通过 MSE 损失对置信度目标进行训练。

### SamplerNet 架构
```
Input:  (B, 512, Hf, Wf)
Block1: DWConv(512) → PWConv(128) → ReLU
Block2: DWConv(128) → PWConv(64)  → ReLU
Head:   Conv1x1(64 → 1) → Sigmoid
Output: (B, 1, Hf, Wf)  ∈ [0, 1]
```

### UncertaintyHead（可选）
```
Input:  (B, 512, Hf, Wf)
        Conv1x1(512 → 256) → Dropout2d(p) → ReLU → Conv1x1(256 → 3)
Output: (B, 3, Hf, Wf)  — 场景坐标预测
```
运行 T=10 次随机前向传播 → 重投影二维位置的方差 = 不确定性图。
通过对冻结 ACE 头网络进行 2 轮蒸馏预训练。

### 第二阶段采样策略
对每张图像，采样 `samples_per_image` 个点：
- **A 部分**（`ratio × spi`）：按 SamplerNet 置信度取 top-k 像素（在有效区域内掩码）
- **B 部分**（`(1-ratio) × spi`）：均匀随机采样（冷启动鲁棒性）

## 4. 评估与验证方案

**数据集**：7-Scenes（chess、fire、heads、office、pumpkin、redkitchen、stairs），Cambridge Landmarks

**指标**：中位平移误差（cm），中位旋转误差（°），5cm/5° 以内帧比例

**基线**：
1. ACE 基线（随机采样，相同缓冲区大小）
2. ACE + SamplerNet（仅重投影误差，β=0）
3. ACE + SamplerNet + MC Dropout（完整方法）

**消融实验**：
- `sampler_ratio` ∈ {0.5, 0.7, 0.9}
- `sampler_alpha` ∈ {0.05, 0.1, 0.2}
- `mc_samples` ∈ {5, 10, 20}

## 5. 预期失效模式与工程风险

**循环依赖**：SamplerNet 在固定头网络上训练，但第二阶段训练的是新头网络。
若新头网络偏差较大，置信度图可能已过时。
*缓解措施*：B 部分随机采样（ratio < 1.0）确保覆盖率。

**冷启动**：第二阶段早期，新头网络在所有位置均有较高的均匀误差。
第一阶段的置信度图可能无法反映新头网络的误差分布。
*缓解措施*：第一阶段头网络应经过充分训练（≥ 若干轮次）。

**确认偏差**：对高置信度区域的采样可能强化已有良好预测，
同时忽略难例（如无纹理墙面、重复纹理区域）。
*缓解措施*：B 部分随机采样；如有需要，调低 `sampler_ratio`。

## 6. 复用计划

| 模块 | 来源 |
|---|---|
| 冻结 ACE 回归器 | `ace_network.py:Regressor`（不变） |
| 训练数据加载器 | `dataset_origin.py:CamLocDataset`（不变） |
| 像素网格 | `ace_util.py:get_pixel_grid`（不变） |
| 第二阶段缓冲区填充 | 扩展 `ace_trainer.py:_fill_buffer` 逻辑 |
| SamplerNet、UncertaintyHead、SamplerTrainer | `ace_sampler/` 中的新代码 |
