# 标题：ACE-FCN-LMC：基于 FCN 编码器与几何潜在记忆压缩的加速坐标编码

## 1. 问题定义

**任务**：6DoF 视觉相机重定位——给定查询图像 $I \in \mathbb{R}^{H \times W \times 1}$（灰度图），预测相机相对于已知场景的位姿 $\mathbf{T} \in SE(3)$。

**方法**：场景坐标回归。网络 $f_\theta$ 预测稠密 3D 场景坐标图 $\mathbf{S} \in \mathbb{R}^{(H/8) \times (W/8) \times 3}$，然后 DSAC* RANSAC 求解 PnP 问题以恢复 $\mathbf{T}$。

**目标函数**：
$$\mathcal{L} = \mathcal{L}_\text{repro}(\mathbf{S}, \mathbf{T}^*, \mathbf{K}) + \lambda \cdot \mathcal{L}_\text{lmc}(\mathbf{S}, \mathbf{M})$$

其中 $\mathbf{T}^*$ 为真实位姿，$\mathbf{K}$ 为相机内参矩阵，$\mathbf{M}$ 为外部 3D 点云记忆，$\mathcal{L}_\text{lmc}$ 为记忆引导的坐标对齐损失。

**输入**：灰度图（1 通道）、相机内参、可选的 pooled 3D 记忆 $\mathbf{M}$

**输出**：训练好的场景专属头网络（~4MB）、每帧 6DoF 位姿估计

## 2. 文献综述与动机

### 代表性方法

1. **ACE（Brachmann et al., CVPR 2023）**：两阶段训练——在 ScanNet 上预训练场景无关 FCN encoder，然后通过 buffer 坐标回归训练场景专属头。训练快（~5 分钟/场景）、模型紧凑（~4MB），但受限于单阶段 buffer 训练。

2. **ACE-G（Brachmann et al., CVPR 2024）**：用全局记忆模块扩展 ACE。迭代 S1+S2 训练：S1 将头网络预测对齐到记忆提供的 3D 坐标；S2 通过重投影损失精化。在 7-Scenes 和 Indoor6 上达到最先进水平。

3. **DINOv2-LMC（本项目，前期工作）**：用 DINOv2 ViT-L/14 替换 FCN encoder。精度更高但推理更慢（~10-15 FPS vs ~30 FPS），显存需求更高。

### 研究空白

ACE-G 精度强，但某些配置需要 DINOv2 backbone。FCN encoder 更快更轻，但尚未以干净、可复用的方式与 GeoLMC 迭代训练框架结合。空白：**能否通过将 LMC 框架应用于 FCN encoder，以 ACE-vanilla 级别的速度获得 ACE-G 级别的精度？**

### 为何非平凡

- FCN encoder 输出基于灰度图的 512 维特征（8x 下采样），而 DINOv2 输出基于 RGB 的 1024 维特征（14x 下采样）。LMC 流水线必须适配这些差异。
- 记忆对齐损失（S1）必须在低维特征下工作。
- 迭代训练调度（预热步数、学习率缩放）可能需要针对 FCN encoder 重新调整。

## 3. 设计空间与选定架构

### 备选方案

| 方案 | 描述 | 权衡 |
|---|---|---|
| A：完全重写 | 从头实现 FCN-LMC | 干净但重复约 2000 行已验证代码 |
| B：薄子类（已选） | 子类化 DINOv2 训练器，仅覆盖 `_create_regressor()` | 代码最少，继承所有已验证逻辑 |
| C：基于配置的切换 | 在现有训练器中添加 `--encoder_type fcn/dinov2` 标志 | CLI 更简单但在共享代码中增加分支复杂度 |

**已选：方案 B** — 薄子类模式。理由：
- DINOv2 LMC 训练器已在多个场景中验证
- 只有 encoder 实例化不同；所有 buffer 管理、损失计算和迭代调度逻辑完全相同
- 最大程度降低在已验证训练逻辑中引入 bug 的风险

### 核心架构

```
输入（灰度图，H×W×1）
    │
    ▼
ACEEncoder（FCN，冻结）
    │  512 维特征，H/8 × W/8
    ▼
RegressorACE（Head，场景专属）
    │  3D 场景坐标，H/8 × W/8 × 3
    ▼
DSAC* RANSAC → 6DoF 位姿 T ∈ SE(3)
```

**LMC 两阶段训练**：
- **Stage 1（S1）**：通过坐标匹配损失将头网络预测对齐到 pooled 记忆 $\mathbf{M}$。使用在线 encoder 前向或预填充 buffer。
- **Stage 2（S2）**：通过重投影损失 $\mathcal{L}_\text{repro}$ 在完整训练集上精化。基于 buffer，迭代快速。

### 关键模块

| 模块 | 文件 | 职责 |
|---|---|---|
| ACEEncoder | `ace_network_ace.py` | 包装 FCN encoder，暴露与 DINOv2 兼容的接口 |
| RegressorACE | `ace_network_ace.py` | 坐标回归头网络（512 维输入） |
| TrainerACEFCN | `trainer_ace_fcn.py` | Vanilla 训练（继承 TrainerACEDINOv2） |
| TrainerACEFCNLMC | `trainer_ace_fcn.py` | LMC 训练（继承 TrainerACEDINOv2LMC） |
| options_ace_lmc | `options_ace_lmc.py` | CLI 参数（encoder_path，无 patch-size 约束） |
| train_ace_lmc | `train_ace_lmc.py` | 训练入口（vanilla + LMC 模式） |
| test_ace_lmc | `test_ace_lmc.py` | 评估脚本 |

## 4. 评估与验证计划

### 数据集
- **7-Scenes**（室内，小规模）：chess、fire、heads、office、pumpkin、redkitchen、stairs
- **Indoor6**（室内，中等规模）：scene1–scene6
- **Cambridge Landmarks**（室外，大规模）：KingsCollege、OldHospital、ShopFacade、StMarysChurch

### 指标
- `pct5`：5cm/5deg 以内的帧百分比（主要指标）
- `pct25_5`：25cm/5deg 以内的帧百分比
- `median_rErr`：中位数旋转误差（度）
- `median_tErr`：中位数平移误差（cm）
- `avg_time`：每帧平均推理时间（ms）

### 基线
- ACE vanilla（单阶段，无记忆）
- ACE-G（官方，带记忆）
- DINOv2-LMC（本项目，前期工作）

### 消融研究
- Vanilla vs. LMC（记忆开/关）
- LMC 迭代次数（4、8、16、28）
- S1 损失模式：`full_map` vs. `sample_per_image`
- S1 buffer 模式：在线 encoder vs. 预填充 buffer
- 多轮 vanilla 迭代（1、2、3）

## 5. 预期失败模式与工程风险

| 风险 | 可能性 | 缓解措施 |
|---|---|---|
| FCN 512 维特征不足以进行记忆对齐 | 中 | 使用更大的头（num_head_blocks=4+） |
| S1 学习率调度未针对 FCN 调整 | 中 | 从 DINOv2 默认值开始，调整 warmup_steps |
| 记忆格式不匹配（pooled_features dim=512 vs 1024） | 低 | 记忆仅包含 3D 点；S1 损失中不使用特征 |
| 16GB GPU 上大 buffer OOM | 低 | 使用 `--buffer_on_cpu True` |
| 灰度→RGB 转换伪影 | 低 | ACEEncoder 内部处理 RGB→灰度转换 |

## 6. 复用计划

完整映射见 `00_ideas/reuse_map.md`。摘要：

- **不变复用**：`ace_network.py`（Encoder）、`ace_network_dinov2.py`（Head）、`trainer_dinov2.py`、`trainer_dinov2_lmc.py`、`dataset.py`、`ace_loss.py`、`result_manager.py`、`utils_lmc.py`、`dsacstar/`
- **新增（薄包装器）**：`ace_network_ace.py`、`options_ace_lmc.py`、`trainer_ace_fcn.py`、`train_ace_lmc.py`、`test_ace_lmc.py`
- **新增代码总量**：~1500 行（vs 从头编写的 ~6000 行）
