# 系统架构：带 Welford 流式归一化的双边超体素提取

**作者**：高级 ML 系统架构师
**阶段**：3 — 架构设计
**日期**：2026-03-26

---

## 1. 架构概览

BSE 记忆提取系统由三个主要模块组成：

```
┌─────────────────────────────────────────────────────────────────┐
│                    run_memory_extraction.py                      │
│                        (主协调器)                                 │
└────────────┬────────────────────────────────────────────────────┘
             │
             ├──> FeatureExtractor (DINOv2 包装器)
             │    - 加载预训练 DINOv2-ViT-L/14
             │    - 提取稠密 patch 特征 [H/14, W/14, 1024]
             │
             ├──> BSEPooler (bse_pooling.py)
             │    - 粗体素哈希
             │    - Otsu 自适应阈值
             │    - 双边特征聚类
             │
             └──> WelfordNormalizer (welford_meter.py)
                  - 流式统计量累积 (FP64)
                  - 两阶段归一化
```

### 关键设计决策

1. **特征提取策略**：使用 DINOv2-ViT-L/14 最后一层 patch tokens（v1 中无多尺度融合）
2. **深度集成**：Depth Anything V2 仅用于反投影，不用于特征加权
3. **记忆格式**：与现有 `dino_lmc_base` trainer buffer schema 兼容
4. **模块化**：每个组件（FeatureExtractor、BSEPooler、WelfordNormalizer）可独立测试

---

## 2. 模块规范

### 2.1 FeatureExtractor

**目的**：包装 DINOv2 模型以实现高效批量特征提取。

**接口**：
```python
class FeatureExtractor:
    def __init__(self, dinov2_path: str, device: str, freeze: bool = True)
    def extract(self, images: Tensor) -> Tensor
        # 输入：[B, 3, H, W]，用 ImageNet 统计量归一化的 RGB
        # 输出：[B, H/14, W/14, 1024]，patch 特征
```

**实现说明**：
- 复用 `trainer_dinov2_lmc.py`（第 1-80 行）的 DINOv2 加载模式
- 使用 `torch.no_grad()` 上下文，因为 backbone 被冻结
- 仅返回 patch tokens（排除 CLS token）
- 输出形状：`[B, num_patches_h, num_patches_w, 1024]`，其中 `num_patches = H // 14`

**复用自 dino_lmc_base**：
- DINOv2 模型初始化模式
- ImageNet 归一化常数
- 设备管理

---

### 2.2 BSEPooler

**目的**：带 Otsu 自适应阈值的双边超体素提取。

**接口**：
```python
class BSEPooler:
    def __init__(self, voxel_size: float = 0.05, use_otsu: bool = True,
                 otsu_bins: int = 256, unimodal_threshold: float = 0.02)

    def pool(self, points: Tensor, features: Tensor, colors: Tensor,
             ray_dirs: Tensor) -> Dict[str, Tensor]
        # 输入：
        #   points: [N, 3] 世界坐标
        #   features: [N, C] DINOv2 特征 (fp16)
        #   colors: [N, 3] RGB 颜色
        #   ray_dirs: [N, 3] 单位射线方向
        # 输出：包含键 {points, features, colors, ray_dirs} 的字典
```

**核心算法**：
```python
def pool(self, points, features, colors, ray_dirs):
    # 步骤 1：粗体素哈希
    cluster_ids = self._voxel_hash(points)  # [N]

    # 步骤 2：计算簇均值特征（fp32 保证稳定性）
    F_mean = self._scatter_mean(features.float(), cluster_ids)  # [M, C]

    # 步骤 3：Otsu 自适应阈值
    sim = self._cosine_similarity(features.float(), F_mean[cluster_ids])  # [N]
    tau = self._otsu_threshold(sim) if self.use_otsu else 0.90

    # 步骤 4：二值分裂
    is_outlier = (sim < tau).long()
    sub_ids = cluster_ids * 2 + is_outlier

    # 步骤 5：细粒度池化
    return self._scatter_mean_all(points, features, colors, ray_dirs, sub_ids)
```

**实现说明**：
- 使用 `torch.unique(return_inverse=True)` 进行体素哈希
- 用 `index_add_` + 计数除法实现 `_scatter_mean`（无 torch_scatter 依赖）
- Otsu 实现：向量化直方图 + 类间方差最大化
- 单峰保护：如果 `std(sim) < unimodal_threshold` 则跳过分裂

---

### 2.3 WelfordNormalizer

**目的**：O(1) 内存的流式全局统计量计算。

**接口**：
```python
class WelfordNormalizer:
    def __init__(self)
    def update(self, points: Tensor) -> None
        # 从一块点累积统计量
    def finalize(self) -> Tuple[Tensor, float]
        # 返回 (mu_scene, sigma_scene)
    def normalize(self, points: Tensor) -> Tensor
        # 应用归一化：(points - mu) / sigma
```

**Welford 算法**（数值稳定的方差）：
```python
def update(self, points):
    for p in points:
        self.count += 1
        delta = p - self.mean
        self.mean += delta / self.count  # FP64
        self.M2 += delta * (p - self.mean)  # FP64

def finalize(self):
    mu = self.mean.float()  # FP64 -> FP32
    sigma = torch.sqrt(self.M2 / self.count).item()  # 标量
    return mu, sigma
```

**实现说明**：
- `mean` 和 `M2` 必须是 `torch.float64` 以避免灾难性相消
- `count` 是 Python int（<10^15 点无溢出风险）
- 输出 `mu` 和 `sigma` 转换为 FP32 以便存储

---

## 3. 主流程：run_memory_extraction.py

### 两阶段处理流程

```python
# 第一阶段：提取 + 池化 + 累积统计量
for batch in dataloader:
    features = feature_extractor.extract(batch['image'])
    points, ray_dirs, colors = unproject_batch(batch, features)
    pooled = bse_pooler.pool(points, features, colors, ray_dirs)
    welford.update(pooled['points'])
    torch.save(pooled, f"/dev/shm/chunk_{i}.pt")

# 第二阶段：归一化 + 组装
mu_scene, sigma_scene = welford.finalize()
for chunk in temp_chunks:
    pooled = torch.load(chunk)
    pooled['points'] = welford.normalize(pooled['points'])
    final_memory.append(pooled)
```

---

## 4. 输出格式与兼容性

**内存文件 Schema**：
```python
{
    "points": [N, 3] float32,      # 归一化坐标
    "ray_dirs": [N, 3] float32,    # 单位向量
    "features": [N, 1024] float16, # DINOv2 特征
    "colors": [N, 3] float32,      # RGB [0, 1]
    "mu": [3] float32,             # 场景均值
    "sigma": float32,              # 场景标准差
}
```

**与 dino_lmc_base 兼容**：输出格式与 `TrainerACEDINOv2LMC.load_memory_features()` 兼容。

---

## 5. 关键技术细节

**DINOv2 特征提取**：
- 模型：DINOv2-ViT-L/14（1024 维特征）
- 策略：单尺度，仅最后一层 patch tokens（v1 中无多尺度融合）
- 特征分辨率：518×518 输入 → [37, 37, 1024] 输出

**深度集成策略**：
- 深度来源：Depth Anything V2
- 用途：仅用于反投影（不用于特征加权）
- 理由：深度质量不稳定，用于加权可能放大误差

**性能优化**：
- 临时文件写入 `/dev/shm`（RAM 盘）
- Welford 累加器：O(1) 内存
- 数值稳定性：余弦相似度用 FP32，Welford 用 FP64

---

## 6. 文件组织

```
memory_extraction/
├── feature_extractor.py      # DINOv2 包装器
├── bse_pooling.py            # BSE 核心算法
├── welford_meter.py          # 流式归一化
├── run_memory_extraction.py  # 主协调器
└── extract_memory.sh         # Bash 启动脚本
```

**无新依赖**：所有模块仅使用 PyTorch 和标准库。

---

## 7. 实施路线图

**冲刺 1**：Welford 流式归一化（`welford_meter.py`）
**冲刺 2**：Otsu 自适应阈值（`bse_pooling.py`）
**冲刺 3**：特征提取与集成（`feature_extractor.py`，完整 `run_memory_extraction.py`）
**冲刺 4**：评估指标（`eval_boundary_metrics.py`）

---

## 8. 已回答的问题

✅ 特征提取：DINOv2-ViT-L/14 最后一层
✅ 深度使用：仅反投影，无加权
✅ 多尺度：v1 中不使用（保持简单）
✅ 兼容性：输出格式匹配 dino_lmc_base

**推迟到 v2**：多尺度特征融合、深度置信度加权、迭代 k-means（k>2）
