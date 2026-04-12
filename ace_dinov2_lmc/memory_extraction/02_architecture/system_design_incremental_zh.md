# 系统架构：BSE 增强的 Map-Anything 记忆提取

**作者**：高级 ML 系统架构师
**阶段**：3 — 架构设计（增量增强）
**日期**：2026-03-26

---

## 设计理念

这是对现有 Map-Anything 记忆提取流水线的**增量增强**，而非新方法。我们保留已验证的组件（视图选择、多尺度特征、真实深度），仅升级池化和归一化步骤。

**保持不变的部分：**
- 通过 `select_optimal_memory_indices` 进行视图选择（基于 FPS 的空间覆盖）
- 通过 `model.infer()` 进行多尺度特征提取（Map-Anything Transformer）
- 真实深度反投影（支持稠密和稀疏）
- 数据集加载器、可视化辅助工具

**改变的部分：**
- 池化：均匀体素池化 → BSE（双边超体素提取）+ Otsu 自适应阈值
- 归一化：无 → Welford 流式全局归一化（零均值、单位标准差）
- 输出：添加光线方向、全局统计量（μ_scene、σ_scene）

---

## 架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│              run_memory_extraction.py (主协调器)                 │
└────────────┬────────────────────────────────────────────────────┘
             │
             ├──> select_optimal_memory_indices (不变)
             │    - 基于 FPS 的视图选择
             │    - N_MEMORY 个视图（如 100 个）
             │
             ├──> model.infer() (不变)
             │    - Map-Anything Transformer
             │    - 多尺度中间特征
             │    - 真实深度支持
             │
             ├──> unproject_with_real_depth (不变)
             │    - 全光反投影
             │    - 稀疏深度处理
             │
             ├──> BSEPooler (新增 - 替换 voxel_pooling_optimized)
             │    - 粗体素哈希
             │    - Otsu 自适应阈值
             │    - 双边特征聚类
             │
             └──> WelfordNormalizer (新增)
                  - 流式统计量（FP64）
                  - 两阶段归一化
```

---

## 模块规范

### 1. BSEPooler (新增)

**目的**：用保留边界的双边聚类替换 `voxel_pooling_optimized`。

**位置**：`memory_extraction/bse_pooling.py`

**接口**：
```python
class BSEPooler:
    def __init__(self, voxel_size: float = 0.05, use_otsu: bool = True,
                 otsu_bins: int = 256, unimodal_threshold: float = 0.02)

    def pool(self, points: Tensor, features: Tensor, colors: Tensor,
             ray_dirs: Tensor) -> Dict[str, Tensor]
        # 输入：
        #   points: [N, 3] 世界坐标
        #   features: [N, C] 多尺度特征 (fp16)
        #   colors: [N, 3] RGB 颜色
        #   ray_dirs: [N, 3] 单位光线方向
        # 输出：包含键 {points, features, colors, ray_dirs} 的字典
```

**算法**：
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

### 2. WelfordNormalizer (新增)

**目的**：O(1) 内存的流式全局统计量计算。

**位置**：`memory_extraction/welford_meter.py`

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

### 3. 主流程（修改）

**文件**：`memory_extraction/run_memory_extraction.py`

**与原始 `map-anything/mapanything/tasks/run_memory_extraction.py` 的变化**：

| 组件 | 原始 | 增强后 |
|------|------|--------|
| 配置系统 | Hydra `@hydra.main` | argparse CLI |
| 视图选择 | `select_optimal_memory_indices` | **不变** |
| 模型推理 | `model.infer()` | **不变** |
| 特征提取 | 多尺度中间层 | **不变** |
| 深度处理 | 真实深度反投影 | **不变** |
| 池化 | `voxel_pooling_optimized` | `BSEPooler.pool()` |
| 光线方向 | 未存储 | **新增**：计算并存储 |
| 归一化 | 无 | **新增**：Welford 全局归一化 |
| 输出格式 | `{pooled_points, pooled_features}` | 扩展 schema（见下文）|

**两阶段处理流程**：
```python
# 第一阶段：提取 + 池化 + 累积统计量
welford = WelfordNormalizer()
bse_pooler = BSEPooler(voxel_size=0.05, use_otsu=True)

for batch in dataloader:
    # 不变：视图选择、model.infer()、特征提取
    memory_indices = select_optimal_memory_indices(train_dataset, N_MEMORY)
    predictions = model.infer(input_views, ...)
    features = process_multiscale_features(predictions)

    # 不变：用真实深度反投影
    points, ray_dirs, colors = unproject_with_real_depth(batch, features)

    # 改变：BSE 池化替代体素池化
    pooled = bse_pooler.pool(points, features, colors, ray_dirs)

    # 新增：累积全局统计量
    welford.update(pooled['points'])
    torch.save(pooled, f"/dev/shm/chunk_{i}.pt")

# 第二阶段：归一化 + 组装
mu_scene, sigma_scene = welford.finalize()
for chunk in temp_chunks:
    pooled = torch.load(chunk)
    pooled['points'] = welford.normalize(pooled['points'])
    final_memory.append(pooled)

# 用扩展 schema 保存
torch.save({
    'points': final_points,
    'ray_dirs': final_ray_dirs,
    'features': final_features,
    'colors': final_colors,
    'mu': mu_scene,
    'sigma': sigma_scene,
}, output_path)
```

---

## 输出格式

**扩展 schema**（向后兼容）：
```python
{
    "points":   [N, 3] float32,      # 归一化坐标（零均值、单位标准差）
    "ray_dirs": [N, 3] float32,      # 单位光线方向（新增）
    "features": [N, C] float16,      # 多尺度特征（C ~5120）
    "colors":   [N, 3] float32,      # RGB [0, 1]
    "mu":       [3] float32,         # 场景均值（新增）
    "sigma":    float32,             # 场景标准差（新增）
}
```

**向后兼容性**：期望 `pooled_points`/`pooled_features` 键的下游代码可以添加别名或使用简单适配器。

---

## 文件组织

```
memory_extraction/
├── __init__.py
├── bse_pooling.py            # 新增：BSE 算法
├── welford_meter.py          # 新增：流式归一化
├── run_memory_extraction.py  # 修改：增强流程
└── extract_memory.sh         # 修改：适配的 bash 脚本
```

**无新依赖**：所有模块仅使用 PyTorch 和标准库。

---

## 实施路线图

### Sprint 1：Welford 流式归一化
**文件**：`welford_meter.py`
- 实现 `WelfordNormalizer` 类
- 单元测试：在合成数据上验证数值稳定的方差
- 集成测试：在 7-Scenes Chess 上运行，验证 `mu` 和 `sigma` 合理

### Sprint 2：Otsu 自适应阈值
**文件**：`bse_pooling.py`（部分）
- 实现 `_otsu_threshold()` 函数
- 单元测试：在合成双峰/单峰分布上验证
- 消融：在真实相似度直方图上比较固定 τ=0.90 vs Otsu

### Sprint 3：BSE 池化
**文件**：`bse_pooling.py`（完整）
- 实现完整的 `BSEPooler` 类
- 在 `run_memory_extraction.py` 中替换 `voxel_pooling_optimized`
- 添加 CLI 标志：`--use_bse`、`--bse_tau`、`--use_otsu`

### Sprint 4：集成与评估
**文件**：`run_memory_extraction.py`（完整）
- 带 Welford 的两阶段处理
- 扩展输出 schema
- Bash 脚本适配
- 评估指标：FVR、SCD（见 proposal.md）

---

## 关键技术细节

**多尺度特征处理**：
- Map-Anything Transformer 输出 4-5 个中间层 + 1 个最终层
- 总通道数：~5120（取决于模型架构）
- 网格分辨率：37×37 或 74×74（取决于模型）
- 处理：网格对齐（默认）或 AnyUp 上采样（可选）

**真实深度处理**：
- 支持稠密和稀疏深度
- 稀疏深度：仅反投影有效像素（depth > 0）
- 无深度预测回退（仅使用真实深度）

**性能优化**：
- 临时文件写入 `/dev/shm`（RAM 盘）
- Welford 累加器：O(1) 内存
- 数值稳定性：余弦相似度用 FP32，Welford 用 FP64

---

## 从 map-anything 迁移

**要保留的依赖**：
- `mapanything.models.init_model`
- `mapanything.datasets.{SevenScenesWAI, Indoor6WAI}`
- `mapanything.tasks.ace.memory_selection.select_optimal_memory_indices`

**要替换的依赖**：
- Hydra 配置 → argparse
- 硬编码路径 → CLI 参数
- 可视化辅助工具 → 可选（保留用于调试）

**向后兼容性**：
- 输出 `.pt` 格式与 map-anything 匹配，用于下游 ACE 训练
- 可以加载现有的 map-anything 提取记忆

---

## 已回答的问题

✅ 特征提取：多尺度 Map-Anything Transformer（不变）
✅ 深度使用：真实深度用于反投影（不变）
✅ 视图选择：`select_optimal_memory_indices`（不变）
✅ 池化：从均匀体素升级到带 Otsu 的 BSE
✅ 归一化：添加 Welford 全局归一化
✅ 兼容性：扩展输出格式，向后兼容

**推迟到未来**：无深度数据集的学习式记忆提取（受 SceneTok 启发）
