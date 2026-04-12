# 实现说明：BSE Memory Extraction

## 状态：已完成

BSE Memory Extraction 管道已完整实现，包含多策略射线方向处理。

---

## 架构概览

```
run_memory_extraction.py
├── main()
│   ├── parse_args()                 # CLI 参数解析
│   ├── MapAnythingExtractor         # MapAnything 模型（多尺度特征）
│   ├── select_memory_views()        # 记忆视图选择
│   ├── two_pass_processing()        # 核心提取循环
│   │   ├── Pass 1: 提取 + 池化 + 累积
│   │   └── Pass 2: 归一化 + 保存
│   └── save_memory()                # 输出 .pt 文件
├── BSEPooler (bse_pooling.py)
│   ├── _voxel_hash()                # 质数编码
│   ├── _compute_cluster_means()     # 粗糙特征
│   ├── _otsu_threshold()            # 自适应阈值
│   ├── _binary_split()              # 离群点分离
│   └── pool()                       # 主池化
└── WelfordNormalizer (welford_meter.py)
    ├── update()                     # 流式统计
    └── finalize()                   # 返回均值/标准差
```

---

## 核心函数

### 1. `MapAnythingExtractor.extract()`

**目的**：使用 MapAnything 模型提取多尺度特征。

**实现**：
```python
class MapAnythingExtractor:
    TARGET_INTERM_LAYERS = [0, 6, 12, 18]  # DPT 风格的层

    def extract(self, images, depths, poses, intrinsics):
        # 准备输入视图（mapanything 格式）
        input_views = [...]

        # 运行推理并保存中间特征
        self.model.infer(input_views, save_filename=temp_file)

        # 加载保存的特征
        saved_data = torch.load(temp_file)
        raw_interm = saved_data["intermediate"]  # 24 层
        raw_final = saved_data["final"]

        # 提取目标层 + 最终层
        result = {}
        for i in range(n_views):
            feat_list = []
            for layer_idx in self.TARGET_INTERM_LAYERS:
                feat_list.append(raw_interm[layer_idx]["features"][i])
            feat_list.append(raw_final["features"][i])
            result[i] = {'features': feat_list, 'grid_H': H, 'grid_W': W}

        return result
```

### 2. `process_multiscale_features_to_grid(feat_list)`

**目的**：将多尺度特征对齐到网格分辨率（DPT 风格）。

**实现**：
```python
def process_multiscale_features_to_grid(feat_list, apply_l2_norm=False):
    # 1. 将所有特征重塑为 (B, C, H, W) 格式
    # 2. 找到所有层中的最大分辨率
    # 3. 将较小特征上采样到最大分辨率
    # 4. 沿通道维度连接
    # 5. 展平为 (grid_H * grid_W, C_total)

    return aligned_features, max_H, max_W
    # 输出: [N, ~5000] 特征（5 层 × 1024）
```

### 3. `two_pass_processing(memory_views, features_dict, ...)`

**目的**：分块内存的两遍处理。

**Pass 1**（提取 + 池化 + 累积）：
```python
for view_idx, view_data in enumerate(memory_views):
    # 获取此视图的多尺度特征
    view_info = features_dict[view_idx]
    feat_list = view_info['features']
    grid_H, grid_W = view_info['grid_H'], view_info['grid_W']

    # 处理特征：对齐到网格并连接
    features_flat, _, _ = process_multiscale_features_to_grid(feat_list)

    # 使用网格特征反投影（在网格中心采样深度）
    points, ray_dirs, colors, features_flat, camera_centers = unproject_with_grid_features(
        view_data, features_flat, grid_H, grid_W, depth_valid_range
    )

    # 可选：SOR 过滤
    if enable_sor:
        inlier_mask = sor_filter(points)
        points = points[inlier_mask]
        # ... 过滤其他数组

    # BSE 池化
    pooled = bse_pooler.pool(points, features_flat, colors, ray_dirs, camera_centers)

    # 累积统计
    welford.update(pooled['points'])

    # 收集视图级别的相机信息
    view_camera_centers.append(camera_center)
    view_camera_rotations.append(R)
    view_camera_intrinsics.append(K)
    # ... 计算 Plücker 主射线
```

**Pass 2**（归一化 + 保存）：
```python
mu, sigma = welford.finalize()

for chunk_path in chunk_paths:
    pooled = torch.load(chunk_path)
    pooled['points'] = (pooled['points'] - mu) / sigma
    # ... 连接到最终记忆

# 组装视图级别的相机信息
view_info = {
    'camera_centers': torch.stack(view_camera_centers, dim=0),
    'camera_rotations': torch.stack(view_camera_rotations, dim=0),
    'camera_intrinsics': torch.stack(view_camera_intrinsics, dim=0),
    'plucker_main_rays': torch.stack(view_plucker_main_rays, dim=0)
}
```

### 4. `BSEPooler.pool(points, features, colors, ray_dirs, camera_centers)`

**目的**：边界保持双边聚类。

**算法**：
1. 粗糙体素哈希 → 体素索引
2. 计算聚类均值特征
3. 特征相似度的 Otsu 阈值
4. 二分分割：将离群点从主聚类分离
5. 使用多种射线策略的细粒度池化

**输出**：
```python
{
    'points': P_bse,              # [M, 3] float32
    'features': F_bse,            # [M, C] float16 (~5000 维)
    'colors': C_bse,              # [M, 3] float32
    'ray_dirs': D_pooled,         # [M, 3] 基于策略
    'ray_dirs_mean': D_mean,
    'ray_dirs_dominant': D_dominant,
    'ray_dirs_first': D_first,
    'plucker_rays': plucker,      # [M, 6]
    'camera_centers': centers,    # [M, 3]
    'cluster_sizes': sizes,       # [M] int64
}
```
    pooled = bse_pooler.pool(
        points, features, colors, ray_dirs, camera_centers
    )

    # 累积统计
    welford.update(pooled['points'])

    # 保存分块
    torch.save(pooled, f"/dev/shm/chunk_{idx}.pt")
    chunk_paths.append(f"/dev/shm/chunk_{idx}.pt")
```

**Pass 2**（归一化 + 保存）：
```python
mu, sigma = welford.finalize()

final_points = []
final_ray_dirs = []
final_features = []
final_colors = []
# ... 其他数组

for chunk_path in chunk_paths:
    pooled = torch.load(chunk_path)

    # 归一化点
    points_norm = (pooled['points'] - mu) / sigma

    final_points.append(points_norm)
    final_ray_dirs.append(pooled['ray_dirs'])
    final_features.append(pooled['features'])
    # ...

    torch.save(final_memory, args.output_path)
```

### 3. `BSEPooler.pool(points, features, colors, ray_dirs, camera_centers)`

**目的**：边界保持双边聚类。

**算法**：
1. 粗糙体素哈希 → 体素索引
2. 计算聚类均值特征
3. 特征相似度的 Otsu 阈值
4. 二分分割：将离群点从主聚类分离
5. 使用多种射线策略的细粒度池化

**输出**：
```python
{
    'points': P_bse,              # [M, 3] float32
    'features': F_bse,            # [M, C] float16
    'colors': C_bse,              # [M, 3] float32
    'ray_dirs': D_pooled,         # [M, 3] 基于策略
    'ray_dirs_mean': D_mean,
    'ray_dirs_dominant': D_dominant,
    'ray_dirs_first': D_first,
    'plucker_rays': plucker,      # [M, 6]
    'camera_centers': centers,    # [M, 3]
    'cluster_sizes': sizes,       # [M] int64
}
```

---

## 射线方向处理

### 问题

观测同一物体的多视角射线方向差异很大。简单平均会丢失几何信息。

### 解决方案：多策略

| 策略 | 公式 | 用例 |
|------|------|------|
| `mean` | `normalize(mean(rays))` | 基线池化 |
| `dominant` | `argmax(similarity)` | 最具代表性 |
| `first` | `rays[0]` | 保留原始 |
| `all` | 保留全部 | 下游学习 |

### 普吕克编码

```python
def compute_plucker_rays(points, ray_dirs, camera_centers):
    moment = torch.cross(camera_centers, ray_dirs, dim=1)
    plucker = torch.cat([ray_dirs, moment], dim=1)  # [N, 6]
    return plucker
```

输出包含所有策略，用于下游实验。

---

## 输出 Schema (v1.2)

```python
{
    # Schema
    'schema_version': '1.2',

    # 点级别数据
    'points': P_norm,           # [N, 3] float32, 零均值单位方差
    'features': F_bse,          # [N, C] float16, 多尺度 DINOv2 特征（~5000 维）
    'colors': C_bse,            # [N, 3] float32, RGB [0,1]
    'cluster_sizes': sizes,     # [N] 每聚类点数

    # 射线表示（点级别，池化后）
    'ray_dirs': D_strategy,     # [N, 3] 由 --ray_pool_strategy 选择
    'ray_dirs_mean': D_mean,
    'ray_dirs_dominant': D_dom,
    'ray_dirs_first': D_first,
    'plucker_rays': plucker,    # [N, 6] (方向, 矩)

    # 视图级别相机信息（M = 记忆视图数量）
    'view_camera_centers': centers,    # [M, 3] 相机位置
    'view_camera_rotations': rots,     # [M, 3, 3] 旋转矩阵
    'view_camera_intrinsics': Ks,      # [M, 3, 3] 内参矩阵
    'view_plucker_main_rays': plucker, # [M, 6] (方向, 矩)

    # 归一化
    'mu': mu_scene,             # [3]
    'sigma': sigma_scene,       # 标量
}
```

---

## 测试

### 单元测试

```python
from ace_dinov2_lmc.memory_extraction import BSEPooler, WelfordNormalizer

# 测试 BSEPooler
pooler = BSEPooler(voxel_size=0.05, use_otsu=True, ray_pool_strategy='mean')
pooled = pooler.pool(points, features, colors, ray_dirs, camera_centers)

# 测试 WelfordNormalizer
welford = WelfordNormalizer()
welford.update(points_chunk)
mu, sigma = welford.finalize()
```

### 烟雾测试

```bash
conda activate mapanything

python -m ace_dinov2_lmc.memory_extraction.run_memory_extraction \
    /mnt/storage/xwh/7Scenes/pgt_7scenes_chess \
    output/test_memory.pt \
    --n_memory 5 \
    --device cuda:0
```

---

## 文件结构

```
memory_extraction/
├── bse_pooling.py              # BSE + 射线策略
├── welford_meter.py            # 流式归一化
├── run_memory_extraction.py    # 主脚本
├── extract_memory.sh           # Shell 启动器
├── README.md                   # 使用指南
├── .research_state.md          # 研究工作流状态
├── 00_ideas/
│   └── reuse_map.md            # 可复用代码索引
├── 01_design/
│   ├── proposal.md             # 原始提案
│   └── proposal_zh.md
├── 02_architecture/
│   ├── system_design_incremental.md  # 最终设计
│   └── system_design_incremental_zh.md
└── 03_implementation/
    ├── IMPLEMENTATION_NOTES.md         # 本文件（英文）
    └── IMPLEMENTATION_NOTES_zh.md      # 本文件（中文）
```

---

## 关键实现细节

### 适配器模式

与 map-anything 解耦：

```python
class FeatureExtractorAdapter:
    def extract(self, images):
        # 独立 DINOv2，不依赖 map-anything
        with torch.no_grad():
            return self.dinov2(images)

class DatasetAdapter:
    def __init__(self, path, dataset_type):
        if dataset_type == "7scenes":
            self.dataset = SevenScenesDataset(path, split='train')
        # ...
```

### GPU 内存清理

```python
try:
    # 处理视图
    for idx in memory_indices:
        # ... 提取代码 ...
finally:
    # 清理临时文件
    for chunk_path in chunk_paths:
        if os.path.exists(chunk_path):
            os.remove(chunk_path)
```

### 输入验证

```python
assert depth.dim() == 2, f"深度必须是 2D，当前形状 {depth.shape}"
assert pose.shape == (4, 4), f"位姿必须是 4x4，当前 {pose.shape}"
assert features.shape[0] == points.shape[0], "特征/点数量不匹配"
```

---

## 后续工作

### 阶段 6：评估

- 在 7-Scenes 和 Indoor6 数据集上运行
- 比较射线策略对下游 ACE 精度的影响
- 消融实验：BSE vs 普通 体素池化
- 指标：压缩率、边界保持、定位精度

### 潜在改进

1. **学习型射线池化**：基于注意力的射线加权
2. **迭代聚类**：k 路而非二分
3. **多尺度 BSE**：层次池化
4. **时间一致性**：用于视频序列
