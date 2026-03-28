# BSE Memory Extraction 流程指南

本文档全面解释了 ACE DINOv2 LMC 的 BSE（双边超体素提取）Memory Extraction 流程。

## 概述

该流程从多视角 RGB-D 数据中提取压缩的场景表示，包括：

1. **视角选择**：从训练数据中选择最优的记忆视图
2. **特征提取**：从 RGB 图像中提取 DINOv2 特征
3. **反投影**：将 2D 像素 + 深度转换为带射线方向的 3D 点
4. **BSE 池化**：边界保持的双边聚类以压缩点云
5. **归一化**：使用 Welford 算法进行全局场景归一化

输出是一个 `.pt` 文件，包含：
- 带特征和颜色的压缩 3D 点
- 多种射线方向表示
- 视图级别的相机信息

---

## 流程架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              输入数据集                                     │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ 训练数据: RGB 图像 + 深度图 + 位姿 + 内参                             │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           步骤 1: 视角选择                                  │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ 输入: 所有训练帧                                                     │   │
│  │ 方法: 均匀采样或 FPS（最远点采样）                                    │   │
│  │ 输出: M 个记忆视图（默认：100）                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           步骤 2: 特征提取                                  │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ 输入: M 张 RGB 图像                                                  │   │
│  │ 模型: DINOv2 ViT-L/14（预训练）                                      │   │
│  │ 输出: M 个特征图 [1024, H/14, W/14]                                  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            步骤 3: 反投影                                    │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ 对每个记忆视图:                                                       │   │
│  │   1. 获取有效深度像素（在范围内）                                     │   │
│  │   2. 反投影: P_cam = Z * K^-1 * [u, v, 1]^T                          │   │
│  │   3. 世界坐标变换: P_world = R * P_cam + t                            │   │
│  │   4. 射线方向: D = normalize(P_world - camera_center)                │   │
│  │   5. 在有效像素处提取特征和颜色                                       │   │
│  │                                                                      │   │
│  │ 每个视图的输出:                                                       │   │
│  │   - points: [N_i, 3] 3D 坐标                                         │   │
│  │   - features: [N_i, 1024] DINOv2 特征                                │   │
│  │   - colors: [N_i, 3] RGB 颜色                                        │   │
│  │   - ray_dirs: [N_i, 3] 单位射线方向                                  │   │
│  │   - camera_centers: [N_i, 3] 相机中心（重复）                        │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           步骤 4: BSE 池化                                  │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ 对每个视图应用，然后跨视图合并:                                       │   │
│  │                                                                      │   │
│  │ 4.1 粗糙体素哈希                                                     │   │
│  │     - 将点量化到体素网格: voxel_id = floor(p / voxel_size)           │   │
│  │     - 使用质数编码以确保唯一性                                       │   │
│  │     - 按 voxel_id 对点分组                                           │   │
│  │                                                                      │   │
│  │ 4.2 计算聚类均值特征                                                 │   │
│  │     - 对每个体素，计算均值特征向量                                   │   │
│  │     - 这代表聚类的"原型"                                             │   │
│  │                                                                      │   │
│  │ 4.3 Otsu 自适应阈值                                                 │   │
│  │     - 计算每个点与聚类均值之间的余弦相似度                           │   │
│  │     - 构建相似度直方图                                               │   │
│  │     - 应用 Otsu 方法找到最优阈值                                     │   │
│  │     - 如果聚类是单峰的（std < threshold），跳过分割                  │   │
│  │                                                                      │   │
│  │ 4.4 二分分割                                                         │   │
│  │     - 主聚类: 相似度 >= threshold 的点                               │   │
│  │     - 离群聚类: 相似度 < threshold 的点（边界）                       │   │
│  │     - 这将边界点单独保留                                             │   │
│  │                                                                      │   │
│  │ 4.5 细粒度池化                                                       │   │
│  │     - 对每个聚类（主 + 离群）:                                        │   │
│  │       * points: 平均位置                                            │   │
│  │       * features: 平均特征                                          │   │
│  │       * colors: 平均颜色                                             │   │
│  │       * ray_dirs: 多种策略（见下文）                                 │   │
│  │                                                                      │   │
│  │ 每个视图的输出:                                                       │   │
│  │   - M_j 个池化聚类（M_j << N_i）                                     │   │
│  │   - 每个聚类有: point, feature, color, ray representations            │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        步骤 5: 两遍处理                                     │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ 第一遍: 提取 + 池化 + 累积                                            │   │
│  │   - 通过步骤 2-4 处理每个记忆视图                                    │   │
│  │   - 从所有视图收集池化聚类                                           │   │
│  │   - 累积全局统计（Welford: 均值、方差）                              │   │
│  │   - 将临时分块保存到 /dev/shm                                        │   │
│  │   - 收集视图级别的相机信息（位姿、内参、Plücker）                    │   │
│  │                                                                      │   │
│  │ 第二遍: 归一化 + 组装                                                  │   │
│  │   - 完成 Welford 统计                                                │   │
│  │   - 归一化所有点: (p - mu) / sigma                                   │   │
│  │   - 将所有聚类连接成最终记忆                                         │   │
│  │   - 组装视图级别的相机信息                                           │   │
│  │                                                                      │   │
│  │ 输出:                                                                │   │
│  │   - result: 包含所有池化张量的字典                                   │   │
│  │   - view_info: 包含视图级别相机信息的字典                            │   │
│  │   - mu, sigma: 场景归一化参数                                        │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            输出（.pt 文件）                                  │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ Schema 版本: 1.2                                                     │   │
│  │                                                                      │   │
│  │ 点级别数据（N = 总聚类数）:                                            │   │
│  │   - points: [N, 3] 归一化 3D 坐标                                    │   │
│  │   - features: [N, 1024] DINOv2 特征（fp16）                          │   │
│  │   - colors: [N, 3] RGB 颜色                                          │   │
│  │   - cluster_sizes: [N] 每聚类的点数                                  │   │
│  │   - ray_dirs: [N, 3] 主射线方向                                      │   │
│  │   - ray_dirs_mean: [N, 3] 平均 + 归一化策略                          │   │
│  │   - ray_dirs_dominant: [N, 3] 最高相似度策略                         │   │
│  │   - ray_dirs_first: [N, 3] 第一条射线策略                            │   │
│  │   - plucker_rays: [N, 6] Plücker 坐标                                │   │
│  │                                                                      │   │
│  │ 视图级别相机信息（M = 视图数）:                                       │   │
│  │   - view_camera_centers: [M, 3] 相机位置                             │   │
│  │   - view_camera_rotations: [M, 3, 3] 旋转矩阵                         │   │
│  │   - view_camera_intrinsics: [M, 3, 3] 内参矩阵                       │   │
│  │   - view_plucker_main_rays: [M, 6] 主射线 Plücker 坐标                │   │
│  │                                                                      │   │
│  │ 归一化:                                                              │   │
│  │   - mu: [3] 场景均值                                                 │   │
│  │   - sigma: 场景标准差                                                │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 详细算法解释

### 1. 视角选择

**目的**：从训练数据中选择代表性的帧子集。

**方法**：
- **均匀采样**：每隔 k 帧选择一帧（回退方案）
- **FPS（最远点采样）**：选择最大化空间覆盖的视图

**输出**：M 个记忆视图（默认：100）

### 2. 特征提取

**目的**：使用 MapAnything 模型从 RGB 图像中提取多尺度语义特征。

**模型**：MapAnything（DINOv2 ViT-L/14 骨干网络 + info-sharing）
- 在大规模图像数据上预训练
- 输出 24 个中间层 + 1 个最终层
- 我们使用层 [0, 6, 12, 18, 24] 进行 DPT 风格的多尺度特征提取
- 每层输出 1024 维特征

**过程**：
```python
# MapAnything 推理并保存中间特征
model.infer(
    input_views,
    save_filename=temp_file,
    memory_efficient_inference=True
)

# 加载保存的特征
saved_data = torch.load(temp_file)
raw_interm = saved_data["intermediate"]  # 24 层
raw_final = saved_data["final"]  # 最终层

# 提取目标层: [0, 6, 12, 18] + 最终层
feat_list = []
for layer_idx in [0, 6, 12, 18]:
    feat_list.append(raw_interm[layer_idx]["features"][view_idx])
feat_list.append(raw_final["features"][view_idx])
```

**多尺度特征处理**（DPT 风格）：
```python
# 1. 将所有特征重塑为空间格式（如果需要）
# 2. 找到所有层中的最大分辨率
# 3. 使用双线性插值将较小特征上采样到最大分辨率
# 4. 沿通道维度连接
# 5. 展平为 (grid_H * grid_W, C_total)

features_flat, grid_H, grid_W = process_multiscale_features_to_grid(feat_list)
# 输出: features_flat [grid_H * grid_W, ~5000], grid_H, grid_W (通常 37x37)
```

### 3. 反投影（2D 网格到 3D）

**目的**：使用真实深度将网格分辨率特征转换为 3D 点。

**关键差异**：不是逐像素反投影，而是在**网格中心**采样深度。

**数学公式**：

给定：
- 深度图: Z[u, v] 在图像分辨率（如 518×518）
- 相机内参: K (3x3 矩阵)
- 相机位姿: T = [R | t] (4x4 矩阵)
- 网格尺寸: grid_H × grid_W (如 37×37)

**分步骤**：

1. **创建网格中心坐标**：
   ```
   u_grid = linspace(0, img_W - 1, grid_W)
   v_grid = linspace(0, img_H - 1, grid_H)
   ```

2. **在网格中心采样深度**：
   ```
   z_sampled = depth[v_grid_int, u_grid_int]
   ```

3. **过滤有效深度**：
   ```
   valid_mask = (z_sampled > min_depth) & (z_sampled < max_depth)
   ```

4. **反投影到相机坐标**：
   ```
   x_cam = (u_valid - cx) * z_valid / fx
   y_cam = (v_valid - cy) * z_valid / fy
   P_cam = [x_cam, y_cam, z_valid]
   ```

5. **变换到世界坐标**：
   ```
   P_world = R * P_cam + t
   ```

6. **计算射线方向**：
   ```
   camera_center = -R^T * t
   ray_dir = normalize(P_world - camera_center)
   ```

7. **采样颜色和特征**：
   - 颜色：在网格中心进行双线性插值
   - 特征：已在网格分辨率，过滤到有效点

**每个视图的输出**：
- points: [N_valid, 3] - 有效的 3D 点（N_valid ≤ grid_H × grid_W）
- features: [N_valid, ~5000] - 多尺度 DINOv2 特征
- colors: [N_valid, 3] - RGB 颜色
- ray_dirs: [N_valid, 3] - 单位射线方向
- camera_centers: [N_valid, 3] - 相机中心（重复）

**压缩**：基于网格的采样将点数减少约 196 倍（从 518×518 到 37×37）。

### 4. BSE 池化

**目的**：在保持边界的同时压缩点云。

#### 4.1 粗糙体素哈希

**目标**：将附近的点分组到体素中。

**方法**：
```python
# 将点量化到体素网格
voxel_coords = torch.floor(points / voxel_size).long()

# 使用质数编码以确保唯一 ID
voxel_id = (voxel_coords[:, 0] * P1 +
             voxel_coords[:, 1] * P2 +
             voxel_coords[:, 2] * P3)

# 按 voxel_id 对点分组
unique_ids, inverse_indices = torch.unique(voxel_id, return_inverse=True)
```

#### 4.2 聚类均值特征

**目标**：为每个聚类计算原型特征。

**方法**：
```python
# 对每个体素，计算均值特征
cluster_means = torch.zeros(len(unique_ids), feature_dim)
for i in range(len(unique_ids)):
    mask = (inverse_indices == i)
    cluster_means[i] = features[mask].mean(dim=0)
```

#### 4.3 Otsu 自适应阈值

**目标**：找到最优阈值来分离内点和离群点。

**方法**：
1. 计算每个点与聚类均值之间的余弦相似度：
   ```
   similarity = (features @ cluster_means.T) / (||features|| * ||cluster_means||)
   ```

2. 构建相似度直方图（256 个 bin）

3. 应用 Otsu 方法找到使类间方差最大化的阈值：
   ```
   对于每个可能的阈值 t:
       w0 = t 以下像素的权重
       w1 = t 以上像素的权重
       mu0 = t 以下像素的均值
       mu1 = t 以上像素的均值
       variance = w0 * w1 * (mu0 - mu1)^2

   选择使方差最大化的 t
   ```

4. 单峰检查：如果相似度 std < 0.02，跳过分割

#### 4.4 二分分割

**目标**：将边界点与主聚类分离。

**方法**：
```python
# 主聚类：高相似度点
main_mask = similarity >= threshold

# 离群聚类：低相似度点（边界）
outlier_mask = similarity < threshold

# 这将几何边界单独保留
```

#### 4.5 细粒度池化

**目标**：为每个聚类计算池化属性。

**点级别池化**：
```python
pooled_point = points[mask].mean(dim=0)
pooled_feature = features[mask].mean(dim=0)
pooled_color = colors[mask].mean(dim=0)
```

**射线池化策略**：

1. **平均 + 归一化**：
   ```python
   ray_mean = ray_dirs[mask].mean(dim=0)
   ray_pooled = F.normalize(ray_mean, dim=-1)
   ```

2. **主导射线**：
   ```python
   # 选择与聚类均值相似度最高的射线
   similarities = cosine_similarity(ray_dirs[mask], cluster_mean)
   ray_pooled = ray_dirs[mask][similarities.argmax()]
   ```

3. **第一条射线**：
   ```python
   # 简单地使用聚类中的第一条射线
   ray_pooled = ray_dirs[mask][0]
   ```

4. **Plücker 编码**：
   ```python
   # 6D 表示: (方向, 矩)
   direction = ray_pooled
   moment = torch.cross(camera_center, direction)
   plucker = torch.cat([direction, moment])  # [6]
   ```

**压缩比**：典型压缩从 ~100 万点到 ~5 万聚类（20 倍）。

### 5. 两遍处理

**目的**：高效内存管理与全局归一化。

#### 第一遍：提取 + 池化 + 累积

```python
for view_idx in range(M):
    # 提取特征
    features = dinov2(images[view_idx])

    # 反投影到 3D
    points, ray_dirs, colors, features, camera_centers = unproject(...)

    # 可选: SOR 过滤
    if enable_sor:
        inliers = sor_filter(points)
        points = points[inliers]

    # BSE 池化
    pooled = bse_pooler.pool(points, features, colors, ray_dirs, camera_centers)

    # 累积统计
    welford.update(pooled['points'])

    # 收集视图级别的相机信息
    view_info['camera_centers'].append(camera_center)
    view_info['camera_rotations'].append(R)
    view_info['camera_intrinsics'].append(K)
    # ... 计算 Plücker 主射线

    # 保存临时分块
    torch.save(pooled, f"/dev/shm/chunk_{view_idx}.pt")
```

#### 第二遍：归一化 + 组装

```python
# 完成全局统计
mu, sigma = welford.finalize()

# 加载并归一化所有分块
final_memory = {}
for chunk_path in chunk_paths:
    pooled = torch.load(chunk_path)

    # 将点归一化到零均值、单位方差
    pooled['points'] = (pooled['points'] - mu) / sigma

    # 连接到最终记忆
    final_memory['points'].append(pooled['points'])
    # ... 其他属性同理

# 连接所有分块
result = {
    'points': torch.cat(final_memory['points'], dim=0),
    'features': torch.cat(final_memory['features'], dim=0),
    # ...
}

# 组装视图级别的相机信息
view_info = {
    'camera_centers': torch.stack(view_camera_centers, dim=0),  # [M, 3]
    'camera_rotations': torch.stack(view_camera_rotations, dim=0),  # [M, 3, 3]
    'camera_intrinsics': torch.stack(view_camera_intrinsics, dim=0),  # [M, 3, 3]
    'plucker_main_rays': torch.stack(view_plucker_main_rays, dim=0),  # [M, 6]
}
```

### 6. Welford 算法

**目的**：数值稳定的均值和方差计算。

**为什么用 Welford？**：
- 传统方法: `variance = E[X^2] - E[X]^2` 会遭受灾难性抵消
- Welford: 在单次遍历中更新均值和方差，数值稳定

**算法**：
```python
# 初始化
count = 0
mean = 0.0
M2 = 0.0  # 差的平方和

# 为每个新点 x 更新
count += 1
delta = x - mean
mean += delta / count
delta2 = x - mean
M2 += delta * delta2

# 完成
variance = M2 / count
std = sqrt(variance)
```

---

## 射线方向处理

### 为什么需要多种策略？

从多个视角观察物体时，射线方向可能差异很大：
- 桌子上的同一个点可能从不同角度观察
- 简单平均（`mean + normalize`）会模糊这种多视角信息

### 可用策略

| 策略 | 公式 | 用例 | 大小 |
|------|------|------|------|
| `mean` | `normalize(mean(rays))` | 基线池化 | [N, 3] |
| `dominant` | `argmax(similarity)` | 最具代表性的射线 | [N, 3] |
| `first` | `rays[0]` | 保留原始方向 | [N, 3] |
| `all` | 保留所有射线 | 下游学习 | 可变 |

### Plücker 射线编码

**定义**：3D 直线的 6D 表示。

```
L = (d, m)
其中:
    d ∈ R^3 是方向向量
    m ∈ R^3 是矩向量: m = camera_center × d
```

**性质**：
- 唯一标识一条 3D 直线
- 满足 Grassmann-Plücker 约束: d · m = 0
- 适用于几何运算（线线距离、相交测试）

**在我们的流程中**：
- 点级别: `plucker_rays` [N, 6] - 池化后的 Plücker 坐标
- 视图级别: `view_plucker_main_rays` [M, 6] - 主相机射线 Plücker 坐标

---

## 使用示例

```bash
python -m ace_dinov2_lmc.memory_extraction.run_memory_extraction \
    /data/xwh/7Scenes/pgt_7scenes_chess \
    output/chess_memory.pt \
    --n_memory 100 \
    --voxel_size 0.05 \
    --use_otsu \
    --ray_pool_strategy mean \
    --device cuda:0
```

**预期输出**：
```
[BSE Memory] Dataset: /data/xwh/7Scenes/pgt_7scenes_chess
[BSE Memory] Output: output/chess_memory.pt
[BSE Memory] N_MEMORY: 100, BSE: True
[BSE Memory] Voxel size: 0.05, Otsu: True
[BSE Memory] Train dataset: 1000 samples
[BSE Memory] Selected 100 views
[BSE Memory] Using standalone DINOv2 extractor
[BSE Memory] Extracting features...
[BSE Memory] Two-pass processing...
[Pass 1] Extracting and pooling...
100%|████████████████████| 100/100
[Pass 2] Normalizing and assembling...
[Save] Memory saved to output/chess_memory.pt
[Save] Schema version: 1.2
[Save] Total points: 52341
[Save] Feature dim: 1024
[Save] Scene mean: [1.234, -0.567, 2.345]
[Save] Scene sigma: 1.2345
[Save] View count: 100
[BSE Memory] Done!
```

---

## 关键设计决策

### 1. 为什么用 BSE 而不是体素池化？

| 方面 | 体素池化 | BSE |
|------|----------|-----|
| 边界保持 | ❌ 合并边界 | ✅ 分离边界 |
| 自适应阈值 | ❌ 固定体素 | ✅ Otsu 自适应 |
| 压缩比 | 固定 | 自适应 |
| 实现复杂度 | 简单 | 中等 |

### 2. 为什么用两遍处理？

| 替代方案 | 优点 | 缺点 |
|-------------|------|------|
| 单遍 | 更简单 | 无法在没有完整统计的情况下归一化 |
| 两遍 | 全局归一化、恢复能力 | 稍微复杂 |

### 3. 为什么用多种射线策略？

- **Mean**：基线，简单
- **Dominant**：保留最具代表性的方向
- **First**：无处理开销
- **All**：为下游学习提供最多信息

### 4. 为什么保存视图级别的相机信息？

- 下游 Transformer 可以使用相机位姿进行视角依赖推理
- 支持几何一致性检查
- 支持未来扩展（如视图选择、注意力机制）

---

## 故障排除

### 内存不足

```bash
# 减少记忆视图数量
--n_memory 50

# 增大体素大小（更粗糙的池化）
--voxel_size 0.08

# 减少特征提取的批大小
#（修改代码以较小的批次处理）
```

### 处理速度慢

```bash
# 禁用 SOR 过滤
--enable_sor false

# 使用更大的体素大小
--voxel_size 0.06

# 禁用 Otsu（使用固定阈值）
--use_otsu false
```

### 结果质量差

- **边界模糊**：减小 `voxel_size`（例如 0.03）
- **聚类太少**：减小 `voxel_size`
- **聚类太多**：增大 `voxel_size`
- **特征噪声**：启用 SOR 过滤 `--enable_sor`

---

## 参考文献

1. **ACE: Accelerated Coordinate Encoding** (CVPR 2023)
   - 原始场景坐标回归框架

2. **DINOv2: Learning Robust Visual Features without Supervision** (2023)
   - 特征提取骨干网络

3. **Otsu's Method** (1979)
   - 用于图像二值化的自适应阈值

4. **Welford's Algorithm** (1962)
   - 数值稳定的在线方差计算

5. **Plücker Coordinates** (19 世纪)
   - 3D 直线的 6D 表示
