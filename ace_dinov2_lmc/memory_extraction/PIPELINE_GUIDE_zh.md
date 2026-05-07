# BSE Memory Extraction 流程指南

本文档说明 **当前仓库真实脚本**（`extract_memory.sh` → `run_memory_extraction.py`）下的 BSE Memory Extraction 流程；参数默认值、路径与 `map-anything/bash_scripts/ace/fps_memory.sh` 对齐处会单独标注。

## 概述

该流程从多视角 RGB-D 数据中提取压缩的场景表示，包括：

1. **视角选择**：优先调用 `mapanything.tasks.ace.memory_selection.select_optimal_memory_indices`（失败则均匀采样）
2. **加载视图 + 深度/RGB 校验导出**（与 `fps_memory` 一致）：在特征推理前写入本次 run 目录下的 `depth_validation/`、`model_input_vis/`
3. **特征提取**：默认 `MapAnythingExtractor` → `model.infer(memory_views, ...)`，从 `store_info_sharing_intermediate_features` 取多尺度特征（非「先 torch.save 再读临时文件」的主路径）
4. **反投影**：在特征网格上对深度采样（`--patch_depth_sampling`：`nearest` / `median` / `nearest_valid`），再与 `process_multiscale_features_to_grid` 展平顺序对齐
5. **BSE 池化** + **Welford 两遍归一化**：临时块默认写在 `--temp_dir`（默认 `/dev/shm`）

**默认视角数**：`extract_memory.sh` 中 `N_VIEWS` 默认为 **20**；直接调用 Python 时 `--n_memory` 默认为 **100**（以实际命令为准）。

输出主文件为 **`memory_bse.pt`**（`--output_path` / 脚本中的 `OUTPUT_FILE`），并包含：
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
│  │ 方法: 优先 FPS（`select_optimal_memory_indices`），否则均匀采样        │   │
│  │ 输出: M 个记忆视图（M = `N_VIEWS` / `--n_memory`）                     │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           步骤 2: 特征提取                                  │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ 输入: M 张 RGB 图像                                                  │   │
│  │ 模型: MapAnything（DINOv2 骨干 + 多尺度中间层拼接）                    │   │
│  │ 输出: 每层特征经对齐后与网格 grid_H×grid_W 对应（维度为拼接通道数）     │   │
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
│  │   - features: [N_i, C] 多尺度拼接特征                                 │   │
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
│  │   - features: [N, C] 多尺度拼接特征（fp16，C 为配置相关数千维）        │   │
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

**与真实脚本一致的补充**：在「步骤 2」之前，`run_memory_extraction.py` 会调用 `export_memory_views_visualization`，向 **`OUTPUT_DIR`**（即本次 run 的时间戳目录）写入 `depth_validation/`、`model_input_vis/`；深度张量会先经 `normalize_depth_to_hw` 处理 `[H,W,1]` 等布局。详见 `README_zh.md`。

---

## 脚本入口、路径与 `fps_memory` 对齐项

| 项目 | 真实行为（`extract_memory.sh`） |
|------|----------------------------------|
| 输出根目录 | `OUTPUT_ROOT` 默认为 **脚本所在目录**下的 `04_evaluation/memory_extract/`，即 `.../memory_extraction/04_evaluation/memory_extract/` |
| 单次 run 目录 | `${OUTPUT_ROOT}/${SCENE_NAME}/${N_VIEWS}v_${CONFIG_TAG}/${TIMESTAMP}/` |
| 主输出文件 | `memory_bse.pt`（`OUTPUT_FILE`） |
| `DATASET_PATH` | 等于 `DATASET_ROOT`；7scenes/indoor6 默认指向 WAI 根路径（见脚本内 `case`） |
| `DEPTH_MIN`/`DEPTH_MAX` | indoor6 默认 `0.1`/`100.0`；7scenes 默认 `0.1`/`6.0` |
| `PATCH_DEPTH_SAMPLING` | indoor6 默认 **`nearest_valid`**；否则默认 **`nearest`**（与 `fps_memory.sh` 一致） |
| `GPU_ID` | 默认 **3**（`CUDA_VISIBLE_DEVICES`） |
| 传入 Python | `--depth_valid_range`、`--patch_depth_sampling` 等由脚本拼接 |

---

## 详细算法解释

### 1. 视角选择

**目的**：从训练数据中选择代表性的帧子集。

**方法**：
- **优先**：`select_optimal_memory_indices`（与 map-anything 共用）
- **回退**：均匀采样（`run_memory_extraction.select_memory_views` 内 `ImportError` 分支）

**输出**：M 个记忆视图（M = 环境变量 `N_VIEWS` 或 CLI `--n_memory`）

### 2. 特征提取

**目的**：从 RGB 图像中提取多尺度语义特征。

**默认模型**：MapAnything（DINOv2 ViT-L/14 骨干网络 + AAT 24 层 + DPT 预测头）
- 在大规模数据上预训练的统一模型
- 目标中间层索引为 `[2,5,8,11,14,17,20,23]` 等（见 `MapAnythingExtractor.TARGET_INTERM_LAYERS`）+ 最终层；单层通道维常见为 1024，**拼接后** `C_total` 为数千维（由 `process_multiscale_features_to_grid` 对齐并 concat）

**Fallback 模型**：`DINOv2Extractor`（`USE_MODEL=dinov2` 或 MapAnything 初始化失败时）

**当前主路径（MapAnything）**（与伪代码差异说明）：
- 使用 `MapAnythingExtractor` 加载 `mapanything_store_intermediates_ace` 等配置，`model.infer(memory_views, memory_efficient_inference=True, ...)` **不写**中间特征到临时 pt 再读回的主流程；
- 多尺度张量来自 `model.get_info_sharing_intermediate_features()`，再经 `_process_saved_features` → `process_multiscale_features_to_grid` 对齐网格。

**过程（逻辑示意）**：
```python
# MapAnything（与仓库实现一致）
predictions = extractor.model.infer(memory_views, memory_efficient_inference=True, ...)
stored = extractor.model.get_info_sharing_intermediate_features()
features_dict = extractor._process_saved_features(stored, n_views)

# DINOv2 分支：拼接 batch 张量后 extract(...)
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

**目的**：使用真实深度将**与特征对齐的网格**上的点反投影到 3D，与 `mapanything.tasks.run_memory_extraction.generate_patch_point_cloud` 使用相同的网格与采样语义。

**关键差异**：不是全图逐像素，而是在 **grid_H×grid_W** 上与特征一一对应；深度张量先 **`normalize_depth_to_hw`**（处理 `[H,W,1]` / `[B,H,W]` 等），避免误用 `depth[0]` 只取一行。

**网格与 flatten 顺序**：使用 `meshgrid(..., indexing='ij')` 得到 `grid_v, grid_u`，再 `reshape(-1)`，与 `feat.permute(0,2,3,1).reshape(-1,C)` 的先行后列顺序一致。

**深度采样**（`--patch_depth_sampling`，与 `fps_memory` 的 `PATCH_DEPTH_SAMPLING` 一致）：

| 模式 | 行为 |
|------|------|
| `nearest` | `z = depth[gy, gx]`（整数格点，与单像素最近） |
| `median` | 以网格为中心的 3×3 邻域深度的中位数 |
| `nearest_valid` | 半径 **10** 像素窗口内，落在 `[depth_min, depth_max]` 且有限的深度的**中位数**；Indoor6 稀疏深度**推荐** |

**数学公式（子步骤）**：

给定深度图 `Z` 形状 `[H,W]`、内参 `K`、位姿 `T=[R|t]`、网格 `grid_H×grid_W`：

1. 生成网格中心 `(u, v)`（浮点，与 `ij` 网格一致）
2. 按上表得到每格 `z_sampled`
3. `valid_mask = (z > depth_min) & (z < depth_max)`
4. `x_cam = (u - cx) * z / fx`，`y_cam = (v - cy) * z / fy`，`P_world = P_cam @ R.T + t`（实现与脚本中矩阵形状一致）
5. `ray_dir = normalize(P_world - camera_center)`，`camera_center = -R^T t`
6. **特征**：`features_flat[valid_mask]`（CPU/GPU 与 mask 对齐）；若 batch 含 `images` 键则对 RGB `grid_sample`，否则灰色占位

**每个视图的输出**：
- points: [N_valid, 3]
- features: [N_valid, C_total]（多尺度拼接维度，通常为数千维而非单层 1024）
- colors / ray_dirs / camera_centers 等同理

**压缩**：网格点数约为 `grid_H*grid_W`（例如 37×37），相对全分辨率像素大幅减少。

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
    # 已由 infer 得到 features_dict[view_idx]; 与 raw_batches[view_idx] 深度对齐
    features_flat, grid_H, grid_W = process_multiscale_features_to_grid(feat_list)
    points, ray_dirs, colors, features_flat, camera_centers = unproject_with_grid_features(
        batch_gpu, features_flat, grid_H, grid_W,
        config.depth_valid_range,
        sampling_method=config.patch_depth_sampling,
    )

    if enable_sor:
        inliers = sor_filter(points)
        points = points[inliers]
        # ... 同步过滤 features_flat / colors / ray_dirs

    pooled = pooler.pool(points, features_flat, colors, ray_dirs, camera_centers=camera_centers)
    welford.update(pooled['points'])
    # ... 视图级相机与 Plücker
    torch.save(pooled, os.path.join(config.temp_dir, f"chunk_{view_idx}.pt"))
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

**推荐（与仓库一致）**：在 `ace_depth` 根目录执行 `extract_memory.sh`，由脚本设置 `PYTHONPATH`、`CUDA_VISIBLE_DEVICES`、`OUTPUT_DIR` 及 `--depth_valid_range`、`--patch_depth_sampling` 等。

```bash
cd /home/xwh/project/ace_depth

# 7-Scenes（WAI ROOT，与脚本默认 DATASET_ROOT 一致）
bash ace_dinov2_lmc/memory_extraction/extract_memory.sh

# Indoor6 + 稀疏深度默认 nearest_valid
DATASET_TYPE=indoor6 SCENE_TRAIN=scene2a_train N_VIEWS=40 \
  bash ace_dinov2_lmc/memory_extraction/extract_memory.sh
```

**直接调用 Python**（需自行保证 `PYTHONPATH` 含 `ace_depth` 与 `map-anything`；`dataset_path` 为 WAI 数据集 ROOT）：

```bash
python -u -m ace_dinov2_lmc.memory_extraction.run_memory_extraction \
    /home/xwh/data/mapanything-dataset/wai_data/indoor6 \
    /path/to/run_dir/memory_bse.pt \
    --n_memory 40 \
    --dataset_type indoor6 \
    --scene_name scene2a_train \
    --depth_valid_range 0.1 100 \
    --patch_depth_sampling nearest_valid \
    --device cuda:0
```

**预期终端日志（节选，真实脚本）**：
```
[BSE Memory] Dataset: ...
[BSE Memory] Scene: scene2a_train (type=indoor6)
[BSE Memory] Output: .../memory_bse.pt
[Config] patch_depth_sampling: nearest_valid (align map-anything fps_memory: ...)
[Config] Depth valid range: [0.100, 100.000] m
[Data] Selected memory indices (40): [...]
[Vis] Exporting depth/RGB validation under run dir: .../<timestamp>
[BSE Memory] Extracting features...
[BSE Memory] Two-pass processing...
[Pass 1] Extracting and pooling...
[Pass 2] Normalizing and assembling...
[BSE Memory] Done!
```

主输出旁会生成 `extraction_config.json`、`extraction_log.txt`，以及 `depth_validation/`、`model_input_vis/`（见 `README_zh.md`）。

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

### `ValueError: No points accumulated` / 网格有效点极少（Indoor6）

- 确认 **`--patch_depth_sampling nearest_valid`**（`extract_memory.sh` 对 indoor6 已默认）且 **`--depth_valid_range`** 与数据一致（如 `0.1 100`）。
- 确认深度为 `[H,W]` / `[H,W,1]` 等常见布局（已内置 `normalize_depth_to_hw`）。

### 内存不足

```bash
# 减少记忆视图数量（脚本用环境变量 N_VIEWS）
N_VIEWS=20 bash ace_dinov2_lmc/memory_extraction/extract_memory.sh

# 或直接调用 Python
--n_memory 20 --voxel_size 0.08
```

### 处理速度慢

```bash
# 脚本默认不启用 SOR；若曾加 --enable_sor，可去掉该 flag
# 增大体素（脚本环境变量 VOXEL_SIZE）
VOXEL_SIZE=0.08 bash ace_dinov2_lmc/memory_extraction/extract_memory.sh
```

### 结果质量差

- **边界模糊**：减小 `VOXEL_SIZE` / `--voxel_size`
- **聚类数量不合适**：同上调节体素
- **特征噪声**：按需启用 SOR：`ENABLE_SOR=true bash ...`（对应 `--enable_sor`）

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
