# Memory Extraction Pipeline 代码阅读指南

## 概述

这个模块实现了基于BSE（Bilateral Supervoxel Extraction）的内存提取管道，用于从多视图RGB-D数据中提取压缩的场景记忆。

**核心功能：**
1. 从训练数据集中选择最优记忆视图
2. 使用MapAnything/DINOv2提取多尺度特征
3. 通过BSE进行3D点云池化
4. 生成压缩的场景记忆文件

## 代码结构

```
memory_extraction/
├── run_memory_extraction.py   # 主入口脚本
├── bse_pooling.py             # BSE算法实现
├── welford_meter.py           # 在线统计归一化
└── extract_memory.sh          # Shell启动脚本
```

## 阅读顺序

### 第一步：理解数据流（推荐顺序）

1. **run_memory_extraction.py: main()函数** (line 1556-1745)
   - 程序入口点
   - 参数解析和配置
   - 调用各个处理阶段

2. **数据加载流程** (line 1598-1640)
   - `load_dataset()`: 加载WAI格式数据集
   - `select_memory_views()`: 选择最优视图

3. **特征提取流程** (line 1641-1720)
   - MapAnythingExtractor / DINOv2Extractor
   - 特征提取和格式化

4. **Two-Pass处理** (line 1062-1260)
   - `two_pass_processing()`: 核心处理循环
   - 点云生成、BSE池化、统计归一化

5. **BSE池化算法** (bse_pooling.py)
   - 理解双边聚类和体素合并

### 第二步：深入核心组件

#### 1. 主入口流程 (run_memory_extraction.py)

**关键函数：main()**

```python
def main():
    # 1. 解析配置
    config = parse_args()

    # 2. 初始化模块
    welford = WelfordNormalizer()  # 在线统计
    pooler = BSEPooler()           # BSE池化器

    # 3. 加载数据集
    train_dataset = load_dataset(...)

    # 4. 选择记忆视图
    memory_indices = select_memory_views(train_dataset, n_memory)

    # 5. 准备输入数据
    # - raw_batches: 原始数据（depth, intrinsics等）
    # - memory_views: 模型输入（处理后的tensor）

    # 6. 提取特征
    if use_mapanything:
        # MapAnything特征提取
        model.infer(memory_views)
        features = model.get_info_sharing_intermediate_features()
    else:
        # DINOv2特征提取
        features = extractor.extract(images, depths, poses, intrinsics)

    # 7. Two-pass处理
    result = two_pass_processing(memory_views, raw_batches, features, ...)

    # 8. 保存记忆文件
    save_memory(output_path, result, ...)
```

**关键点：**
- Line 1615-1635: 同时保存`raw_batches`和`memory_views`
  - `raw_batches`: 原始数据字典，包含depth, intrinsics, pose等
  - `memory_views`: 处理后的模型输入，包含img, depth_z, intrinsics等

#### 2. 数据准备 (prepare_batch_input)

**位置：** line 607-674

**功能：** 将原始数据转换为模型输入格式

```python
def prepare_batch_input(raw_data, device, mode="memory"):
    view_input = {}

    # 1. 图像处理
    img = raw_data["img"].to(device)
    view_input["img"] = img  # [B, 3, H, W]
    view_input["data_norm_type"] = ["dinov2"] * B

    # 2. 内参处理
    intr = raw_data["camera_intrinsics"]
    view_input["intrinsics"] = intr  # [B, 3, 3]

    if mode == "memory":
        # 3. 位姿处理
        pose = raw_data["camera_pose"]
        view_input["camera_poses"] = pose  # [B, 4, 4]

        # 4. 深度处理
        depth = raw_data["depthmap"]
        view_input["depth_z"] = depth  # [B, H, W, 1]

    return view_input
```

**重要细节：**
- MapAnything要求特定key名称：`depth_z`（不是`depths`）
- 深度格式：`[B, H, W, 1]`（最后一维是通道）
- 必须包含`data_norm_type`字段

#### 3. 特征提取

##### MapAnything模式

**位置：** line 150-354

**关键步骤：**

```python
class MapAnythingExtractor:
    def __init__(self, ...):
        # 初始化MapAnything模型
        self.model = MapAnything(**config)

        # 强制启用特征存储
        self.model.model_config.store_info_sharing_intermediate_features = True

    def _process_saved_features(self, stored_data, n_views):
        # 处理中间层特征
        # stored_data包含：
        # - "final": 最终层特征
        # - "intermediate": 中间层特征列表
        pass
```

**特征提取流程（main函数中）：**

```python
# 1. 调用推理
predictions = model.infer(
    memory_views,
    memory_efficient_inference=True,
    use_amp=False,  # 禁用AMP避免BFloat16问题
)

# 2. 从内存获取特征（不是文件！）
stored_features = model.get_info_sharing_intermediate_features()

# 3. 处理特征
features_dict = extractor._process_saved_features(stored_features, n_views)
```

**关键配置：**
- 必须设置`store_info_sharing_intermediate_features = True`
- 特征存储在内存中，通过`get_info_sharing_intermediate_features()`获取
- **不要设置**`info_sharing_storage_path`，否则会尝试写文件

##### DINOv2模式

**位置：** line 357-570

**特征提取：**

```python
def extract(self, images, depths, poses, intrinsics):
    # 1. 编码图像
    features = self.encoder(images)

    # 2. 提取中间层
    for layer_idx in intermediate_layers:
        layer_feat = self.backbone.blocks[layer_idx](features)
        all_features.append(layer_feat)

    return features_dict
```

#### 4. Two-Pass处理核心

**位置：** line 1062-1260

**整体流程：**

```python
def two_pass_processing(memory_views, raw_batches, features_dict, pooler, welford, config):
    # Pass 1: 提取 -> 反投影 -> 池化 -> 累积统计
    for view_idx, (view_data, raw_batch) in enumerate(zip(memory_views, raw_batches)):
        # 1. 获取特征
        feat_list = features_dict[view_idx]['features']
        grid_H = features_dict[view_idx]['grid_H']
        grid_W = features_dict[view_idx]['grid_W']

        # 2. 处理特征：对齐到网格分辨率
        features_flat = process_multiscale_features_to_grid(feat_list)

        # 3. 从raw_batch准备几何数据
        batch_gpu = {}
        for key in ["depth", "depthmap", "depth_z", "intrinsics", ...]:
            if key in raw_batch:
                batch_gpu[key] = raw_batch[key].to(device)

        # 4. 反投影：生成3D点云
        points, ray_dirs, colors, features, camera_centers = \
            unproject_with_grid_features(batch_gpu, features_flat, grid_H, grid_W)

        # 5. BSE池化
        pooled = pooler.pool(points, features, colors, ray_dirs, camera_centers)

        # 6. 更新统计
        welford.update(pooled['points'])

        # 7. 保存块到磁盘
        save_chunk(pooled, chunk_path)

    # Pass 2: 最终化统计 -> 加载块 -> 归一化 -> 拼接
    mu, sigma = welford.finalize()

    for chunk_path in chunk_paths:
        chunk = load_chunk(chunk_path)
        normalized = (chunk['points'] - mu) / sigma
        all_points.append(normalized)

    result = concatenate(all_points)
    return result, mu, sigma, view_info
```

**关键细节：**

1. **使用raw_batches获取深度** (line 1130-1142)
   ```python
   # 从原始数据构建batch_gpu
   batch_gpu = {}
   for key in ["depth", "depthmap", "depth_z", ...]:
       if key in raw_batch:
           batch_gpu[key] = raw_batch[key].to(device)
   ```
   - **重要**：必须从`raw_batch`获取，`memory_views`中没有完整深度数据

2. **深度格式处理** (unproject_with_grid_features, line 805-828)
   ```python
   # 处理各种深度格式
   if depth.ndim == 4:
       if depth.shape[1] == 1:
           depth = depth[0, 0]  # [B, 1, H, W] -> [H, W]
       elif depth.shape[-1] == 1:
           depth = depth[0, :, :, 0]  # [B, H, W, 1] -> [H, W]
   ```

3. **网格特征采样** (line 840-870)
   - 在网格中心采样深度
   - 过滤有效深度点
   - 反投影到3D空间

#### 5. BSE池化算法

**文件：** bse_pooling.py

**核心思想：**
- 使用双边聚类保留几何边界
- Otsu阈值分割前景/背景
- 多策略射线池化

**关键步骤：**

```python
class BSEPooler:
    def pool(self, points, features, colors, ray_dirs, camera_centers):
        # 1. 创建双边体素网格
        # 空间维度 + 颜色维度
        bilateral_space = BilateralSpace(points, colors, voxel_size)

        # 2. 聚类到超体素
        supervoxels = bilateral_space.cluster()

        # 3. Otsu阈值（可选）
        if self.use_otsu:
            foreground_mask = otsu_threshold(supervoxels)

        # 4. 射线池化
        for supervoxel in supervoxels:
            if ray_pool_strategy == 'mean':
                ray_dir = rays.mean()
            elif ray_pool_strategy == 'dominant':
                ray_dir = dominant_direction(rays)

        # 5. 合并特征
        pooled_features = supervoxels.aggregate(features)

        return {
            'points': pooled_points,
            'features': pooled_features,
            'colors': pooled_colors,
            'ray_dirs': pooled_ray_dirs,
        }
```

**双边空间定义：**
```python
# 5维双边空间：[x, y, z, r, g, b]
bilateral_coords = [
    points[:, 0] / voxel_size,  # x
    points[:, 1] / voxel_size,  # y
    points[:, 2] / voxel_size,  # z
    colors[:, 0] / color_sigma, # r
    colors[:, 1] / color_sigma, # g
    colors[:, 2] / color_sigma, # b
]
```

#### 6. Welford在线归一化

**文件：** welford_meter.py

**用途：** 两遍处理中的在线统计累积

```python
class WelfordNormalizer:
    def update(self, points):
        # 在线更新均值和方差
        self.count += len(points)
        delta = points - self.mean
        self.mean += delta / self.count
        delta2 = points - self.mean
        self.M2 += delta * delta2

    def finalize(self):
        # 计算最终统计量
        variance = self.M2 / self.count
        std = torch.sqrt(variance)
        return self.mean, std
```

**为什么需要两遍：**
1. Pass 1: 累积所有点的统计量（均值、方差）
2. Pass 2: 使用全局统计量归一化所有点

## 数据流图

```
┌─────────────────────┐
│  1. 数据集加载      │
│  load_dataset()     │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  2. 视图选择        │
│  select_memory_     │
│  views()            │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐     ┌──────────────────────┐
│  3a. 原始数据       │     │  3b. 模型输入        │
│  raw_batches        │     │  memory_views        │
│  (depth, intrin...) │     │  (img, depth_z...)   │
└──────┬──────────────┘     └──────┬───────────────┘
       │                           │
       │                           ▼
       │                  ┌─────────────────────┐
       │                  │  4. 特征提取        │
       │                  │  MapAnything/DINOv2 │
       │                  └──────┬──────────────┘
       │                         │
       │                         ▼
       │                  ┌─────────────────────┐
       │                  │  5. 特征处理        │
       │                  │  process_multi...   │
       │                  └──────┬──────────────┘
       │                         │
       └─────────────────────────┤
                                 │
                                 ▼
                        ┌─────────────────────┐
                        │  6. Two-Pass处理    │
                        │  two_pass_          │
                        │  processing()       │
                        └──────┬──────────────┘
                               │
                ┌──────────────┼──────────────┐
                ▼              ▼              ▼
         ┌──────────┐   ┌──────────┐   ┌──────────┐
         │ 反投影    │   │ BSE池化  │   │ 统计累积 │
         └──────────┘   └──────────┘   └──────────┘
                               │
                               ▼
                        ┌─────────────────────┐
                        │  7. 保存记忆文件    │
                        │  save_memory()      │
                        └─────────────────────┘
```

## 关键数据结构

### 1. ExtractionConfig

```python
@dataclass
class ExtractionConfig:
    dataset_path: str
    output_path: str
    n_memory: int = 100
    device: str = "cuda:0"
    voxel_size: float = 0.05
    use_bse: bool = True
    use_otsu: bool = True
    depth_valid_range: Tuple[float, float] = (0.1, 6.0)
    use_model: str = "mapanything"
    ray_pool_strategy: str = "mean"
```

### 2. memory_views元素

```python
{
    'img': torch.Tensor,           # [B, 3, H, W] 图像
    'depth_z': torch.Tensor,       # [B, H, W, 1] 深度
    'camera_poses': torch.Tensor,  # [B, 4, 4] 位姿
    'intrinsics': torch.Tensor,    # [B, 3, 3] 内参
    'data_norm_type': List[str],   # ["dinov2", ...]
    'is_metric_scale': torch.Tensor,  # [B] bool
}
```

### 3. raw_batches元素

```python
{
    'img': torch.Tensor,              # 原始图像
    'depthmap': torch.Tensor,         # 原始深度
    'camera_pose': torch.Tensor,      # 原始位姿
    'camera_intrinsics': torch.Tensor, # 原始内参
    'dataset': str,
    'label': str,
    'instance': str,
    ...  # 其他元数据
}
```

### 4. features_dict

```python
{
    view_idx: {
        'features': List[torch.Tensor],  # 多尺度特征列表
        'grid_H': int,                    # 网格高度
        'grid_W': int,                    # 网格宽度
    }
}
```

### 5. 输出记忆文件

```python
{
    'schema_version': '1.2',
    'points': torch.Tensor,          # [N, 3] 归一化点云
    'features': torch.Tensor,        # [N, C_total] 多尺度拼接特征 (C_total ~ 5000-9000, 非 1024)
    'colors': torch.Tensor,          # [N, 3] 颜色
    'ray_dirs': torch.Tensor,        # [N, 3] 射线方向
    'ray_dirs_dominant': torch.Tensor,  # [N, 3] 主方向
    'ray_dirs_first': torch.Tensor,     # [N, 3] 首射线
    'ray_dirs_all': List[torch.Tensor], # 所有射线
    'plucker_main_rays': torch.Tensor,  # [N, 6] Plücker坐标
    'scene_mean': torch.Tensor,      # [3] 场景均值
    'scene_sigma': float,            # 场景标准差
    'camera_centers': torch.Tensor,  # [M, 3] 相机中心
    'camera_rotations': torch.Tensor, # [M, 3, 3] 旋转矩阵
    'camera_intrinsics': torch.Tensor, # [M, 3, 3] 内参
}
```

## 常见问题与调试

### 1. KeyError: 'depths'

**原因：** 在`unproject_with_grid_features`中使用了错误的key

**解决：**
```python
# 错误
depth = view_data['depths']

# 正确
depth = view_data.get('depth', view_data.get('depthmap', view_data.get('depth_z')))
```

### 2. Depth全为0

**原因：** 深度维度处理错误

**检查：**
```python
# 添加调试
print(f"Depth shape: {depth.shape}, min={depth.min()}, max={depth.max()}")

# 确保正确处理各种格式
if depth.ndim == 4:
    if depth.shape[-1] == 1:
        depth = depth[0, :, :, 0]  # [B, H, W, 1] -> [H, W]
```

### 3. MapAnything特征文件为空

**原因：** 未正确配置特征存储

**解决：**
```python
# 初始化时设置
model.model_config.store_info_sharing_intermediate_features = True

# 推理后从内存获取
stored_features = model.get_info_sharing_intermediate_features()
# 不要从文件加载！
```

### 4. RuntimeError: permute dimension mismatch

**原因：** 深度格式不是4维

**解决：** 使用更健壮的维度处理
```python
if depth.ndim == 4:
    if depth.shape[-1] == 1:
        depth = depth[0, :, :, 0]
    elif depth.shape[1] == 1:
        depth = depth[0, 0, :, :]
elif depth.ndim == 3:
    depth = depth[0]
```

### 5. ValueError: No points accumulated

**原因：**
1. 深度有效范围设置错误
2. 深度数据本身无效

**调试：**
```python
# 检查深度统计
print(f"Depth: min={depth.min()}, max={depth.max()}, "
      f"valid_range={depth_valid_range}")

# 放宽范围
--depth_valid_range 0.1 100.0
```

## 性能优化建议

1. **内存管理**
   - 使用`buffer_on_cpu=True`将缓冲区放在CPU
   - 减少batch_size避免OOM

2. **GPU利用**
   - MapAnything推理时禁用AMP（`use_amp=False`）
   - 使用`memory_efficient_inference=True`

3. **I/O优化**
   - BSE块存储在`/dev/shm`（共享内存）
   - 使用checkpoint支持断点续传

## 与MapAnything对比

| 方面 | MapAnything | 本实现 |
|------|-------------|--------|
| 数据准备 | 分离raw_batches和input_views | 相同 |
| 特征获取 | `get_info_sharing_intermediate_features()` | 相同 |
| 深度访问 | 从raw_batches | 相同 |
| 池化方法 | 简单体素 | BSE双边聚类 |
| 输出格式 | 单一表示 | 多射线策略 |

## 延伸阅读

1. **BSE算法原理**
   - 论文：Bilateral Supervoxel Segmentation
   - 关键：双边空间的定义和聚类

2. **Welford算法**
   - 在线均值/方差计算
   - 数值稳定性优于两遍算法

3. **Plücker坐标**
   - 3D射线的6D表示
   - 用于射线比较和插值

4. **MapAnything架构**
   - 多视图Transformer
   - 中间层特征聚合

## 总结

核心要点：
1. **数据分离**：`raw_batches`（原始数据）vs `memory_views`（模型输入）
2. **特征获取**：从内存获取，不依赖文件
3. **深度处理**：处理各种格式，确保[H, W]形状
4. **两遍处理**：累积统计 -> 归一化
5. **BSE池化**：保留边界的几何聚类

遵循本指南的阅读顺序，你将能够深入理解整个memory extraction管道的设计和实现。
