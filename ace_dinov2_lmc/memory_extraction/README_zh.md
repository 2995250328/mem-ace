# BSE Memory Extraction

基于双边超体素提取 (Bilateral Supervoxel Extraction, BSE) 增强的 ACE DINOv2 LMC 记忆提取管道。

## 概述

本模块使用以下技术从多视角 RGB-D 数据中提取压缩场景记忆：
- **BSE 池化**：边界保持的双边聚类，配合 Otsu 自适应阈值
- **Welford 归一化**：数值稳定的全局度量归一化
- **多策略射线编码**：保留多视角几何信息

## 快速开始

### 基本用法

```bash
# 激活环境
conda activate mapanything

# 运行提取
python -m ace_dinov2_lmc.memory_extraction.run_memory_extraction \
    /path/to/dataset \
    output/memory.pt \
    --n_memory 100 \
    --device cuda:0
```

### 使用 Shell 脚本

```bash
cd ace_dinov2_lmc/memory_extraction

# 基本提取
./extract_memory.sh /path/to/dataset output/memory.pt 100 0

# 自定义 BSE 参数
VOXEL_SIZE=0.03 USE_OTSU=true ./extract_memory.sh /path/to/dataset output/memory.pt 100 0
```

## 参数说明

### 核心参数

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `dataset_path` | 必需 | 场景数据集路径 |
| `output_path` | 必需 | 输出 .pt 文件路径 |
| `--n_memory` | 100 | 选择的记忆视图数量 |
| `--device` | cuda:0 | GPU 设备 |

### BSE 池化参数

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `--voxel_size` | 0.05 | 粗糙体素网格大小 |
| `--use_otsu` | True | 使用 Otsu 自适应阈值 |
| `--otsu_bins` | 256 | Otsu 直方图 bin 数 |
| `--unimodal_threshold` | 0.02 | 相似度标准差小于阈值时跳过分割 |

### 射线方向参数

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `--ray_pool_strategy` | mean | 射线方向池化策略 |
| `--save_all_ray_strategies` | True | 保存所有策略结果用于比较 |

#### 射线池化策略

| 策略 | 描述 |
|------|------|
| `mean` | 平均 + L2 归一化（默认基线） |
| `dominant` | 选择特征相似度最高的射线 |
| `first` | 使用聚类中的第一条射线（保留原始方向） |
| `all` | 保留聚类中的所有射线 |

### 深度参数

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `--depth_valid_range` | 0.1 6.0 | 有效深度范围（最小，最大） |
| `--enable_sor` | False | 启用统计离群点移除 |

### 模型参数

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `--model_str` | None | MapAnything 模型架构字符串 |
| `--model_config` | None | 模型配置文件路径 |
| `--model_checkpoint` | None | 可选的模型检查点路径 |
| `--dinov2_checkpoint` | /data/xwh/checkpoints/dinov2_vitl14_pretrain.pth | DINOv2 权重（未指定模型时使用） |

## 输出格式

输出的 `.pt` 文件包含：

```python
{
    # Schema
    'schema_version': '1.2',

    # 点级别数据
    'points': P_norm,                  # [N, 3] float32, 归一化坐标
    'features': F_bse,                 # [N, C] float16, 多尺度 DINOv2 特征（~5000 维）
    'colors': C_bse,                   # [N, 3] float32, RGB [0,1]
    'cluster_sizes': sizes,            # [N] int64, 每聚类点数

    # 射线方向表示（点级别，池化后）
    'ray_dirs': D_mean,                # [N, 3] float32, 按策略池化
    'ray_dirs_mean': D_mean,           # [N, 3] float32, 平均+归一化
    'ray_dirs_dominant': D_dominant,   # [N, 3] float32, 最高相似度
    'ray_dirs_first': D_first,         # [N, 3] float32, 聚类首条射线
    'plucker_rays': plucker,           # [N, 6] float32, (方向, 矩)

    # 视图级别相机信息（M = 记忆视图数量）
    'view_camera_centers': centers,    # [M, 3] float32, 相机位置
    'view_camera_rotations': rots,     # [M, 3, 3] float32, 旋转矩阵
    'view_camera_intrinsics': Ks,      # [M, 3, 3] float32, 内参矩阵
    'view_plucker_main_rays': plucker, # [M, 6] float32, (方向, 矩)

    # 归一化统计
    'mu': mu_scene,                    # [3] float32, 场景均值
    'sigma': sigma_scene,              # 标量 float32, 场景标准差
}
```

## 射线方向处理

### 为什么需要多种策略？

从多个视角观测同一个物体时，射线方向可能差异很大。简单的平均可能会丢失重要的多视角信息。本模块提供多种策略：

1. **Mean + Normalize**：基线方法，可能会模糊多视角信息
2. **Dominant Ray**：保留每个聚类中最具代表性的射线
3. **First Ray**：简单，保留原始方向不做处理
4. **Plücker Encoding**：6D 射线表示，用于下游处理

### 普吕克射线编码

普吕克坐标提供 3D 直线的唯一 6D 表示：

```
plucker = (direction, moment)
moment = camera_center × direction
```

这种编码对以下场景有用：
- 下游 Transformer 中的视角依赖推理
- 多视角一致性检查
- 后续阶段的学习压缩

## 架构

```
memory_extraction/
├── __init__.py           # 模块导出
├── bse_pooling.py        # BSE 算法 + 射线策略
├── welford_meter.py      # 流式归一化
├── run_memory_extraction.py  # 主提取脚本
└── extract_memory.sh     # Shell 启动器
```

### BSEPooler

核心池化算法：

1. **粗糙体素哈希**：将点量化到体素网格
2. **聚类均值特征**：计算每个体素的平均特征
3. **Otsu 阈值**：自适应相似度阈值
4. **二分分割**：将离群点从主聚类中分离
5. **细粒度池化**：使用多种射线策略池化所有属性

### WelfordNormalizer

流式统计计算：

- FP64 中数值稳定的均值/方差
- 增量更新，内存高效
- 两遍归一化：收集统计 → 归一化

## 示例

### 从 7-Scenes 提取

```bash
python -m ace_dinov2_lmc.memory_extraction.run_memory_extraction \
    /data/xwh/7Scenes/pgt_7scenes_chess \
    output/chess_memory.pt \
    --n_memory 100 \
    --device cuda:0
```

### 启用 SOR 过滤

```bash
python -m ace_dinov2_lmc.memory_extraction.run_memory_extraction \
    /path/to/dataset \
    output/memory.pt \
    --n_memory 100 \
    --enable_sor \
    --device cuda:0
```

### 自定义体素大小

```bash
python -m ace_dinov2_lmc.memory_extraction.run_memory_extraction \
    /path/to/dataset \
    output/memory.pt \
    --n_memory 100 \
    --voxel_size 0.03 \
    --device cuda:0
```

## 故障排除

### 内存不足

```bash
# 减少记忆视图数量
--n_memory 50

# 增大体素大小（更粗糙的池化）
--voxel_size 0.08
```

### 处理速度慢

```bash
# 禁用 SOR 过滤
--enable_sor false

# 使用更大的体素
--voxel_size 0.06
```

### 导入错误

```bash
# 确保激活 mapanything 环境
conda activate mapanything

# 检查 Python 路径
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

## 依赖

- PyTorch >= 2.0
- MapAnything（主要特征提取模型）
- DINOv2 (facebookresearch/dinov2) - 未指定模型时的回退选项
- Open3D（可选，用于可视化）
