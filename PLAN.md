# 迁移计划：将 map-anything 的 Memory 压缩/融合 + 两阶段训练 迁入 DINO ACE

## 现状分析

### ace_depth 现有 DINO ACE 流程
- `ace_network_dinov2.py`: DINOv2Encoder + Head (回归头)，Regressor 封装
- `trainer_dinov2.py`: TrainerACEDINOv2 — 单阶段训练：fill buffer → train head
- `train_ace_dinov2.py` / `test_ace_dinov2.py`: CLI 入口
- `dataset_dinov2.py`: RGB 数据集，14x 下采样
- `ace_loss.py`: ReproLoss（reprojection loss）

### 现有 LMC 文件（需要重写）
- `trainer_dinov2_lmc.py`: 当前是 trainer_dinov2.py 的拷贝，无 LMC 逻辑
- `train_ace_dinov2_lmc.py` / `test_ace_dinov2_lmc.py`: 引用不存在的 tasks/ace/ 模块
- `ace_compressor.py`: 简单线性 autoencoder，不是 map-anything 的 GeoLMC
- `tasks/ace/` 目录不存在

### map-anything 要迁入的核心
- `compressor.py`: GeoLMC — 几何感知 Transformer 压缩器（FPS采样 + 多模式注意力）
- `fusion.py`: LMCFeatureFusion — 交叉注意力融合（memory token → query features）
- `regression_head.py`: ACEHead_Homogeneous_Mean（带 scale token 调制的回归头）
- 两阶段多迭代训练策略（小步快跑：小 buffer + 多轮次）

## 迁移原则

1. **最小侵入**：不修改现有 DINO ACE 核心文件（ace_network_dinov2.py, dataset_dinov2.py, ace_loss.py）
2. **退化开关**：所有 LMC 功能通过开关控制，关闭时完全退化为原版 DINO ACE
3. **不迁移已有功能**：buffer 收集、数据集处理、loss 计算沿用 DINO ACE 现有逻辑
4. **Memory 只读**：ace_depth 通过路径读取 map-anything 已保存的 memory，不实现提取逻辑

## 文件变更清单

### 新建文件（3 个核心模块）

#### 1. `ace_compressor.py` — 重写（替换现有简单版本）
- 从 map-anything `compressor.py` 复刻 GeoLMC
- 包含：FourierPositionEncoding, farthest_point_sampling, GeoLMC
- 包含：LocalGeometricBias, AdaptiveGeometricBias, 各种 Attention 模块
- 去除对 mapanything 包的依赖，改为自包含
- 保留所有 4 种模式：global, local, hierarchical, learned

#### 2. `ace_fusion.py` — 新建
- 从 map-anything `fusion.py` 复刻 LMCFeatureFusion
- 包含：LMCFusionBlock, LMCFeatureFusion
- 引用 ace_compressor.py 中的 FourierPositionEncoding
- 保留所有模式支持

#### 3. `ace_head_lmc.py` — 新建
- 从 map-anything `regression_head.py` 复刻 ACEHead_Homogeneous_Mean
- 包含：PointwiseBlock, ScaleModulator, ACEHead_Homogeneous_Mean
- 支持 scale_token 调制（退化开关：use_scale_token=False 时等价于原版 Head）

### 修改文件（3 个）

#### 4. `trainer_dinov2_lmc.py` — 重写
基于现有 `trainer_dinov2.py`，添加：
- **Memory 加载**：从给定路径读取预保存的 memory_dict（pooled_points, pooled_features, scene_center 等）
- **GeoLMC 压缩**：在 buffer 填充前，用 GeoLMC 压缩 memory → latent tokens
- **LMCFeatureFusion 融合**：在 encoder 提取特征后，将 query features 与 latent tokens 融合
- **两阶段多迭代训练**：
  - 外循环：多轮迭代（默认 15 轮）
  - 每轮 Stage 1：用当前 compressor 压缩 memory，重新填充 buffer（融合后的特征）
  - 每轮 Stage 2：从 buffer 训练 head
  - 最后一轮使用更大 buffer
- **Head 重置策略**：支持 first_only / every / output_only / none
- **退化开关**：
  - `--use_lmc False`：跳过所有 LMC 逻辑，完全退化为原版 trainer_dinov2.py 的行为
  - `--memory_path`：memory 文件路径（None 时自动退化）

#### 5. `train_ace_dinov2_lmc.py` — 重写
添加 LMC 相关 CLI 参数：
- `--use_lmc`: 总开关（默认 False，退化为原版）
- `--memory_path`: 预保存 memory 路径
- `--lmc_mode`: global / local / hierarchical / learned
- `--num_latent_tokens`: 潜在 token 数量
- `--num_attn_layers`: 注意力层数
- `--use_scale_token`: 是否使用 scale token
- `--lmc_iterations`: 两阶段迭代轮次
- `--buffer_size_final`: 最后一轮 buffer 大小
- `--head_reset_strategy`: Head 重置策略
- `--lmc_train_steps`: S1 每轮步数
- `--head_lr_multiplier_s2`: S2 阶段 Head 学习率倍率

#### 6. `test_ace_dinov2_lmc.py` — 重写
- 从 checkpoint 恢复 LMC 配置
- 加载 compressor + fusion 权重
- 推理流程：encode → compress memory → fuse → head → DSAC* RANSAC
- 退化：checkpoint 中无 LMC 配置时，退化为原版测试流程

## 数据流（LMC 开启时）

```
训练阶段（每轮迭代）：
  Stage 1 - Buffer 填充:
    1. 加载预保存 memory_dict（从 --memory_path）
    2. GeoLMC 压缩 memory → (latent_z, latent_p)  [可训练]
    3. 遍历训练图像：
       a. DINOv2 encoder 提取 query_features (B,1024,H/14,W/14)
       b. LMCFeatureFusion 融合 query_features + (latent_z, latent_p) → fused_features
       c. 采样 fused_features 存入 buffer
    4. 训练 compressor（S1 步数）

  Stage 2 - Head 训练:
    1. 从 buffer 采样 batch
    2. Head 预测 scene coordinates
    3. Reprojection loss（沿用 ace_loss.py）
    4. 更新 head（+ compressor 如果未冻结）

测试阶段：
  1. 加载 memory_dict
  2. GeoLMC 压缩 → latent tokens
  3. 对每张测试图：
     a. DINOv2 encode → query features
     b. Fusion → fused features
     c. Head → scene coordinates
     d. DSAC* RANSAC → 6DoF pose
```

## 退化验证

当 `--use_lmc False`（或 `--memory_path` 未提供）时：
- 跳过 memory 加载、compressor、fusion
- buffer 填充直接使用 encoder 原始特征（与 trainer_dinov2.py 完全一致）
- 单阶段训练（与 trainer_dinov2.py 完全一致）
- checkpoint 格式兼容原版（只保存 head state_dict）

## Checkpoint 格式（LMC 开启时）

```python
{
    'head_state_dict': ...,           # 回归头权重
    'compressor_state_dict': ...,     # GeoLMC 权重
    'fusion_state_dict': ...,         # Fusion 权重
    'mean_cam_center': ...,           # 场景中心
    'lmc_config': {                   # LMC 配置（用于测试时恢复）
        'use_lmc': True,
        'lmc_mode': 'global',
        'num_latent_tokens': 64,
        'num_attn_layers': 2,
        'use_scale_token': True,
        'compress_dim': 1024,
        'memory_path': '...',
    },
}
```
