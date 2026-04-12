# DINOv2 Integration for ACE

本文档说明如何使用DINOv2作为ACE的特征提取backbone。

## 概述

已将原始ACE的FCN encoder替换为DINOv2 ViT-L/14，主要修改包括：

### 关键变化

| 组件 | 原始ACE | DINOv2版本 |
|------|---------|-----------|
| Encoder | FCN (卷积网络) | DINOv2 ViT-L/14 (Transformer) |
| 输入通道 | 1 (灰度图) | 3 (RGB) |
| 特征维度 | 512 | 1024 |
| 下采样率 | 8x | 14x |
| Patch Size | N/A | 14x14 |
| 默认图像尺寸 | 480 | 518 (37×14) |

## 文件结构

新增文件：
```
ace_network_dinov2.py      # DINOv2网络架构
dataset_dinov2.py          # 适配DINOv2的数据集
trainer_dinov2.py          # 训练器
train_ace_dinov2.py        # 训练脚本
test_ace_dinov2.py         # 测试脚本
```

## 环境要求

### 1. 安装依赖

```bash
# 基础环境（已有）
conda activate ace

# 安装DINOv2依赖（如果需要）
pip install timm
```

### 2. 准备DINOv2权重

确保DINOv2预训练权重位于：
```
/mnt/storage/xwh/checkpoints/dinov2_vitl14_pretrain.pth
```

如果权重在其他位置，使用`--dinov2_path`参数指定。

## 使用方法

### 训练

基础训练命令：

```bash
./train_ace_dinov2.py datasets/7scenes_chess output/chess_dinov2.pt
```

完整参数示例：

```bash
./train_ace_dinov2.py \
    datasets/7scenes_chess \
    output/chess_dinov2.pt \
    --dinov2_path /mnt/storage/xwh/checkpoints/dinov2_vitl14_pretrain.pth \
    --freeze_backbone True \
    --device cuda:3 \
    --image_resolution 518 \
    --epochs 16 \
    --batch_size 512 \
    --num_head_blocks 1 \
    --learning_rate_max 0.001 \
    --learning_rate_min 0.0001
```

### 测试

基础测试命令：

```bash
./test_ace_dinov2.py datasets/7scenes_chess output/chess_dinov2.pt
```

完整参数示例：

```bash
./test_ace_dinov2.py \
    datasets/7scenes_chess \
    output/chess_dinov2.pt \
    --dinov2_path /mnt/storage/xwh/checkpoints/dinov2_vitl14_pretrain.pth \
    --device cuda:3 \
    --image_resolution 518 \
    --hypotheses 64 \
    --threshold 10 \
    --session dinov2_test
```

## 重要参数说明

### 图像尺寸 (--image_resolution)

**必须是14的倍数**，因为DINOv2使用14×14的patch size。

推荐尺寸：
- 518 (37×14) - 默认，接近原始480
- 532 (38×14)
- 504 (36×14)
- 560 (40×14)

如果指定的尺寸不是14的倍数，会自动调整到最近的倍数。

### Backbone冻结 (--freeze_backbone)

- `True` (默认): 冻结DINOv2参数，只训练head
  - 优点：训练快，显存占用少，防止过拟合
  - 缺点：可能无法充分适应特定场景

- `False`: 微调整个网络
  - 优点：可能获得更好的性能
  - 缺点：训练慢，显存占用大，容易过拟合

### Head深度 (--num_head_blocks)

控制场景特定head的深度：
- `1` (默认): 轻量级head，约4MB
- `2-4`: 更深的head，可能提升性能但增加模型大小

## 与原始ACE的差异

### 1. 图像预处理

**原始ACE:**
```python
# 灰度图 + 自定义归一化
transforms.Grayscale()
transforms.Normalize(mean=[0.4], std=[0.25])
```

**DINOv2版本:**
```python
# RGB + ImageNet归一化
transforms.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
```

### 2. 输出分辨率

由于下采样率从8x变为14x：
- 输入: 518×518
- 输出: 37×37 (518/14)

相机内参会自动调整以匹配新的输出分辨率。

### 3. 特征维度

Head网络的输入从512维变为1024维，但内部处理保持512维不变。

## 性能考虑

### 显存占用

DINOv2 ViT-L/14较大：
- 冻结backbone: ~6GB (batch_size=512)
- 微调backbone: ~12GB (batch_size=256)

如果显存不足，减小batch_size：
```bash
--batch_size 256  # 或更小
```

### 训练时间

- 冻结backbone: 约5-10分钟/场景
- 微调backbone: 约15-30分钟/场景

### 推理速度

DINOv2比FCN慢：
- 原始ACE: ~30 FPS
- DINOv2版本: ~10-15 FPS

## 故障排查

### 1. 图像尺寸错误

```
AssertionError: Input size (480, 640) must be multiple of patch_size (14)
```

**解决**: 使用`--image_resolution 518`或其他14的倍数。

### 2. DINOv2加载失败

```
RuntimeError: Error loading DINOv2 weights
```

**解决**:
- 检查权重路径是否正确
- 确保权重文件完整
- 尝试重新下载DINOv2权重

### 3. 显存不足

```
RuntimeError: CUDA out of memory
```

**解决**:
- 减小batch_size: `--batch_size 256`
- 使用更小的图像: `--image_resolution 504`
- 确保freeze_backbone=True

### 4. torch.hub加载失败

如果网络问题导致torch.hub无法下载DINOv2模型定义：

**解决**: 手动下载DINOv2代码到本地，修改`ace_network_dinov2.py`中的加载方式。

## 预期性能

基于DINOv2的强大特征提取能力，预期：

**优势:**
- 更好的泛化能力
- 对光照变化更鲁棒
- 可能在大场景中表现更好

**劣势:**
- 推理速度较慢
- 显存占用较大
- 可能需要更多训练数据

## 示例工作流

完整的训练和测试流程：

```bash
# 1. 训练
./train_ace_dinov2.py \
    datasets/7scenes_chess \
    output/chess_dinov2.pt \
    --device cuda:3 \
    --epochs 16

# 2. 测试
./test_ace_dinov2.py \
    datasets/7scenes_chess \
    output/chess_dinov2.pt \
    --device cuda:3

# 3. 在其他场景上测试
./test_ace_dinov2.py \
    datasets/7scenes_fire \
    output/chess_dinov2.pt \
    --device cuda:3
```

## 进一步优化

可能的改进方向：

1. **LoRA微调**: 使用LoRA而不是完全冻结/微调
2. **多尺度特征**: 使用DINOv2的多层特征
3. **知识蒸馏**: 从大模型蒸馏到小模型
4. **混合架构**: 结合CNN和Transformer的优势

## 技术细节

### DINOv2特征提取

```python
# DINOv2输出patch tokens (不含CLS token)
features_dict = self.dinov2.forward_features(x)
patch_tokens = features_dict['x_norm_patchtokens']  # [B, N, 1024]

# 重塑为空间特征图
# [B, N, 1024] -> [B, 1024, H//14, W//14]
features = patch_tokens.transpose(1, 2).reshape(B, 1024, H//14, W//14)
```

### 相机内参适配

由于下采样率变化，内参矩阵需要相应调整：

```python
# 原始: 8x下采样
# 新: 14x下采样
# 焦距和主点都需要按比例调整
```

数据集加载器会自动处理这个转换。

## 参考

- DINOv2论文: https://arxiv.org/abs/2304.07193
- ACE论文: https://arxiv.org/abs/2305.14059
- DINOv2 GitHub: https://github.com/facebookresearch/dinov2
