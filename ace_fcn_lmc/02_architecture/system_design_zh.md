# 系统设计：ACE-FCN-LMC

## 概述

该系统是现有 DINOv2-LMC 技术栈的薄扩展。唯一的结构变化是将 backbone 从 DINOv2 替换为 ACE FCN encoder。所有训练编排、buffer 管理、损失计算和结果管理均原封不动地继承。

## 模块依赖图

```
train_ace_lmc.py
    ├── options_ace_lmc.py          （CLI 参数）
    ├── utils_lmc.py                （共享工具）
    ├── result_manager.py           （结果管理）
    └── trainer_ace_fcn.py
            ├── TrainerACEFCN       → 继承 TrainerACEDINOv2（trainer_dinov2.py）
            └── TrainerACEFCNLMC    → 继承 TrainerACEDINOv2LMC（trainer_dinov2_lmc.py）
                    └── _create_regressor() → RegressorACE（ace_network_ace.py）
                                                └── ACEEncoder → Encoder（ace_network.py）
                                                └── Head（ace_network_dinov2.py）

test_ace_lmc.py
    ├── options_ace_lmc.py
    └── run_evaluation_lmc()
            └── RegressorACE（ace_network_ace.py）
            └── dsacstar（RANSAC 位姿估计）
```

## 数据流

### 训练（Vanilla 模式）

```
CamLocDataset（灰度图，480×480）
    → ACEEncoder.forward(x)          # (B,1,H,W) → (B,512,H/8,W/8)
    → TrainingBuffer.fill()          # 采样特征 + 坐标
    → Head.forward(features)         # (B,512,H/8,W/8) → (B,3,H/8,W/8)
    → ReproLoss                      # 重投影损失
    → optimizer.step()
```

### 训练（LMC 模式）

```
Stage 1（S1）：记忆对齐
    PooledMemory.load()              # (N,3) 3D 点
    CamLocDataset → ACEEncoder       # 在线或 buffer
    Head.forward() → 坐标
    L_lmc = coord_alignment_loss(坐标, 记忆点)
    optimizer_s1.step()

Stage 2（S2）：重投影精化
    TrainingBuffer.fill()            # 用当前 encoder 重新填充
    Head.forward() → 坐标
    L_repro = ReproLoss(坐标, pose_gt, K)
    optimizer_s2.step()

重复 S1+S2 共 lmc_iterations 轮
```

### 推理

```
查询图像（灰度图，H×W）
    → ACEEncoder.forward()           # (1,512,H/8,W/8)
    → Head.forward()                 # (1,3,H/8,W/8) 场景坐标
    → dsacstar.forward()             # RANSAC PnP → T ∈ SE(3)
    → 与真实位姿比较误差
```

## 关键设计决策

### 1. 薄子类模式

`TrainerACEFCN` 和 `TrainerACEFCNLMC` 仅覆盖 `_create_regressor()`：

```python
class TrainerACEFCN(TrainerACEDINOv2):
    def _create_regressor(self):
        return RegressorACE.create_from_encoder(
            encoder_path=self.options.encoder_path,
            mean=self.dataset.mean_cam_center,
            num_head_blocks=self.options.num_head_blocks,
            num_encoder_features=512,
            freeze_backbone=self.options.freeze_backbone,
        )
```

所有 buffer 管理、损失计算、学习率调度、检查点保存和结果管理均从 DINOv2 训练器原封不动地继承。

### 2. ACEEncoder 中的 RGB→灰度转换

原始 ACE FCN encoder 期望灰度输入。数据集流水线提供 RGB 图像（为与共享 DINOv2 数据集加载器兼容）。ACEEncoder 在内部处理转换：

```python
# 在 ACEEncoder.forward() 中：
gray = (x * self.rgb_weights).sum(dim=1, keepdim=True)  # (B,3,H,W) → (B,1,H,W)
return self.encoder(gray)
```

这避免了修改数据集加载器，并保持接口一致。

### 3. 特征维度适配

`ace_network_dinov2.py` 中的 Head 网络由 `num_encoder_features` 参数化。对于 ACE FCN，设置为 512（DINOv2 为 1024）。Head 架构自动适配。

### 4. 无 Patch-Size 约束

与 DINOv2（要求图像尺寸为 14 的倍数）不同，FCN encoder 支持任意分辨率。默认为 480×480。

## 输出目录结构

```
output/
└── {数据集}/
    └── {场景}/
        ├── ace_fcn_vanilla/
        │   └── {时间戳}_{配置标签}/
        │       ├── run_metadata.json
        │       ├── training_summary.json
        │       ├── best_*.pt
        │       └── eval_results/
        ├── ace_fcn_lmc_iter/          # lmc_flow=iterative
        │   └── {时间戳}_{配置标签}/
        └── ace_fcn_lmc_aceg/          # lmc_flow=ace_g
            └── {时间戳}_{配置标签}/
```

## 配置参数

完整参数列表见 `options_ace_lmc.py`。与 DINOv2 选项的关键差异：

| 参数 | ACE FCN | DINOv2 |
|---|---|---|
| `--encoder_path` | 必需（FCN 权重） | `--dinov2_path` |
| `--image_resolution` | 默认 480，无约束 | 默认 518，必须是 14 的倍数 |
| `--num_encoder_features` | 512 | 1024 |
| `--freeze_backbone` | 默认 True | 默认 True |
