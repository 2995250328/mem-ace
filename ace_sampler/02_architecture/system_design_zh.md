# 系统架构设计

## 1. 目录结构

实现代码直接放在 `ace_sampler/` 目录下（而非 `03_implementation/` 子目录，
因为该模块在研究工作流规范确立之前已创建）。

```
ace_sampler/                    # Python module + idea folder
├── __init__.py                 # Exports: SamplerNet, SamplerTrainer, fill_buffer_with_sampler
├── model.py                    # NEW: SamplerNet (~80K params)
├── uncertainty.py              # NEW: UncertaintyHead with MC Dropout
├── trainer.py                  # NEW: Phase 1 SamplerTrainer
├── buffer_sampler.py           # NEW: Phase 2 fill_buffer_with_sampler
├── options.py                  # NEW: add_sampler_train_args, add_sampler_buffer_args
train_ace_sampler.py            # NEW: Phase 1 entry script (project root)
```

无需修改、直接复用：
```
ace_network.py                  # REUSE: Regressor (frozen in Phase 1)
dataset_origin.py               # REUSE: CamLocDataset
ace_util.py                     # REUSE: get_pixel_grid
ace_trainer.py                  # REUSE: Phase 2 calls fill_buffer_with_sampler as drop-in
```

## 2. 模块职责

| 文件 | 职责 |
|---|---|
| `model.py` | `SamplerNet`：深度可分离卷积网络，从编码器特征图预测置信度图 |
| `uncertainty.py` | `UncertaintyHead`：轻量级场景坐标预测器，使用 Dropout2d 进行 MC 方差估计 |
| `trainer.py` | `SamplerTrainer`：第一阶段训练循环——计算误差/方差图，通过 MSE 训练 SamplerNet |
| `buffer_sampler.py` | `fill_buffer_with_sampler`：第二阶段——top-k 加随机采样，缓冲区结构与原版完全一致 |
| `options.py` | 第一阶段（`add_sampler_train_args`）和第二阶段（`add_sampler_buffer_args`）的参数组 |
| `train_ace_sampler.py` | CLI 入口：解析参数 → `SamplerTrainer(options).train()` |

## 3. 张量流图（接口契约）

### 第一阶段 — SamplerTrainer

```
Input image:        (1, 1, H, W)   float16/32  cuda   — grayscale, from CamLocDataset
                         ↓ Regressor.get_features()
Encoder features:   (1, 512, Hf, Wf)  float16  cuda   — Hf=H/8, Wf=W/8
                         ↓ Regressor.get_scene_coordinates()
Scene coords:       (1, 3, Hf, Wf)  float32   cuda   — 3D scene coordinates
                         ↓ project via gt_pose_inv + intrinsics
Reprojection error: (1, 1, Hf, Wf)  float32   cuda   — per-pixel error in pixels
                         ↓ (optional) UncertaintyHead.mc_coordinate_samples() T=10
MC variance:        (1, 1, Hf, Wf)  float32   cuda   — reprojection variance in px²
                         ↓ exp(-(α·error + β·variance))
Confidence target:  (1, 1, Hf, Wf)  float32   cuda   — ∈ [0, 1]
                         ↓ SamplerNet(features)
Confidence pred:    (1, 1, Hf, Wf)  float32   cuda   — ∈ [0, 1]
                         ↓ MSE loss
Loss:               scalar          float32   cuda
```

### 第二阶段 — fill_buffer_with_sampler

```
Input image:        (1, 1, H, W)   float16/32  cuda
                         ↓ Regressor.get_features()
Encoder features:   (1, 512, Hf, Wf)  float16  cuda
                         ↓ SamplerNet (frozen)
Confidence map:     (Hf, Wf)       float32   cuda   — masked by image_mask
                         ↓ top-k (Part A) + multinomial (Part B)
Pixel coords:       (N, 2)         float32   cuda   — in original image space
                         ↓ grid_sample on features
Sampled features:   (N, 512)       float16   cuda   — stored in buffer
Buffer entry:       features(N,512) + target_px(N,2) + gt_poses_inv(N,3,4)
                    + intrinsics(N,3,3) + intrinsics_inv(N,3,3)
```

### 关键维度说明
- `H, W`：原始图像高度/宽度（默认 480 × ~640）
- `Hf = H/8, Wf = W/8`：特征图分辨率（ACE FCN 编码器，8× 下采样）
- `N = samples_per_image`（默认每张图像 1024 个采样点）
- 缓冲区总量：`training_buffer_size` 条目（默认 8M）

## 4. 复用决策

| 模块 | 决策 | 来源 |
|---|---|---|
| ACE 编码器 + 回归头 | 冻结复用 | `ace_network.py:Regressor` |
| 数据集加载器 | 原样复用 | `dataset_origin.py:CamLocDataset` |
| 像素网格 | 原样复用 | `ace_util.py:get_pixel_grid` |
| 缓冲区结构 | 完全一致复用 | `ace_trainer.py` buffer dict keys |
| SamplerNet | 新增 | `ace_sampler/model.py` |
| UncertaintyHead | 新增 | `ace_sampler/uncertainty.py` |
| 第一阶段训练循环 | 新增 | `ace_sampler/trainer.py` |
| 第二阶段缓冲区填充 | 新增（即插即用） | `ace_sampler/buffer_sampler.py` |

## 5. 未验证假设与硬件约束

- **假设**：冻结回归头的重投影误差能提供有意义的训练信号。
  要求回归头至少经过部分训练（而非随机初始化）。
- **假设**：SamplerNet 在第一阶段学到的置信度可迁移至第二阶段的回归头。
  通过第二阶段的 Part B 随机采样部分缓解此问题。
- **硬件**：第一阶段在单 GPU 上运行，无需 DDP（网络参数量约 80K）。
- **内存**：第一阶段峰值显存 ≈ ACE 推理时的显存占用（batch_size=1 约 2GB）。
  MC Dropout（T=10）仅增加约 10 倍 UncertaintyHead 前向传播开销（可忽略不计）。
