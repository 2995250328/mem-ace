# BSE Memory Extraction

基于双边超体素提取 (Bilateral Supervoxel Extraction, BSE) 增强的 ACE DINOv2 LMC 记忆提取管道。

## 概述

本模块使用以下技术从多视角 RGB-D 数据中提取压缩场景记忆：
- **BSE 池化**：边界保持的双边聚类，配合 Otsu 自适应阈值
- **Welford 归一化**：数值稳定的全局度量归一化
- **多策略射线编码**：保留多视角几何信息

## 快速开始

<!-- Updated 2026-03-31: added ACE dataset loader examples -->

### 使用 Shell 脚本（推荐）

```bash
cd /home/xwh/project/ace_depth

# === WAI 数据集（默认） ===

# 7-Scenes chess，20 视角（默认配置，WAI 加载器）
bash ace_dinov2_lmc/memory_extraction/extract_memory.sh

# 通过环境变量覆盖
SCENE_TRAIN=fire_train N_VIEWS=40 bash ace_dinov2_lmc/memory_extraction/extract_memory.sh

# Indoor6 场景
DATASET_TYPE=indoor6 SCENE_TRAIN=scene2a_train N_VIEWS=40 bash ace_dinov2_lmc/memory_extraction/extract_memory.sh

# 严格 FPS 协议（默认）：实际送入 MapAnything 的 40 张图与 FPS list 一一对应
WAI_VIEW_MODE=fps_flat DATASET_TYPE=indoor6 SCENE_TRAIN=scene2a_train N_VIEWS=40 \
  bash ace_dinov2_lmc/memory_extraction/extract_memory.sh

# 复现 map-anything 原版 fps_memory.sh：第一个 FPS anchor 扩展成 40 个 covisibility views
WAI_VIEW_MODE=original_multiview DATASET_TYPE=indoor6 SCENE_TRAIN=scene2a_train N_VIEWS=40 \
  bash ace_dinov2_lmc/memory_extraction/extract_memory.sh

# 共视性加权 FPS：尽量保持 FPS 覆盖，同时避免低共视大 baseline 视图破坏 MapAnything 推理
WAI_VIEW_MODE=covis_fps COVIS_FPS_ALPHA=1.0 COVIS_FPS_TAU=-1.0 \
  DATASET_TYPE=indoor6 SCENE_TRAIN=scene2a_train N_VIEWS=40 \
  bash ace_dinov2_lmc/memory_extraction/extract_memory.sh

# === ACE 数据集加载器（CamLocDatasetDINOv2） ===
# 适用于任何含 rgb/、poses/、calibration/ 目录的场景（7-Scenes / Indoor6 / 自定义）
#
# 路径说明：
#   DATASET_ROOT 指向 scene 根目录（如 pgt_7scenes_chess/），
#   脚本会自动拼接 SCENE_TRAIN（如 train）形成最终 dataset_path（前提是根目录下没有 rgb/ 子目录）。
#   若 DATASET_ROOT 下已有 rgb/，则不做拼接。

# 7-Scenes chess — 脚本自动拼接 /train（pgt_7scenes_chess/ 下无 rgb/，但 pgt_7scenes_chess/train/ 下有）
DATASET_LOADER=ace DATASET_ROOT=/home/xwh/data/7Scenes/pgt_7scenes_chess \
  SCENE_TRAIN=train N_VIEWS=20 GPU_ID=0 \
  bash ace_dinov2_lmc/memory_extraction/extract_memory.sh

# 7-Scenes heads — 同上，自动拼接
DATASET_LOADER=ace DATASET_ROOT=/home/xwh/data/7Scenes/pgt_7scenes_heads \
  SCENE_TRAIN=train N_VIEWS=20 GPU_ID=0 \
  bash ace_dinov2_lmc/memory_extraction/extract_memory.sh

# Indoor6 scene2a — 同样格式（scene2a/ 下有 train/rgb/）
DATASET_LOADER=ace DATASET_ROOT=/home/xwh/data/indoor6_ace/scene2a \
  SCENE_TRAIN=train N_VIEWS=40 GPU_ID=0 \
  bash ace_dinov2_lmc/memory_extraction/extract_memory.sh

# === 其他选项 ===

# 消融实验：不同 voxel size
VOXEL_SIZE=0.10 N_VIEWS=40 bash ace_dinov2_lmc/memory_extraction/extract_memory.sh

# 使用 DINOv2 fallback（默认是 MapAnything）
USE_MODEL=dinov2 bash ace_dinov2_lmc/memory_extraction/extract_memory.sh

# 关闭 SOR（默认已开启）
ENABLE_SOR=false bash ace_dinov2_lmc/memory_extraction/extract_memory.sh
```

### 直接调用 Python

**WAI 数据集：**

```bash
python -m ace_dinov2_lmc.memory_extraction.run_memory_extraction \
    /home/xwh/data/mapanything-dataset/wai_data/7scenes \
    /path/to/out/memory.pt \
    --n_memory 20 \
    --dataset_type 7scenes \
    --dataset_loader wai \
    --patch_depth_sampling nearest \
    --enable_sor \
    --device cuda:0
```

**ACE 数据集（CamLocDatasetDINOv2）：**

`dataset_path` 直接指向包含 `rgb/`、`poses/`、`depth/`、`calibration/` 的目录。注意与 WAI 不同，ACE 不需要 `--scene_name`。

```bash
# dataset_path 指向含 rgb/ 的目录（7-Scenes: pgt_7scenes_chess/train/）
python -m ace_dinov2_lmc.memory_extraction.run_memory_extraction \
    /home/xwh/data/7Scenes/pgt_7scenes_chess/train \
    /path/to/out/memory.pt \
    --n_memory 20 \
    --dataset_loader ace \
    --enable_sor \
    --device cuda:0

# Indoor6: 同样直接指向 train/ 目录
python -m ace_dinov2_lmc.memory_extraction.run_memory_extraction \
    /home/xwh/data/indoor6_ace/scene2a/train \
    /path/to/out/memory.pt \
    --n_memory 40 \
    --dataset_loader ace \
    --enable_sor \
    --depth_valid_range 0.02 100.0 \
    --device cuda:0
```

### 两种数据集加载器的区别

| 特性 | WAI (`--dataset_loader wai`) | ACE (`--dataset_loader ace`) |
|------|---------------------------|---------------------------|
| 数据格式 | MapAnything WAI 目录结构 | ACE CamLocDataset 格式 |
| `dataset_path` | 数据集根目录（其下含各 scene 子目录） | **直接指向含 `rgb/`、`poses/` 的目录** |
| 场景指定 | `--scene_name chess_train` | 无需额外指定 |
| 深度来源 | WAI 格式 depth map（COLMAP sparse / GT） | ACE 格式 depth map |
| 归一化 | MapAnything 风格 | ImageNet（DINOv2） |
| Shell 自动拼接 | 不拼接（WAI 的 DATASET_ROOT 即数据集根） | 若 `DATASET_ROOT/` 下无 `rgb/`，自动拼接 `/${SCENE_TRAIN}` |
| 适用数据 | `/home/xwh/data/mapanything-dataset/wai_data/` | `/home/xwh/data/7Scenes/pgt_7scenes_*/`、`/home/xwh/data/indoor6_ace/*/` |

**ACE 路径拼接规则**：Shell 脚本在 `DATASET_LOADER=ace` 时会检查 `DATASET_ROOT/rgb/` 是否存在：
- 若存在 → 直接用 `DATASET_ROOT` 作为 `dataset_path`
- 若不存在 → 拼接 `DATASET_ROOT/${SCENE_TRAIN}`（如 `pgt_7scenes_chess/train`）作为 `dataset_path`

## 与 map-anything `fps_memory.sh` 的对齐

- **WAI 视图加载协议**：`WAI_VIEW_MODE=fps_flat`（默认）表示每个 FPS index 只加载 1 张图，因此实际送入 MapAnything 的 `input_views` 与 `select_optimal_memory_indices()` 返回的 FPS list 一一对应；`WAI_VIEW_MODE=original_multiview` 表示复现原版 `fps_memory.sh` 行为，即 WAI `dataset[idx]` 在 `num_views=N_VIEWS` 下会把第一个 FPS anchor 扩展成 `N_VIEWS` 个 covisibility views，后续 FPS indices 可能不会被实际消费；`WAI_VIEW_MODE=covis_fps` 表示读取 WAI `scene_root/covisibility/v0/*.npy` 的真实共视矩阵，按 `score = normalized_min_dist_to_selected * (eps + normalized_max_covis_to_selected)^alpha` 做共视性加权 FPS。三者是不同实验协议，PoseEval / pooled memory 点数不可直接当作同一选帧策略比较。
- **推荐用法**：需要严格验证 FPS 选出的 40 张图时使用默认 `fps_flat`；需要复现 map-anything 原版 pooled memory / PoseEval 时使用 `WAI_VIEW_MODE=original_multiview`；需要折中覆盖性和 MapAnything 多视图稳定性时使用 `WAI_VIEW_MODE=covis_fps`，先用 `COVIS_FPS_ALPHA=1.0 COVIS_FPS_TAU=-1.0` 做 soft-score 消融。
- **实际输入检查**：每次 run 会在日志中打印 `[Data] Actual model input views:`，并在输出目录写出 `loaded_model_input_views.json`，字段包括 `selected_memory_indices`、`wai_view_mode`、`dataset_num_views`、每个 view 的 `actual_flat_idx`、`fps_idx_at_view_position`、`matches_fps_at_view_position` 和 `source`。`model_input_vis/view_XX_model_input.png` 左上角也会标注 `src`、`actual_idx`、`fps_idx`、`outer/inner`，可直接和 FPS list 对照。
- **网格深度采样**：与 `mapanything/tasks/run_memory_extraction.py` 中 `generate_patch_point_cloud` 一致，支持 `nearest` / `median` / `nearest_valid`。Indoor6 等 **稀疏深度** 必须用 **`nearest_valid`**（在网格中心邻域内取有效深度的中位数），否则容易出现「整图有深度、网格上几乎无有效点」。
- **Shell 默认**：`DATASET_TYPE=indoor6` 时脚本默认 `PATCH_DEPTH_SAMPLING=nearest_valid`；`7scenes` 默认 `nearest`。
- **深度校验图**：选帧并加载视图后，会在本次 run 目录下写出 `depth_validation/` 与 `model_input_vis/`（行为对齐 map-anything；若本机可导入 `mapanything.tasks.run_memory_extraction`，则优先复用其 `_save_raw_rgb` / `_save_depth_on_rgb_vis` 等；稀疏深度用 scatter，极稠密时自动改用向量化 turbo 上色，避免卡顿）。
- **深度张量布局**：WAI 常见 `[H,W,1]`（HWC），反投影前会规范为 `[H,W]`，避免误当成 `[B,H,W]` 导致只取一行深度。

## 输出目录结构

默认根目录为 **本脚本所在目录**下的 `04_evaluation/memory_extract/`（即  
`ace_dinov2_lmc/memory_extraction/04_evaluation/memory_extract/`），可通过环境变量 `OUTPUT_ROOT` 覆盖。

```
memory_extract/
├── chess/
│   ├── 20v_v0.05_bilinear_l2/
│   │   └── 20260328_143000/
│   │       ├── memory_bse.pt                    # 主输出
│   │       ├── extraction_config.json           # 配置快照（含 patch_depth_sampling 等）
│   │       ├── extraction_log.txt               # 运行日志
│   │       ├── loaded_model_input_views.json    # 实际进入 MapAnything 推理的 view 文件名 / FPS 对照
│   │       ├── depth_validation/                # 深度与 RGB 对齐检查（与 fps_memory 一致）
│   │       │   ├── view_00_rgb_raw.png
│   │       │   ├── view_XX_depth_vis.png
│   │       │   ├── view_XX_depth_on_rgb.png
│   │       │   └── view_00_raw_depth_pointcloud.ply  # 若 Open3D 可用且 view0 有有效点
│   │       └── model_input_vis/                  # 送入网络的归一化图像可视化（左上角标注实际文件名/FPS index）
│   │           └── view_XX_model_input.png
│   └── 20v_v0.10_bilinear_l2/            # 消融：不同 voxel size
│       └── 20260328_150000/
│           └── ...
├── fire/
│   └── 40v_v0.05_bilinear_l2/
│       └── ...
└── scene2a/
    └── 40v_v0.05_bilinear_l2/
        └── ...
```

**命名规则**：`<N>v_<voxel_size>_<提取模式>[_sor][_l2]`

| 组成部分 | 含义 |
|----------|------|
| `20v` | 记忆视角数量 |
| `v0.05` | 体素大小 |
| `bilinear` / `patch` | 特征提取模式 |
| `sor` | 启用 SOR 过滤 |
| `l2` | 特征 L2 归一化 |

## Shell 脚本参数

所有参数通过环境变量设置，可修改脚本默认值或命令行覆盖。

### 数据集与场景

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `DATASET_TYPE` | `7scenes` | 数据集类型：`7scenes`、`indoor6`、`custom` |
| `SCENE_TRAIN` | `chess_train` | 提取用场景名 |
| `SCENE_TEST` | 自动推导 | 测试场景名（从 `SCENE_TRAIN` 自动推导） |
| `DATASET_ROOT` | 自动 | 数据集根目录（根据 `DATASET_TYPE` 自动设置） |
| `DATASET_LOADER` | `wai` | 数据集加载器：`wai`=WAI 格式；`ace`=ACE CamLocDatasetDINOv2 |
| `WAI_VIEW_MODE` | `fps_flat` | WAI 选帧协议：`fps_flat`=实际输入严格等于 FPS list；`original_multiview`=复现原版 `fps_memory.sh` 的 covisibility 多视图 sample；`covis_fps`=共视性加权 FPS |
| `COVIS_FPS_ALPHA` | `1.0` | `covis_fps` 的共视性权重；`0` 退化成纯 FPS，越大越偏向共视性 |
| `COVIS_FPS_EPS` | `1e-6` | `covis_fps` soft score 的 epsilon |
| `COVIS_FPS_TAU` | `-1.0` | `covis_fps` 硬共视阈值；`<0` 表示关闭硬阈值，仅使用 soft score |
| `COVIS_MAX_DIST_TO_REF` | `-1.0` | `covis_fps` 局部性约束：候选帧到 reference 帧 camera center 的最大距离（米）；`<0` 关闭 |
| `N_VIEWS` | `20` | 记忆视角数量 |
| `GPU_ID` | `3` | GPU 编号（`CUDA_VISIBLE_DEVICES`） |
| `DATASET_LOADER` | `wai` | 数据集加载器：`wai`（MapAnything WAI）或 `ace`（ACE CamLocDatasetDINOv2） |

**数据集路径自动推导：**

| DATASET_TYPE | `DATASET_PATH`（传给 Python 的根路径） | 说明 |
|--------------|----------------------------------------|------|
| `7scenes` | `/home/xwh/data/mapanything-dataset/wai_data/7scenes`（或 `DATASET_ROOT`） | `SCENE_TRAIN=chess_train` 等由数据集类解析 |
| `indoor6` | `/home/xwh/data/mapanything-dataset/wai_data/indoor6`（或 `DATASET_ROOT`） | WAI Indoor6 ROOT；`SCENE_TRAIN=scene2a_train` 指定场景划分 |
| `custom` | `$DATASET_ROOT` | 用户自定义 |

**深度范围自动适配：**

| DATASET_TYPE | DEPTH_MIN | DEPTH_MAX |
|--------------|-----------|-----------|
| `7scenes` | 0.1 | 6.0 |
| `indoor6` | 0.1 | 100.0（COLMAP / GT 稀疏深度，避免远距被裁） |

**网格深度采样（与 `fps_memory.sh` 的 `PATCH_DEPTH_SAMPLING` 一致）：**

| 变量 | 默认（由 `DATASET_TYPE` 决定） | 说明 |
|------|-------------------------------|------|
| `PATCH_DEPTH_SAMPLING` | indoor6 → `nearest_valid`；否则 → `nearest` | `nearest`：单像素；`median`：3×3 中位数；`nearest_valid`：半径 10 邻域内有效深度中位数（**稀疏深度推荐**） |

### BSE 池化

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `VOXEL_SIZE` | `0.05` | 体素网格大小（米） |
| `USE_OTSU` | `true` | 使用 Otsu 自适应阈值 |
| `ENABLE_SOR` | `false` | 启用统计离群点移除 |
| `RAY_POOL_STRATEGY` | `mean` | 策略：`mean`、`dominant`、`first`、`all` |

### 特征提取

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `USE_MODEL` | `mapanything` | 特征提取器：`mapanything`（默认）或 `dinov2` |
| `USE_PATCH_BASED` | `false` | Patch 方式（网格）vs 双线性上采样 |
| `USE_L2_NORMALIZATION` | `true` | 拼接前对特征做 L2 归一化 |
| `DINOV2_CHECKPOINT` | `/home/xwh/data/checkpoints/dinov2_vitl14_pretrain.pth` | DINOv2 权重（fallback 时使用） |

### 输出

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `OUTPUT_ROOT` | `$(dirname extract_memory.sh)/04_evaluation/memory_extract` | 输出根目录，即 **`memory_extraction/04_evaluation/memory_extract`** |

## Python CLI 参数

直接调用 `run_memory_extraction.py` 时：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `dataset_path` | 位置参数 | 场景数据集路径 |
| `output_path` | 位置参数 | 输出 .pt 文件路径 |
| `--n_memory` | 100 | 记忆视角数量 |
| `--dataset_type` | `auto` | `auto`、`7scenes`、`indoor6`、`custom` |
| `--device` | `cuda:0` | GPU 设备 |
| `--voxel_size` | 0.05 | BSE 体素大小 |
| `--ray_pool_strategy` | `mean` | 射线池化策略 |
| `--depth_valid_range` | 0.1 6.0 | 有效深度范围（最小 最大）；Indoor6 建议命令行与脚本一致设为 `0.1 100` |
| `--patch_depth_sampling` | `nearest_valid` | `nearest` / `median` / `nearest_valid`，与 map-anything 网格深度语义一致 |
| `--enable_sor` | 开关 | 启用 SOR 过滤 |
| `--sor_k` | `20` | SOR 近邻个数 k |
| `--sor_std_ratio` | `2.0` | SOR 标准差倍数阈值（越小越严格） |
| `--dataset_loader` | `wai` | 数据集加载器：`wai`（MapAnything WAI 格式，需 `--scene_name`）或 `ace`（ACE CamLocDataset 格式，`dataset_path` 直接指向含 `rgb/` 的目录） |
| `--wai_view_mode` | `fps_flat` | WAI 视图加载协议：`fps_flat` 严格使用 FPS list；`original_multiview` 复现原版 `fps_memory.sh` 多视图 sample；`covis_fps` 共视性加权 FPS |
| `--covis_fps_alpha` | `1.0` | `covis_fps` 的共视性权重 |
| `--covis_fps_eps` | `1e-6` | `covis_fps` soft score epsilon |
| `--covis_fps_tau` | `-1.0` | `covis_fps` 硬共视阈值，`<0` 关闭 |
| `--covis_max_dist_to_ref` | `-1.0` | `covis_fps` 局部性约束：候选帧到 reference 帧 camera center 的最大距离（米）；`<0` 关闭 |
| `--dinov2_checkpoint` | /home/xwh/data/checkpoints/... | DINOv2 权重（fallback 时使用） |
| `--use_model` | `mapanything` | 特征提取器：`mapanything`（默认）或 `dinov2` |

## 输出格式

输出的 `.pt` 文件（schema v1.2）包含：

```python
{
    'schema_version': '1.2',

    # 点级别数据
    'points': P_norm,                  # [N, 3] float32, 归一化坐标
    'features': F_bse,                 # [N, C] float16, 多尺度 DINOv2 特征
    'colors': C_bse,                   # [N, 3] float32, RGB [0,1]
    'cluster_sizes': sizes,            # [N] int64, 每聚类点数

    # 射线方向表示（点级别，池化后）
    'ray_dirs': D_mean,                # [N, 3] float32, 按策略池化
    'ray_dirs_mean': D_mean,           # [N, 3] float32, 平均+归一化
    'ray_dirs_dominant': D_dominant,   # [N, 3] float32, 最高相似度
    'ray_dirs_first': D_first,         # [N, 3] float32, 聚类首条射线
    'plucker_rays': plucker,           # [N, 6] float32, (方向, 矩)

    # 视图级别相机信息（M = 记忆视图数量）
    'view_camera_centers': centers,    # [M, 3] float32
    'view_camera_rotations': rots,     # [M, 3, 3] float32
    'view_camera_intrinsics': Ks,      # [M, 3, 3] float32
    'view_plucker_main_rays': plucker, # [M, 6] float32

    # 归一化统计
    'mu': mu_scene,                    # [3] float32, 场景均值
    'sigma': sigma_scene,              # 标量 float32, 场景标准差
}
```

## 射线方向策略

| 策略 | 说明 | 适用场景 |
|------|------|----------|
| `mean` | 平均 + L2 归一化 | 默认基线 |
| `dominant` | 选择特征相似度最高的射线 | 保留最强视角 |
| `first` | 使用聚类中的第一条射线 | 保留原始方向 |
| `all` | 保留聚类中的所有射线 | 学习压缩时保留完整信息 |

### 普吕克编码

6D 射线表示：`plucker = (direction, moment)`，其中 `moment = camera_center x direction`。用于视角依赖推理和多视角一致性检查。

## 架构

```
memory_extraction/
├── __init__.py               # 模块导出
├── bse_pooling.py            # BSE 算法 + 射线策略
├── welford_meter.py          # 流式归一化
├── run_memory_extraction.py  # 主提取脚本
└── extract_memory.sh         # Shell 启动器（环境变量驱动）
```

## 常用工作流

### 批量提取所有 7-Scenes 场景

```bash
for scene in chess fire heads office pumpkin redkitchen stairs; do
    SCENE_TRAIN=${scene}_train bash ace_dinov2_lmc/memory_extraction/extract_memory.sh
done
```

### 体素大小消融实验

```bash
for vs in 0.03 0.05 0.08 0.10; do
    VOXEL_SIZE=$vs N_VIEWS=40 bash ace_dinov2_lmc/memory_extraction/extract_memory.sh
done
```

### Indoor6 场景提取

```bash
for scene in scene1 scene2a scene3 scene4a scene5 scene6; do
    DATASET_TYPE=indoor6 SCENE_TRAIN=${scene}_train N_VIEWS=40 \
        bash ace_dinov2_lmc/memory_extraction/extract_memory.sh
done
```

### Reference-consistent 的 C1 验证

当你要在多个 Indoor6 场景上做同口径验证时，可以固定这套提取配置：
`C1 + anchor_support + reference gate + post-repair`。这和当前 `scene2a`
以及 `scene3` 的验证方式一致。

```bash
GPU_ID=0 \
ACE_DATA_ROOT=/home/xwh/data \
DATASET_ROOT=/home/xwh/data/mapanything-dataset/wai_data/indoor6 \
DATASET_TYPE=indoor6 \
SCENE_TRAIN=scene2a_train \
N_VIEWS=40 \
OUTPUT_ROOT=/home/xwh/project/ace_depth/ace_dinov2_lmc/memory_extraction/04_evaluation/memory_extract \
CONTRACT_MODE=C1 \
WAI_VIEW_MODE=anchor_support \
WAI_TRANSFORM=imgnorm \
WAI_AUG_CROP=0 \
POOL_MODE=bse \
VOXEL_SIZE=0.05 \
USE_OTSU=true \
UNIMODAL_THRESHOLD=0.02 \
PREPOOL_MODE=per_view \
ENABLE_SOR=true \
GLOBAL_MERGE=true \
USE_L2_NORMALIZATION=true \
ASB_ADAPTIVE=true \
ASB_CANDIDATE_POOL_RATIO=1.0 \
ENABLE_REFERENCE_POLICY_GATE=true \
REFERENCE_POLICY_TOP_M=4 \
REFERENCE_POLICY_LIGHT_MIN_SELECTED_LINKS=2 \
REFERENCE_POLICY_PROBE_Q90_M=0.18 \
REFERENCE_POLICY_PROBE_MAX_M=0.25 \
ASB_POST_REPAIR=true \
ASB_POST_REPAIR_MAX_SWAPS=8 \
ASB_POST_REPAIR_TAIL_PERCENTILE=95.0 \
ASB_POST_REPAIR_MIN_TAIL_IMPROVEMENT_M=0.02 \
ASB_POST_REPAIR_CLUSTER_TOP_K=3 \
bash ace_dinov2_lmc/memory_extraction/extract_memory.sh
```

`scene3` 只要把 `SCENE_TRAIN=scene3_train`，并在需要并行跑时把
`GPU_ID=1` 即可。

提取完成后，训练命令模板如下：

```bash
ACE_DATA_ROOT=/home/xwh/data \
python ace_dinov2_lmc/train_ace_dinov2_lmc.py \
  /home/xwh/data/indoor6_ace/scene2a \
  scene2a_c1_<timestamp>.pt \
  --train_preset memory_compare_ace_g_v1 \
  --data_backend ace \
  --device cuda:0 \
  --post_train_eval_device cuda:0 \
  --use_lmc True \
  --memory_path /home/xwh/project/ace_depth/ace_dinov2_lmc/memory_extraction/04_evaluation/memory_extract/scene2a/40v_v0.05_bilinear_bse_ut0.02_noaug_asb_adaptive_pool1.0_rgate4_prepair8_sor_gm_l2/<timestamp>/memory_bse.pt \
  --lmc_mode global \
  --experiment_subdir memory_pooled_vs_asb_c1 \
  --buffer_on_cpu False \
  --batch_size 10240
```

```bash
ACE_DATA_ROOT=/home/xwh/data \
python ace_dinov2_lmc/train_ace_dinov2_lmc.py \
  /home/xwh/data/indoor6_ace/scene3 \
  scene3_c1_<timestamp>.pt \
  --train_preset memory_compare_ace_g_v1 \
  --data_backend ace \
  --device cuda:1 \
  --post_train_eval_device cuda:1 \
  --use_lmc True \
  --memory_path /home/xwh/project/ace_depth/ace_dinov2_lmc/memory_extraction/04_evaluation/memory_extract/scene3/40v_v0.05_bilinear_bse_ut0.02_noaug_asb_adaptive_pool1.0_rgate4_prepair8_sor_gm_l2/<timestamp>/memory_bse.pt \
  --lmc_mode global \
  --experiment_subdir memory_pooled_vs_asb_c1 \
  --buffer_on_cpu False \
  --batch_size 10240
```

关键参数含义：

- `CONTRACT_MODE=C1`：把主监督切到 `points_ref` / `points_ref_norm`。
- `WAI_VIEW_MODE=anchor_support`：使用 anchor/support 选帧，而不是普通 FPS。
- `ENABLE_REFERENCE_POLICY_GATE=true`：先做 top-M reference gate，再继续提取。
- `ASB_POST_REPAIR=true`：selected set 填满后启用 coverage-tail repair。
- `buffer_on_cpu=False`：训练时把 buffer 留在 GPU。
- `batch_size=10240`：提高 S2 训练 batch size。
- `buffer_batch_size` 保持预设默认 `1`，不额外放大。

## 故障排除

| 问题 | 解决方案 |
|------|----------|
| `ValueError: No points accumulated` / 网格上有效点极少 | Indoor6 确认使用 **`PATCH_DEPTH_SAMPLING=nearest_valid`**（脚本默认已设）；检查 `--depth_valid_range` 是否与数据一致（如 `0.1 100`） |
| 日志里深度 `shape` 异常（如 `[518,1]`） | 已修复 HWC/`[H,W,1]` 解析；请使用当前版 `run_memory_extraction.py` |
| `features_flat` 与 `valid_mask` 设备不一致 | 已修复：索引特征时会与 CPU/GPU 对齐 |
| 深度校验阶段长时间无输出 | 极稠密有效深度时已改用向量化上色；仍慢时可检查磁盘与 Matplotlib 后端 |
| 内存不足 | 减少 `N_VIEWS`，增大 `VOXEL_SIZE`，或使用更大显存的 GPU |
| 数据集找不到 | 检查 `DATASET_ROOT` 和 `SCENE_TRAIN`；脚本会打印推导路径 |
| `mapanything` 导入错误 | 脚本自动将 `/home/xwh/data/map-anything` 加入 PYTHONPATH；可视化会回退到内置实现 |
| 处理速度慢 | 增大 `VOXEL_SIZE`，或设置 `ENABLE_SOR=false` |

## 依赖

- PyTorch >= 2.0
- MapAnything — 默认多尺度特征提取器（DINOv2 + AAT + DPT）
- DINOv2 (facebookresearch/dinov2) — fallback 特征提取器
- Matplotlib、PIL — 深度/RGB 校验图导出
- Open3D（可选 — 用于 `view_00_raw_depth_pointcloud.ply`）
