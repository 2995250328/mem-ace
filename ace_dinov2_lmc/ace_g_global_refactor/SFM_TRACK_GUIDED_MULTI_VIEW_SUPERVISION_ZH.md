# COLMAP-Keyframe Inter-Frame ACE Loss

Updated: 2026-06-15

定位：本文档记录监督侧创新 **COLMAP-Keyframe Inter-Frame ACE Loss**。它借鉴 Xu 等 2024 / SeqACE 的 sequence-based mapping 中的 keyframe channel 与 cross ACE loss，但不采用其重新匹配 pipeline，也不采用其测试时 sequence relocalization。

核心结论：

1. 本方向是 **训练期辅助监督**，推理期不需要 SfM、COLMAP、稀疏深度、track、keyframe 或序列输入。
2. 主线不是简单的稀疏 3D 点 L2 / XYZ loss，而是 **SeqACE-style inter-frame ACE reprojection loss**。
3. 本项目已有基于稀疏深度图的采样逻辑，因此不重新设计采样器；只为已有 sparse-depth-guided sampled patches 附加 COLMAP keyframe channel。
4. COLMAP / SfM 的核心作用不是直接提供主监督 `P ≈ X_colmap`，而是提供全局 point3D track identity：
   - anchor sparse-depth pixel 属于哪个 `point3D_id`；
   - 同一 `point3D_id` 在哪些 train keyframes 中被观测；
   - target keyframe pixel、pose、intrinsics 是什么。
5. COLMAP 3D point `X_m^w` 只用于 keyframe 选择、几何过滤、invalid fallback 和 diagnostic；不作为 V1 主 loss。
6. 必须显式处理 patch-support alignment：只有 COLMAP anchor observation 与当前训练 patch / feature cell 的代表位置足够对齐时，才启用 inter-frame ACE loss。neighbor-expanded sparse-depth samples 不能默认继承 seed point 的 keyframe channel。

一句话方法：

> 在已有 sparse-depth-guided sampling 选出的 anchor patch 中，仅对与 COLMAP observation 经过 patch-support alignment 检查的子集附加 SeqACE-style keyframe channel，并对同一个预测 scene coordinate 同时计算当前帧 ACE self-loss 和关键帧 inter-frame ACE loss。

---

## 1. 为什么要重写这个设计

旧 STGS 设计把第一阶段设为：

```text
L_xyz = || P_pred(anchor pixel) - X_colmap ||
```

这个方向太朴素，容易被理解成“用额外 COLMAP 3D 点做稀疏坐标回归监督”。它没有充分利用 SeqACE 的关键思想：

```text
同一个 anchor prediction 同时解释当前帧观测和 keyframe 观测。
```

SeqACE 的核心不是简单 3D 点监督，而是：

```text
P_i^w = network(anchor image, anchor pixel)

self ACE loss:
  project P_i^w to anchor frame -> anchor pixel

cross ACE loss:
  project the same P_i^w to keyframe -> matched keyframe pixel
```

本项目比 SeqACE 更适合这个思想，因为 memory 构建阶段已经有 COLMAP / SfM 结果。COLMAP track 提供的是全局几何验证过的多帧对应，而不是单纯 pairwise feature matches。

因此，本设计的主线应该是：

```text
SeqACE keyframe channel
+ COLMAP point3D track identity
+ existing sparse-depth-guided sampling
= COLMAP-Keyframe Inter-Frame ACE Loss
```

---

## 2. 与 SeqACE 的关键对应关系

本地参考论文：

```text
/home/xwh/project/ace_depth/papers/Xu 等 - 2024 - An Efficient Scene Coordinate Encoding and Relocalization Method.pdf
```

SeqACE sequence-based mapping 的相关字段可概括为：

```text
target_px       # 当前帧 patch pixel
gt_poses_inv    # 当前帧 pose
intrinsics      # 当前帧 K

target_px2      # keyframe matched pixel
gt_poses_inv2   # keyframe pose
intrinsics2     # keyframe K
track_flag      # 是否存在有效 cross match
```

训练时对同一个预测 `P_i^w` 计算：

```text
L_self  = ACEReproLoss(P_i^w, target_px,  pose_i, K_i)
L_cross = ACEReproLoss(P_i^w, target_px2, pose_j, K_j) if track_flag
```

本项目的替换关系：

| SeqACE | 本项目 |
|---|---|
| SuperPoint / LightGlue pairwise match | COLMAP / SfM global `point3D_id` track |
| matched keyframe pixel `target_px2` | same COLMAP point 在 keyframe 中的 observation pixel |
| feature matching confidence / F-matrix RANSAC | COLMAP reprojection error、track length、parallax、positive depth |
| keypoint/saliency patch selection | 已有 sparse-depth-guided sampling |
| invalid proxy from fixed target depth | optional COLMAP XYZ invalid fallback |
| sequence-based test mode | 不采用，推理保持单图 |

最重要的区别：

```text
SeqACE 需要离线 feature matching 生成 keyframe channel；
本项目可以从 COLMAP global tracks 直接生成 keyframe channel。
```

---

## 3. 当前代码中已有的采样基础

当前代码已经支持基于稀疏深度 / 有效 scene-coordinate 区域的 buffer sampling，不需要为本方向重新设计采样。

相关参数位于：

```text
ace_dinov2_lmc/options_dinov2_lmc.py
```

已有深度类型：

```text
--c1_aux_depth_kind gt_depth|colmap_depth|sparse_depth|sparse_depth_sampling_sp|sparse_depth_mapanything
```

已有采样相关参数包括：

```text
--buffer_sample_valid_coords
--buffer_valid_coord_sample_ratio
--buffer_valid_coord_neighbor_radius
--buffer_valid_coord_neighbor_mode
```

相关实现位于：

```text
ace_dinov2_lmc/trainer_dinov2_lmc.py
```

核心函数：

```text
_expand_valid_coord_sampling_mask(...)
_sample_buffer_indices(...)
```

它们已经能做到：

```text
1. 从 sparse depth / valid coord mask 得到有效 patch；
2. 可选扩展到邻域 patch；
3. 按比例优先采样这些有效位置；
4. 剩余样本仍从普通 image mask 中随机采样。
```

因此，本文档不再设计新的 observation sampler、image-level dataloader 或 `max_obs_per_image` 采样流程。

本设计只做一件事：

> 对现有 sparse-depth-guided sampling 选中的 anchor patch，查表补充 keyframe channel 字段；但只有通过 patch-support alignment 检查的样本才真正启用 inter-frame ACE loss。

---

## 4. Patch-support alignment 风险与修正

SeqACE-style cross loss 有一个容易被忽略的前提：

```text
训练 patch 代表的位置 == 跟踪 / 匹配 keypoint 代表的位置
```

如果这个前提不成立，cross loss 会把一个 patch center 的 scene coordinate 监督到另一个 keypoint 的对应像素上，产生系统性偏差。

### 4.1 偏差来源

#### 4.1.1 Anchor patch center 与 COLMAP observation 不对齐

SCR 训练通常在 feature grid 上采样：

```text
feature cell / patch -> one predicted scene coordinate
```

这个 prediction 通常代表 feature cell center 或当前代码定义的 `target_px`。COLMAP observation 是 subpixel keypoint：

```text
u_colmap = (x, y)
```

如果：

```text
||u_colmap - u_patch_center|| > 0
```

那么 `P_patch^w` 代表的可能是 patch center，而不是 COLMAP keypoint。此时把 `P_patch^w` 投到 keyframe observation `u_j` 会产生 biased supervision。

偏差量级不可忽视。近似：

```text
3D offset ≈ depth / focal * pixel_offset
```

例如 depth=5m、focal=500px、pixel_offset=4px 时，3D 偏差约 4cm；DINO stride 14 下若 offset 接近 7px，偏差可达约 7cm，足以影响 5cm / 2cm 阈值。

#### 4.1.2 Matcher / COLMAP keypoint support 与训练 patch support 不一致

SuperPoint/LightGlue/COLMAP keypoint 是局部关键点；ACE-FCN / DINO / GLACE 的 training patch 是 stride-based feature cell。即使 keypoint 落在某个 feature cell 中，它也不一定代表该 cell 的中心或完整 receptive field。

因此不能把：

```text
keypoint match -> sampled training patch
```

无条件视为同一个监督对象。

#### 4.1.3 Neighbor-expanded sparse-depth samples 不能继承 seed track

当前 `_expand_valid_coord_sampling_mask(...)` 会把 sparse seed 扩展到邻域 patch 以改善采样覆盖。这对普通训练采样有用，但对 inter-frame ACE loss 危险。

必须区分：

```text
valid-for-sampling:
  sparse seed + neighbor-expanded ROI

valid-for-inter-frame-loss:
  true sparse seed only
  + has point3D_id
  + anchor observation aligned to patch representative point
  + has valid keyframe target
```

neighbor-expanded samples 可以继续参与普通 ACE self-loss，但不能默认继承 seed point 的 COLMAP keyframe channel。

### 4.2 对齐检查

对每个候选 anchor observation，统一转换到训练模型使用的像素坐标系：

```text
u_colmap_original -> u_colmap_model
u_patch_center_model = feature-cell representative point used by target_px
```

计算：

```text
alignment_error_px = ||u_colmap_model - u_patch_center_model||
```

启用条件：

```text
anchor_is_sparse_seed = True
alignment_error_px <= align_threshold_px
```

建议初值：

```text
align_threshold_px = min(0.25 * feature_stride, 2.0 px)
```

更宽松版本可用：

```text
align_threshold_px = 0.5 * feature_stride
```

但高精度定位第一版应尽量保守。

### 4.3 对齐权重

除了 hard threshold，还建议对 cross loss 加 soft weight：

```text
w_align = exp(- alignment_error_px^2 / (2 * sigma_align^2))
```

最终：

```text
L_inter = w_align * track_flag * ACEReproLoss(project_j(P_i^w), u_j)
```

如果 alignment error 超阈值：

```text
track_flag = 0
```

### 4.4 Target 侧是否需要 patch 对齐

V1 不要求 target observation 落在 target training patch center，因为 target image 不 forward，target side 只是 2D projection target。只要 target observation 是同一 COLMAP `point3D_id` 的几何验证观测，就可以作为 `u_j`。

真正需要严格对齐的是 anchor 侧：

```text
P_i^w 是否真的代表 u_colmap_i 这个点？
```

### 4.5 相对 SeqACE 的改进点

SeqACE 原版 keyframe loss 隐含 matched patch 与 training patch 对齐。对 stride-based SCR backbones，这可能引入 patch-support bias。

本设计应明确作为改进：

> 不盲目把 keyframe match 附加到训练 patch，而是利用 COLMAP observation 与训练 feature cell 的显式对齐检查，只在 anchor-side spatial support 一致时启用 inter-frame ACE loss，并按 alignment uncertainty 降权。

---

## 5. 方法定义

### 5.1 Anchor sample

当前训练 buffer 中已有 anchor sample：

```text
image i
anchor pixel u_i
gt pose/intrinsics: T_i, K_i
network predicted scene coordinate: P_i^w
```

当前 ACE self-loss 已经约束：

```text
project(P_i^w, T_i, K_i) -> u_i
```

### 5.2 COLMAP keyframe channel

如果 anchor pixel `u_i` 来自真实 sparse seed、与当前训练 patch 代表位置通过 patch-support alignment 检查，并且能找到对应的 COLMAP `point3D_id = m`，则使用同一 track 的另一个 train observation 构造 keyframe channel：

```text
target image j
target pixel u_j
pose/intrinsics: T_j, K_j
alignment_weight = exp(-alignment_error_px^2 / (2 sigma_align^2))
track_flag = 1
```

如果没有有效 target observation：

```text
track_flag = 0
```

### 5.3 Inter-frame ACE loss

对同一个 anchor prediction `P_i^w`：

```text
u_hat_j = project(P_i^w, T_j, K_j)
```

计算：

```text
L_inter = ACEReproLoss(u_hat_j, u_j)
```

总 loss：

```text
L = L_self + lambda_if * track_flag * alignment_weight * L_inter
```

这里 `L_self` 仍是当前 ACE / LMC 主 loss。`L_inter` 只是附加的帧间 ACE 几何约束。

### 5.4 为什么它能补单帧 reprojection 的弱点

单帧 reprojection 只要求：

```text
P_i^w lies on anchor camera ray
```

帧间 ACE loss 要求：

```text
P_i^w lies on anchor ray
and
P_i^w also reprojects to target keyframe observation
```

这等价于用 COLMAP track 提供的另一个视角形成三角化式约束，可以加强沿视线方向的深度 / metric scale 监督。

---

## 6. COLMAP XYZ 的正确角色

COLMAP 3D point `X_m^w` 不应作为 V1 主监督。

不推荐主线：

```text
L_xyz = || P_i^w - X_m^w ||
```

原因：

1. 这会把方法降级成 sparse 3D coordinate regression；
2. 容易被质疑使用额外 3D 点监督；
3. 不能体现 SeqACE 的 inter-frame ACE loss 思想；
4. 可能过拟合 sparse keypoint 区域。

推荐用途：

### 6.1 Keyframe selection

用 `X_m^w` 计算 anchor 与 target 之间的 parallax：

```text
ray_i = normalize(X_m^w - C_i)
ray_j = normalize(X_m^w - C_j)
parallax = arccos(ray_i dot ray_j)
```

选择合适 keyframe：

```text
parallax_min <= parallax <= parallax_max
```

### 6.2 Geometric filtering

过滤：

```text
negative depth
out-of-bound target pixel
high COLMAP reprojection error
short track
train/test contamination
bad image-name mapping
```

### 6.3 Invalid fallback

SeqACE 对 invalid prediction 使用 proxy 3D target。本项目可以可选使用 COLMAP point 作为 invalid fallback：

```text
if prediction invalid and track_flag:
    L_invalid += lambda_invalid_colmap * SmoothL1(P_i^w - X_m^w)
```

这只是 fallback，不是主监督。

### 6.4 Diagnostics

记录但不一定反传：

```text
||P_i^w - X_m^w||
inter-frame reprojection error
self reprojection error
parallax distribution
track length distribution
```

---

## 7. Keyframe channel sidecar，而不是新 sampler

由于采样已经存在，本方向只需要一个 sidecar，用于把 anchor sparse-depth pixel 映射到 keyframe channel。

建议格式：

```text
<scene>/train/colmap_keyframe_channel_v1/
  manifest.json
  keyframe_channel.npz
```

### 7.1 `manifest.json`

必需字段：

```text
schema_version = colmap_keyframe_channel_v1
scene_name
split = train
coordinate_space = ace_world
source_type = known_pose_colmap | colmap | nvm
source_model
sparse_depth_kind = sparse_depth_sampling_sp | sparse_depth | colmap_depth
image_root
num_images
num_anchor_observations
num_keyframe_channels
image_names
parallax_min_deg
parallax_max_deg
max_reprojection_error_px
min_track_length
feature_stride
align_threshold_px
sigma_align_px
```

强约束：

```text
1. 只允许 train split；
2. 不包含 test image；
3. image_names 必须与 ACE/WAI train images 可审计对齐；
4. 坐标系必须与训练 pose/world 坐标一致；
5. sidecar 不负责采样，只负责 keyframe channel lookup。
```

### 7.2 `keyframe_channel.npz`

建议扁平字段：

```text
anchor_image_idx      int32    [N]
anchor_xy_original    float32  [N, 2]
anchor_xy_model       float32  [N, 2]
anchor_feature_yx     int32    [N, 2]
anchor_patch_center_xy float32 [N, 2]
anchor_alignment_error_px float32 [N]
anchor_is_sparse_seed bool     [N]
alignment_weight      float32  [N]

point3D_id            int64    [N]
target_image_idx      int32    [N]
target_xy_original    float32  [N, 2]
target_xy_model       float32  [N, 2]

target_track_flag     bool     [N]
track_flag            bool     [N]  # target_track_flag & anchor alignment gate
parallax_deg          float32  [N]
colmap_reproj_error   float32  [N]
track_length          int32    [N]

optional_track_xyz_world float32 [N, 3]
```

这里的 `anchor_xy` 必须能和当前 sparse-depth sampling 选中的 patch 对齐。`track_flag` 不是单纯“有 target observation”，而必须同时满足 anchor 是真实 sparse seed、anchor alignment error 低于阈值、target observation 通过几何过滤。

如果当前 sparse depth `.npz` 没保留 `point3D_id`，则需要在 sparse depth generation 阶段或旁路 exporter 中补充 point id 到 sidecar。不要改采样策略本身。

---

## 8. Keyframe 选择策略

对每个 anchor observation `(image_i, u_i, point3D_id=m)`，从同一 COLMAP track 的其他 train observations 中选择一个 target keyframe observation `(image_j, u_j)`。

### 8.1 Hard filters

```text
anchor_is_sparse_seed = true
anchor_alignment_error_px <= align_threshold_px
j != i
image_j belongs to train split
point has positive depth in image_j
target pixel inside image_j
COLMAP reprojection error <= threshold
track length >= threshold
parallax in [parallax_min, parallax_max]
```

建议初值：

```text
min_track_length = 4
max_reprojection_error_px = 2.0
parallax_min_deg = 2.0
parallax_max_deg = 45.0 or 60.0
align_threshold_px = min(0.25 * feature_stride, 2.0)
sigma_align_px = align_threshold_px
```

### 8.2 Target selection rule

V1 不要采多个 target，也不要随机每步改变 target。为每个 anchor 固定选择一个 target，保证可复现。

推荐规则：

```text
在合法 target 中选择 parallax 最大但不超过 parallax_max 的 observation。
```

理由：

- 大 parallax 对 depth/scale 约束更强；
- 上限避免遮挡和极端视角；
- 固定选择便于 debug 和复现。

可选 tie-breaker：

```text
1. lower COLMAP reprojection error;
2. target frame has more shared tracks with anchor frame;
3. smaller temporal distance if sequence order可靠。
```

---

## 9. 训练接入方式

### 9.1 Buffer row 扩展

当前 buffer row 已有：

```text
features
target_px
gt_pose_inv
intrinsics
...
```

新增可选字段：

```text
target_px2
gt_pose_inv2
intrinsics2
track_flag
inter_frame_weight
alignment_weight
anchor_alignment_error_px
anchor_is_sparse_seed
optional point3D_id
optional colmap_xyz_world for diagnostics/fallback
```

这些字段仅在 `--use_colmap_keyframe_channel True` 时存在。

### 9.2 不新增 image-level dataloader

本设计不走：

```text
_build_sfm_track_dataloader()
_next_sfm_track_batch()
```

原因：

1. 当前已有 sparse-depth-guided buffer sampling；
2. SeqACE 的 keyframe channel 就是 buffer metadata；
3. target image 不需要 forward；
4. 只需要 target pose、K、pixel。

### 9.3 Training step

对每个 sampled prediction：

```text
P_i^w = predicted scene coordinate for anchor sample
```

已有 self reprojection：

```text
loss_self = ReproLoss(project_i(P_i^w), target_px)
```

新增 inter-frame reprojection：

```text
if track_flag:
    loss_inter = ReproLoss(project_j(P_i^w), target_px2)
else:
    loss_inter = 0
```

总损失：

```text
loss = loss_self + lambda_inter * track_flag * alignment_weight * loss_inter
```

其中 `ReproLoss` 应复用当前 ACE / LMC 的 robust schedule，而不是新增一套完全不同的 loss。

---

## 10. CLI 建议

最小参数：

```text
--use_colmap_keyframe_channel False
--colmap_keyframe_channel_path PATH
--inter_frame_ace_loss_weight 0.1
--inter_frame_ace_apply_to stage2
--inter_frame_ace_start_ratio 0.2
--inter_frame_ace_loss_type same_as_repro
--inter_frame_ace_invalid_colmap_weight 0.0
```

可选统计参数：

```text
--inter_frame_ace_log_stats True
```

不需要新增：

```text
--sfm_track_image_step_interval
--sfm_track_image_batch_size
--sfm_track_max_obs_per_image
```

这些属于旧 image-level auxiliary 设计，应删除。

---

## 11. Loss schedule

建议只在 S2 启用：

```text
S1: memory alignment / feature adaptation，不启用 inter-frame ACE loss
S2: coordinate refinement，启用 inter-frame ACE loss
```

初始权重：

```text
lambda_inter = 0.05 or 0.1
```

不要直接使用 SeqACE 的 `lambda=0.5`。本项目已有较强主 reprojection、LMC 训练流程和可能的 relative-depth 辅助；过大 cross 权重可能破坏主训练。

可用 warmup：

```text
inter_frame_ace_start_ratio = 0.2
```

即 S2 前 20% 步只用 self loss，后面逐渐启用 inter-frame loss。

---

## 12. Diagnostics

必须记录：

```text
track_flag ratio among sampled rows
anchor sparse-seed ratio
alignment_error_px median/p90/max
alignment_weight median/p10
samples rejected by alignment threshold
neighbor-expanded sample ratio excluded from IF loss
inter-frame valid projection ratio
behind-camera ratio
out-of-bounds ratio
nonfinite ratio
self reprojection median/p90
inter-frame reprojection median/p90
parallax median/p10/p90
track length median/p10/p90
inter loss raw / weighted
```

可选记录：

```text
||P_i^w - X_colmap|| median/p90
inter-frame error vs parallax
inter-frame error vs track length
inter-frame error vs COLMAP reprojection error
```

关键判断：

```text
inter-frame loss 是否在训练中下降；
inter-frame error 下降是否伴随 self reprojection / pose metrics 改善；
alignment_error 越小的样本是否更稳定；
大 parallax target 是否更有效；
track_flag 覆盖率是否足够高。
```

---

## 13. Ablation 矩阵

### 13.1 最小矩阵

| ID | 设置 | 目的 |
|---|---|---|
| Base | 当前 sparse-depth-guided sampling + self ACE loss | baseline |
| IF-0.05 | inter-frame ACE loss weight 0.05 | 最小正向测试 |
| IF-0.10 | inter-frame ACE loss weight 0.10 | 权重敏感性 |
| IF-strict-align | 更小 align threshold | 验证 patch-support 对齐重要性 |
| IF-no-align-gate | 不使用 alignment gate，仅作负面对照 | 验证原版 SeqACE-style 盲绑定是否有偏差 |
| IF-shuffle | 打乱 target keyframe channel | 验证几何对应是否真实有效 |
| IF-no-parallax-filter | 放宽 parallax filter | 验证 keyframe selection 重要性 |

### 13.2 不作为第一阶段

不要第一阶段做：

```text
XYZ-only main loss
image-level sfm dataloader
multi-target per anchor
relative-depth + inter-frame joint sweep
PMRF/STGS 联合
sequence inference
```

### 13.3 如果 IF loss 有正信号

再考虑：

```text
1. 多 target keyframe channel；
2. angular inter-frame reprojection；
3. invalid COLMAP fallback；
4. 与 PMRF/STGS memory centroid consistency 结合。
```

---

## 14. 与普通 XYZ loss 的区别

普通 sparse XYZ loss：

```text
P_i^w -> X_colmap
```

COLMAP-Keyframe Inter-Frame ACE loss：

```text
P_i^w -> anchor pixel u_i
P_i^w -> target keyframe pixel u_j
```

前者是点坐标回归，后者是多视图 ACE 几何约束。

后者更符合：

1. ACE / DSAC 的 reprojection training philosophy；
2. SeqACE 的 keyframe cross-loss 思想；
3. 本项目保持单图推理的边界；
4. 通过多视角投影约束 depth/scale 的目标。

---

## 15. 与当前 sparse-depth sampling 的关系

当前已有 sparse-depth-guided sampling 负责：

```text
选择 anchor patch
```

本设计负责：

```text
给被选中的 anchor patch 附加 keyframe channel
```

两者关系：

```text
sparse depth sampling tells us where to sample;
COLMAP keyframe channel tells us how to supervise the sampled point from another view.
```

不要把两者混成一个新的 sampler。

同时必须区分两个 mask：

```text
valid_for_sampling:
  sparse seed + neighbor-expanded ROI

valid_for_inter_frame_loss:
  sparse seed only
  + has point3D_id
  + anchor-side patch-support aligned
  + has valid target keyframe channel
```

也就是说，neighbor-expanded samples 仍然可以提升普通 ACE self-loss 的采样覆盖，但默认不能使用 seed point 的 keyframe channel。

---

## 16. 与 PMRF / STGS memory consistency 的关系

本文档只定义 supervision-side inter-frame ACE loss。

如果未来要和 PMRF 结合，不应直接约束 token distribution 一致。更合理的是利用 COLMAP track 做：

```text
same track observations -> predicted coordinates should be cross-view consistent
same track observations -> memory attention centroids can be close in 3D
```

但这属于后续，不属于 V1。

V1 只做：

```text
SeqACE-style keyframe channel + inter-frame ACE loss
```

---

## 17. 推理边界

必须明确：

```text
COLMAP keyframe channel 只在训练期使用。
推理期不需要 COLMAP。
推理期不需要 sparse depth。
推理期不需要 target keyframe。
推理期不需要序列。
推理期不需要 feature matching。
```

推理仍然是：

```text
single query image
-> backbone / memory-conditioned network
-> scene coordinate map
-> DSAC / PnP
-> camera pose
```

---

## 18. 论文表述建议

英文：

> We extend ACE-style training with a patch-support-aligned COLMAP-keyframe inter-frame reprojection loss. For sparse-depth-guided training samples, we use the COLMAP point track to retrieve a verified keyframe observation of the same 3D point, but enable the cross-view loss only when the COLMAP anchor observation is spatially aligned with the sampled feature cell. The predicted scene coordinate from the anchor view is then supervised not only by the anchor reprojection loss, but also by an alignment-weighted keyframe reprojection loss. This follows the spirit of SeqACE's keyframe channel, while avoiding the patch-keypoint support mismatch that can arise when pairwise matches are blindly attached to stride-based training patches. The additional supervision is used only during training and adds no inference-time dependency.

中文：

> 我们在 ACE 风格训练中加入 patch-support 对齐的 COLMAP 关键帧通道帧间重投影损失。对于已有稀疏深度采样选中的训练 patch，我们通过 COLMAP point track 找到同一 3D 点在另一个训练关键帧中的几何验证观测，但只有当 anchor 侧 COLMAP observation 与被采样 feature cell 的代表位置足够对齐时，才启用 cross-view loss。anchor 视角预测出的同一个 scene coordinate 不仅需要投影回当前像素，还需要通过 alignment-weighted keyframe reprojection loss 投影到关键帧中的对应像素。该设计继承了 SeqACE keyframe channel 的思想，同时避免原版 pairwise match 盲目绑定到 stride-based training patch 时可能产生的 patch-keypoint support mismatch。该监督只在训练期使用，推理期没有任何额外依赖。

---

## 19. 最终推进顺序

1. 审计当前 sparse depth sampling 输出是否能追溯到 COLMAP `point3D_id`。
2. 如果不能，补充 sidecar exporter，把 sparse-depth seed anchor pixel 映射到 COLMAP track observation。
3. 为每个 anchor observation 计算 feature-cell representative point、alignment error 和 alignment weight。
4. 只为通过 anchor-side patch-support alignment 检查的 seed observation 固定选择一个合法 target keyframe observation。
5. 在 buffer row 中附加 `target_px2 / pose2 / K2 / track_flag / alignment_weight / alignment_error`。
6. 在 training step 中复用当前 ReproLoss 计算 alignment-weighted inter-frame ACE loss。
7. 先跑 `IF-0.05`、`IF-strict-align`、`IF-no-align-gate` 和 `IF-shuffle`，验证收益来自对齐后的真实几何对应。
8. 有正信号后再考虑 angular loss、multi-target、invalid COLMAP fallback 或与 PMRF 的 centroid consistency 结合。

停止条件：

```text
1. track_flag 覆盖率太低；
2. alignment 后剩余样本太少；
3. IF loss 不下降；
4. IF-shuffle 与真实 keyframe channel 表现接近；
5. IF-no-align-gate 与 aligned version 表现接近或更好，说明 patch-support 假设/实现需要复查；
6. self reprojection / pose metrics 变差；
7. 收益只来自简单 XYZ fallback，而不是 inter-frame reprojection。
```

最终判断标准：

> 只有当 patch-support 对齐后的真实 COLMAP keyframe channel 的 inter-frame ACE loss 明显优于 shuffle control 和 no-alignment blind-binding control，并且不损害主 self reprojection 与最终 pose metrics，才能把它作为监督侧创新。
