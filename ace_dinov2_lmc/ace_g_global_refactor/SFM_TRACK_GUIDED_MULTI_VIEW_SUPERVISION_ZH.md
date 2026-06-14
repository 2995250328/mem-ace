# SfM-Track Guided Multi-View Supervision

Updated: 2026-06-13

定位：本文档单独记录监督侧创新 **SfM-Track Guided Multi-View Supervision (STGS)**。它从原“双插件”总计划中拆出，专门描述稀疏深度 / SfM track / 多视图重投影监督的研究依据、方法边界、数据合同、实现计划和实验矩阵。

核心结论：

1. STGS 是训练期监督，不改变单图测试接口。
2. 参考 Xu 等 2024 / SeqACE 的 sequence-based mapping，但不采用其测试时 sequence-based relocalization。
3. 如果已有 COLMAP / SfM tracks，优先直接使用真实 track identity、3D 点和多帧观测，而不是重新跑 SuperPoint/LightGlue 或 optical flow。
4. 第一版先做 metric XYZ sparse supervision，再做 cross-view reprojection；不要一开始同时加 relative-depth、temporal model、sequence inference 或新的 matcher。

## 1. 为什么需要单独文档

早期“双插件”总计划同时记录结构侧和监督侧机制。结构侧已经从 ZG-RMR/CIFA 调整为 PMRF；监督侧 STGS 仍然成立，但它的依据、数据合同和实现边界足够独立，继续塞在总文档里容易造成三个混淆：

1. 把 **sequence supervision** 误读成测试时序列输入；
2. 把 **稀疏深度**、**SfM XYZ**、**cross-view reprojection** 混为同一个 loss；
3. 把 Xu/SeqACE 的工程实现直接搬过来，而不是利用本项目已有的 COLMAP track 资源。

本文档只维护监督侧合同。结构侧 PMRF 另见 `PROGRESSIVE_MEMORY_REREADING_FUSION_PLAN_ZH.md`。

## 2. 文献与代码检索结论

### 2.1 ACE / DSAC* 背景

[ACE](https://arxiv.org/abs/2305.14059) 将 SCR 拆成冻结的 scene-agnostic backbone 与 scene-specific MLP head，用 RGB+pose 和 reprojection loss 在几分钟内完成场景编码。它不需要训练期 3D model 或 depth，但也意味着几何主要来自隐式三角化。

[DSAC*](https://arxiv.org/abs/2002.12324) 证明 RGB-only SCR 可以只用 pose 与可微 RANSAC / reprojection 学习场景坐标，但训练更慢，且单视角 reprojection 对射线方向上的深度约束弱。

STGS 不替换 ACE/DSAC 的主监督，而是在已有 reprojection loss 旁边加入训练期稀疏、多视图的几何约束。

### 2.2 Xu 等 2024 / SeqACE

本地论文：

```text
/home/xwh/project/ace_depth/papers/Xu 等 - 2024 - An Efficient Scene Coordinate Encoding and Relocalization Method.pdf
```

网页版本与代码：

- paper: [arXiv 2412.06488](https://arxiv.org/abs/2412.06488)
- official code: [sair-lab/SeqACE](https://github.com/sair-lab/SeqACE)

论文的 sequence-based mapping 包含三部分：

1. 用 keypoint detection / saliency 选择更稳定的训练 patches；
2. 对每个 patch 保留 ACE 风格 self-loss，即当前帧 reprojection loss；
3. 通过特征匹配把当前帧 patch 与 keyframe patch 关联，加入 cross-loss。

其 self-loss 可概括为：

\[
L_{S,i} =
\begin{cases}
\tau(t)\tanh(e_\pi(p_i, P_i^w, T_{wc})/\tau(t)), & P_i^w \in V \\
\lVert P_i^w - \bar P_i^w \rVert, & \text{otherwise}
\end{cases}
\]

其中 \(\bar P_i^w\) 是用 fixed target depth 反投影得到的 invalid proxy target，\(\tau(t)\) 从大到小调度。

其 cross-loss 是把当前预测点投影到 keyframe：

\[
L_{C,i} =
\begin{cases}
\tau(t)\tanh(e_\pi(p_{k,j}, P_i^w, T_{wk})/\tau(t)), & P_i^w \in V \\
\lVert P_i^w - \bar P_{k,j}^w \rVert, & \text{otherwise}
\end{cases}
\]

训练总损失：

\[
L = \frac{1}{N_b}\sum_i(L_{S,i}+\lambda L_{C,i}),
\quad \lambda=0.5 \text{ for inlier matches, else } 0.
\]

官方代码细节：

- `feature_matching_for_traing.py` 使用 SuperPoint + LightGlue 离线匹配，并用 fundamental matrix RANSAC 过滤；
- keyframe 规则依赖匹配数量和平均视差；
- `ace_trainer.py` 在 training buffer 中额外存 `target_px2`、`gt_poses_inv2`、`intrinsics2`、`track_flag`；
- 训练时对同一预测 scene coordinate 分别算当前帧 loss 和 keyframe cross loss，cross 权重固定为 `0.5`。

对本项目的启发：

1. 可以借鉴 **训练期多视图约束**；
2. 可以借鉴 **同一个 anchor prediction 投到另一个视图**，无需 target 图像 forward；
3. 不必借鉴其 **重新匹配 pipeline**，因为 COLMAP/SfM tracks 已经给出更强的跨帧 identity；
4. 不应借鉴其 **测试时 sequence mode**，因为当前论文主线需要保持单图 relocalization。

### 2.3 Li 等 2018：angle-based reprojection 与 multi-view constraint

[Scene Coordinate Regression with Angle-Based Reprojection Loss](https://arxiv.org/abs/1808.04999) 指出传统 reprojection loss 有退化，angle-based reprojection 可以缓解初始化依赖，并能利用 available multi-view constraints。

对 STGS 的启发：

- cross loss 可以考虑 pixel reprojection，也可以考虑 bearing/angular residual；
- 第一版仍用 pixel reprojection，因为它和 ACE/SeqACE 的 `ReproLoss` 兼容；
- 如果 pixel cross 对大尺度 / 大深度不稳定，第二版可以加入 angular cross ablation。

### 2.4 FocusTune

[FocusTune](https://arxiv.org/abs/2311.02872) 用 3D model 投影到训练图像，围绕关键几何区域采样，从而避免 ACE 均匀采样大量无效区域。

对 STGS 的启发：

- 稀疏 3D 点不仅可以提供 loss，也可以提供 training-point sampling prior；
- 但第一版不要同时改采样策略和 loss，否则难以定位收益来源；
- 若 XYZ loss 有正信号，后续可做 “track-aware sampling only” control。

### 2.5 GLACE

[GLACE](https://arxiv.org/abs/2406.04340) 的核心诊断是大场景中 SCR 需要同时具备跨视角不变性和相似区域可分性，纯隐式三角化在重复纹理下困难。

对 STGS 的启发：

- STGS 不是和 GLACE/PMRF 竞争，而是补充监督；
- GLACE 类 global/local conditioning 改 feature representation，STGS 改训练约束；
- 最终应做 2x2：结构增强开/关 x STGS 开/关。

### 2.6 KFNet、DSM、HSCNet 等替代方向

[KFNet](https://arxiv.org/abs/2003.10629) 和 [Dense Scene Matching](https://arxiv.org/abs/2103.16792) 都能利用 temporal / scene matching 信息提升定位，但它们会改变测试时输入或架构边界。[HSCNet](https://arxiv.org/abs/1909.06216) 用 coarse-to-fine 分类回归提升大场景鲁棒性，但会改变 head 与训练目标。

这些方法适合长期对照，不适合 STGS v1：

- KFNet：测试时序列与滤波状态，不符合单图推理边界；
- DSM/SANet 路线：需要 scene matching / cost volume，不是 ACE head 的轻量插件；
- HSCNet：需要区域分类层，改变 coordinate head contract；
- D2S / sparse descriptor matching：更接近显式 map，不是当前 compact implicit map 叙事。

## 3. 与 Xu/SeqACE 的逐项差异

| 项目 | Xu/SeqACE | STGS v1 |
|---|---|---|
| 关联来源 | SuperPoint + LightGlue + F-matrix RANSAC | COLMAP / SfM track identity |
| 3D 点 | 当前网络预测 + invalid proxy depth | 已三角化 `track_xyz_world` + 网络预测 |
| anchor loss | ACE self reprojection | ACE 主 loss 保持不变，额外加 sparse XYZ |
| cross loss | anchor prediction 投到 keyframe matched pixel | anchor prediction 投到同 track target observation |
| target forward | 不需要 | 不需要 |
| 测试时序列 | 提供 sequence-based relocalization | 明确不做 |
| 数据依赖 | 离线匹配文件 | train-only track package |
| 适用数据 | 连续序列更自然 | 任意有可靠 SfM tracks 的训练集 |

核心替换：

```text
SeqACE matched keyframe pixel
  -> STGS target observation from same COLMAP point3D track

SeqACE pseudo target from fixed depth
  -> STGS metric target from triangulated XYZ

SeqACE sequence assumption
  -> STGS multi-view track assumption
```

## 4. 方法定义

给定 SfM track \(m\)：

- \(X_m^w\)：该 track 的三维点，位于 ACE raw world 坐标；
- \((i, u_i)\)：anchor 图像和该点在图像 \(i\) 中的观测；
- \((j, u_j)\)：同一 track 在 target 图像 \(j\) 中的另一个观测；
- \(S_i\)：网络对图像 \(i\) 输出的 scene-coordinate map；
- \(K_j, T_{j,w}\)：target 图像的内参和 world-to-camera 变换。

### 4.1 Sparse XYZ loss

对 anchor 观测做双线性采样：

\[
P_i^w = \mathrm{Sample}(S_i, u_i).
\]

metric XYZ loss：

\[
L_{xyz} = \frac{1}{N}\sum_{(m,i)} \rho(P_i^w - X_m^w).
\]

V1 设置：

```text
rho = component-wise SmoothL1
beta = 0.1 m
coordinate_space = raw ACE world
initial weight = 0.01
```

这个 loss 是最干净的第一步，因为它直接验证“稀疏真实 3D 监督是否能改善 coordinate head”，不引入 target camera 投影边界。

### 4.2 Cross-view reprojection loss

将 anchor prediction 投影到 target 观测图像：

\[
\hat u_j = \pi(K_j T_{j,w} P_i^w).
\]

cross loss：

\[
L_{cross} = \frac{1}{N_p}\sum_{(i,j)} \rho_\pi(\hat u_j - u_j).
\]

V1 设置：

```text
rho_pi = ACE/SeqACE-compatible dyntanh reprojection loss or SmoothL1 pixel loss
initial weight = 0.0
enable only after XYZ passes
target image forward = never
```

cross loss 的意义不是再监督 target 图像，而是要求 anchor 图像预测出的 3D 点也能解释同一 track 在另一个视角的观测。

### 4.3 总损失

第一阶段：

\[
L = L_{ACE} + \lambda_{xyz} L_{xyz}.
\]

第二阶段：

\[
L = L_{ACE} + \lambda_{xyz} L_{xyz} + \lambda_{cross} L_{cross}.
\]

不在第一版加入：

- relative depth distillation；
- saliency/keypoint auxiliary loss；
- optical flow temporal consistency；
- sequence inference loss；
- routing / usage loss；
- dense correspondence distillation。

## 5. 数据合同

STGS 读取一个独立的 train-only track package：

```text
<scene>/train/sfm_tracks_v1/
  manifest.json
  tracks.npz
```

### 5.1 `manifest.json`

必需字段：

```text
schema_version
scene_name
split = train
coordinate_space = ace_world
source_type = known_pose_colmap | colmap | nvm
source_model
image_root
min_track_length
max_reprojection_error_px
num_images
num_tracks
num_observations
image_names
```

原则：

1. 默认拒绝非 train split；
2. 不写入 test image name；
3. 不写入 descriptor 或 RGB feature；
4. 坐标系必须和 ACE 训练用 raw world 对齐。

### 5.2 `tracks.npz`

必需数组：

```text
track_ids             int64    [T]
track_xyz_world       float32  [T, 3]
track_reproj_error    float32  [T]
track_length          int32    [T]
track_offsets         int64    [T + 1]

obs_image_idx         int32    [O]
obs_xy_original       float32  [O, 2]

image_size_hw         int32    [I, 2]
camera_K_original     float32  [I, 3, 3]
camera_c2w_world      float32  [I, 4, 4]

image_obs_offsets     int64    [I + 1]
image_obs_indices     int64    [O]
```

cross loss 需要额外数组：

```text
pair_anchor_obs_idx   int64    [P]
pair_target_obs_idx   int64    [P]
pair_parallax_deg     float32  [P]
```

### 5.3 过滤策略

V1 hard filters：

```text
track length >= 4
point reprojection error <= 2.0 px
finite XYZ and pixel
pixel inside source image
image present in train RGB set
positive depth in observing camera
```

pair filters：

```text
anchor and target image must differ
parallax in [2 deg, 60 deg]
one target per anchor in V1
prefer largest valid parallax below upper bound
discard anchor without valid target
```

## 6. 本仓库可复用代码

已有稀疏深度 / COLMAP 相关代码：

```text
tools/project_colmap_sparse_depth.py
tools/wayspots_known_pose_sparse_depth.py
tools/project_cambridge_nvm_sparse_depth.py
```

这些脚本证明本仓库已经有：

1. ACE 格式的 RGB / pose / calibration 读取；
2. COLMAP / NVM sparse point 解析；
3. train split triangulation / projection；
4. sparse depth `.npz` 生成。

但现有 sparse depth 文件不够 STGS 使用，因为它们丢掉了：

```text
point3D_id
track identity
subpixel observation xy
per-observation image id
target pair
```

因此 STGS 需要新 exporter，不能复用 sparse depth `.npz` 作为 track package。

## 7. 数据集策略

### 7.1 Wayspots：V1 首选

原因：

1. 已有 train-only known-pose triangulated model；
2. image names 与 ACE train images 可以精确对齐；
3. SquareBench 是已知 global conditioning failure case，适合验证监督是否能改善困难场景；
4. track 数量足够支持稀疏监督和 cross pair sampling。

第一目标：

```text
wayspots_squarebench
```

第二目标：

```text
wayspots_therock 或 wayspots_tendrils
```

### 7.2 Indoor6：暂不做 full STGS

当前问题：

```text
COLMAP image names: 06-frames001539.jpg
ACE train names:    000000.jpg
direct overlap:     0
```

不能用排序或近邻 pose 强行匹配。除非完成独立 image-name / pose alignment audit，否则 Indoor6 只能做：

- sparse depth metric supervision；
- 不带 track identity 的 sparse XYZ；
- 或完全不启用 STGS。

### 7.3 Cambridge / MuSHRoom / RIO10

Cambridge 有 NVM，可以未来写 `nvm_tracks_v1` exporter，但需要坐标系审计。

MuSHRoom 是否有可靠 train-only COLMAP tracks 需要单独 audit。

RIO10 当前 sparse depth 没有 persistent point ID，不支持 cross-view STGS。

## 8. 实现计划

### 8.1 Phase B0：track exporter

新增：

```text
tools/export_sfm_tracks.py
```

职责：

1. 读取 COLMAP / NVM reconstruction；
2. 只接受 train split；
3. 验证 image-name mapping；
4. 验证 pose / intrinsics 与 ACE 文件一致；
5. 导出 `manifest.json` 和 `tracks.npz`；
6. 打印 track length、reprojection error、observation per image、parallax 分布；
7. 支持 `--audit_only`；
8. 可选输出 RGB overlay 供人工检查。

SquareBench 出口前必须满足：

```text
100% train image-name mapping
pose max error <= 1e-4
intrinsics max error <= 1e-3
finite XYZ ratio = 100%
pixels in bounds
no test image names
observations in >= 95% train images
```

### 8.2 Phase B1：XYZ-only STGS

新增：

```text
sfm_track_supervision.py
```

包含：

```text
SFMTrackStore
sample_scene_coordinates_at_observations
compute_track_xyz_loss
```

新增最小 CLI：

```text
--use_sfm_track_supervision False
--sfm_track_path PATH
--sfm_track_xyz_weight 0.01
--sfm_track_cross_weight 0.0
--sfm_track_apply_to stage2
--sfm_track_image_step_interval 10
--sfm_track_image_batch_size 1
--sfm_track_max_obs_per_image 512
--sfm_track_xyz_beta_m 0.1
--sfm_track_start_ratio 0.2
```

训练路径：

1. 初始化并校验 `SFMTrackStore`；
2. 建立 unaugmented image dataloader；
3. 每 `sfm_track_image_step_interval` 步取一个完整图像 batch；
4. 走当前配置的 backbone / memory / fusion / head；
5. 恢复 raw-world scene coordinates；
6. 在 track observations 上双线性采样；
7. 计算 `L_xyz`；
8. 记录覆盖率、metric error、raw/weighted loss；
9. checkpoint 只保存元数据，不保存 track arrays。

### 8.3 Phase B2：cross-view STGS

只在 B1 通过后实现。

步骤：

1. 从 package 读取 `pair_anchor_obs_idx` 和 `pair_target_obs_idx`；
2. 对 anchor 图像预测图采样 \(P_i^w\)；
3. 用 target pose/intrinsics 投影得到 \(\hat u_j\)；
4. 计算 pixel reprojection residual；
5. 记录 valid pair ratio、behind-camera、nonfinite、pixel median/p90；
6. 与 XYZ-only 做严格 ablation。

## 9. 像素与坐标细节

### 9.1 禁用增强

STGS full-image auxiliary batch 必须使用：

```text
augment=False
aug_rotation=0
aug_scale_min=1.0
aug_scale_max=1.0
```

理由：现有 sampled S2 buffer 不保留完整 original-to-augmented transform。第一版不能把增强反变换作为隐性误差源。

### 9.2 原图像素到 resized input

对原图 \(H_0,W_0\) 和网络输入 \(H_r,W_r\)：

```text
xr = (x0 + 0.5) * Wr / W0 - 0.5
yr = (y0 + 0.5) * Hr / H0 - 0.5
```

### 9.3 resized input 到 output grid

ACE 输出中心：

```text
pixel = stride * (grid + 0.5)
```

因此：

```text
grid_x = xr / stride - 0.5
grid_y = yr / stride - 0.5
```

用 `grid_sample` 做双线性采样，并显式返回 validity mask。不要把 subpixel observation round 到整数 feature cell。

### 9.4 raw-world recovery

如果训练路径使用 C1/reference normalization，loss 前必须恢复到 raw ACE world：

```text
head output
  -> C1/reference recovery
  -> optional coord_sigma / coord_mu inverse normalization
  -> raw-world P_i^w
```

XYZ 和 cross 共用同一个 recovery helper。

## 10. 实验矩阵

主场景：

```text
wayspots_squarebench
```

| ID | Fusion | XYZ | Cross | 目的 |
|---|---|---:|---:|---|
| B0 | accepted baseline | 0 | 0 | control |
| B1 | same | 0.01 | 0 | 最小 STGS |
| B2 | same | 0.05 | 0 | 仅当 B1 明显 underweight |
| B3 | same | best | 0.05 | cross 是否超过 XYZ |
| B-shuffle | same | best | best | track-image shuffle negative control |
| B-hard | same on second scene | best | best | 第二场景验证 |

与结构创新组合时使用 2x2：

| ID | PMRF | STGS |
|---|---|---|
| C0 | off | off |
| C1 | on | off |
| C2 | off | on |
| C3 | on | on |

只有 C3 同时保留 C1/C2 优点时，才能说“结构与监督互补”。

## 11. 必须记录的诊断

STGS 训练日志至少包含：

```text
sfm_track_enabled
sfm_track_images_with_obs
sfm_track_obs_requested
sfm_track_obs_valid
sfm_track_obs_used
sfm_track_xyz_loss_raw
sfm_track_xyz_loss_weighted
sfm_track_xyz_error_median_m
sfm_track_xyz_error_p90_m
sfm_track_cross_pairs_requested
sfm_track_cross_pairs_valid
sfm_track_cross_pixel_l1_median
sfm_track_cross_pixel_l1_p90
sfm_track_behind_camera_ratio
sfm_track_nonfinite_ratio
sfm_track_aux_step_time_ms
```

结果分析必须按以下维度 bucket：

```text
track length
point reprojection error
parallax
observations per image
image region / sparse point density
```

## 12. 验证与测试

合成测试：

1. valid package loads；
2. malformed offsets fail；
3. non-train source fails by default；
4. image-name mismatch fails；
5. pixel resize transform is correct；
6. bilinear sampling recovers a synthetic linear coordinate field；
7. XYZ loss is zero for exact prediction；
8. cross loss is zero for exact projection；
9. raw-world normalization round-trip is exact within tolerance；
10. no-track images return differentiable zero。

真实数据 gate：

```text
export audit passes
overlay inspection passes
one CUDA smoke run has finite loss and gradients
checkpoint save succeeds
checkpoint eval succeeds without track files
```

最后一条很关键：checkpoint eval 不依赖 track files，证明 STGS 没有改变 inference boundary。

## 13. 风险与停止条件

### 13.1 主要风险

| 风险 | 后果 | 缓解 |
|---|---|---|
| 坐标系错位 | XYZ loss 直接拉坏 head | exporter 做 pose/intrinsics audit，先 synthetic zero test |
| train/test leakage | 论文不可辩护 | exporter 默认拒绝 non-train，manifest 记录 split |
| sparse loss 过强 | dense reprojection collapse | start ratio + ramp + 小权重 |
| cross pair baseline 太小 | 约束近似重复 self-loss | parallax lower bound |
| cross pair baseline 太大 | occlusion/outlier 多 | parallax upper bound + reprojection error filter |
| overhead 过高 | 训练成本不划算 | 每 10 step 一次，max obs per image |

### 13.2 停止条件

停止 STGS 扩展，如果出现：

1. exact train-only alignment 无法证明；
2. XYZ-only 不改善 pose 或 strict metric；
3. sparse XYZ error 降低但 dense pose 明显变差；
4. shuffled control 和真实 tracks 差不多；
5. cross 不超过 XYZ-only；
6. 需要强场景调参才有效；
7. 训练开销不成比例。

负结果后不要马上加 optical flow、sequence model、learned matcher 或 test-time sequence input。先定位是数据合同、loss 权重、采样覆盖还是坐标恢复问题。

## 14. 论文表述建议

推荐表述：

> We introduce a training-only SfM-track guided supervision objective for latent-memory-conditioned scene coordinate regression. Verified train-set tracks provide sparse metric 3D anchors and cross-view reprojection constraints, strengthening implicit triangulation without changing the single-image inference interface.

中文含义：

> 我们使用训练集内已验证的 SfM tracks，为 scene coordinate regression 加入稀疏三维锚点和跨视角重投影约束。该监督只在训练期使用，推理仍是单张图像输入和原 DSAC/PnP 求解器。

避免表述：

```text
no 3D geometry is used
sequence localization
test-time temporal model
free supervision
universal improvement
works on all datasets
```

准确表述：

```text
uses additional train-time reconstructed geometry
does not require extra inference input
supports only datasets with audited train tracks
```

## 15. 开工顺序

```text
1. export_sfm_tracks.py audit-only
2. SquareBench train package + overlays
3. sfm_track_supervision.py synthetic tests
4. XYZ-only trainer integration
5. SquareBench B0/B1
6. second Wayspots scene
7. cross-view B3 only after XYZ passes
8. PMRF x STGS 2x2
```

## 16. 当前决策

STGS 值得作为监督侧独立创新继续推进，但第一版必须保守：

1. Wayspots train-only tracks；
2. XYZ-only first；
3. unaugmented auxiliary full-image batch；
4. raw-world metric target；
5. cross loss 延后；
6. 不引入测试时序列；
7. 不用 sparse depth `.npz` 冒充 track package。

这条路线和 PMRF 的关系是互补而不是互相依赖：PMRF 回答 memory 如何被更好读取，STGS 回答坐标监督如何更几何一致。
