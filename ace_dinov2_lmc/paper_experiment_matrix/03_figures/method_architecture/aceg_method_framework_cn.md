# ACE-G 场景压缩表征方法框架理解稿

日期：2026-06-28

本文档用于把方法主线先讲清楚，再进入流程图绘制。当前版本按你给出的事实收束范围：

- 只关心 ACE-G 训练流程。
- ACE-G 包含 S1 和 S2-G 两个阶段。
- GeoLMC 压缩模块只关心 global 模式。
- Fusion 只关心 single 模式。
- Loss 主线包括单帧重投影，以及多帧 / track 重投影。
- 与 GLACE 的结合只关心原始 concat 模式，不展开 residual / FiLM / 其他桥接。
- 论文故事重点是：设计一种针对场景的压缩表征方法，为现有 SCR 方法提供通用增强。实验可落在 ACE、DINO-ACE、GLACE 上，但流程图重点不是某个下游 head 的桥接细节，而是 memory 如何由 COLMAP 恢复点云构建、如何压缩、如何融合，以及如何同时准备多帧 track 重投影需要的稀疏深度、track 对应和重建点云信息。

## 1. 一句话方法定义

我们的方法是在 scene coordinate regression（SCR）框架上增加一个场景级压缩记忆分支：先从训练集和重建几何中构建一个 scene memory bank，再用 GeoLMC 将大量场景点和特征压缩成少量 global latent tokens，最后在查询图像的 dense feature 上通过 single cross-attention fusion 注入这些 scene tokens，从而增强 ACE / DINO-ACE / GLACE 等 SCR regressor 的场景表征能力。

核心不是替换 SCR 的 pose solver，也不是重新设计 GLACE head，而是给现有 SCR pipeline 加一个统一的场景压缩表征接口：

```text
训练图像 + 位姿 + COLMAP 稀疏点云 / 稀疏深度 / tracks
  -> scene memory bank
  -> GeoLMC global compressor
  -> K 个 latent scene tokens
  -> single memory-to-image fusion
  -> SCR coordinate head
  -> scene coordinates
  -> DSAC* / RANSAC pose
```

## 2. 方法要解决的问题

传统 SCR 方法把每张 query image 编码成 dense features，然后直接回归每个像素 / patch 的 3D scene coordinate。这个流程有两个限制：

1. 单张 query 的特征只包含当前图像可见内容，对场景长期结构、跨视角几何和稀疏重建信息利用不足。
2. 如果直接把大量训练帧或 3D 点作为 memory 参与推理，计算量和噪声都会变大，而且 memory 与不同 SCR backbone 之间很难统一。

因此我们设计一个 scene compressed representation：

- memory construction 阶段负责把训练集中的多视角图像、位姿、COLMAP 恢复点云、由点云投影得到的稀疏深度 / 3D 对应，以及 backbone 特征整理成统一 memory bank；
- GeoLMC compressor 负责把这个大 memory bank 压缩成固定数量的 latent scene tokens；
- fusion 模块负责把 latent scene tokens 注入 query dense features；
- 下游仍然保持 SCR 的标准输出：dense scene coordinates，再交给 DSAC* / RANSAC 求 pose。

这个设计使方法能作为通用增强模块接到 ACE、DINO-ACE、GLACE 等已有 SCR 方法上。

## 3. 总体流程

整体可以分成四个部分：

1. Offline memory construction：构建 scene memory bank。
2. Global memory compression：用 GeoLMC global 模式压缩 memory。
3. Query feature fusion：用 single fusion 把 memory tokens 融合进 query features。
4. ACE-G training：用 S1 / S2-G 两阶段训练压缩器、融合模块和 coordinate head，并用单帧和多帧几何约束监督。

推理时，memory compression 是 scene-level 的，不是 frame-level 的。对一个场景，压缩器只运行一次，得到 K 个 latent scene tokens；之后所有 test query 都复用这组 tokens。

## 4. Offline memory construction

Memory construction 是方法的第一核心。它把训练场景整理成可被 GeoLMC 消费的 pooled memory bank。

### 4.1 输入

输入来自训练 split，主要包括：

- RGB training images；
- camera-to-world pose；
- camera intrinsics；
- COLMAP 恢复的 sparse 3D point cloud；
- 由 COLMAP 点云投影 / 对齐到训练图像得到的 sparse depth 或 patch-level 3D point support；
- COLMAP tracks 及其跨图像 2D observations；
- backbone 或 foundation model 提取的 dense / patch features。

这些信息最终要落到统一格式：

```text
pooled_points:   N x 3
pooled_features: N x D
scene_center:    3
all_poses:       M x 4 x 4 或 M x 3 x 4
all_intrinsics:  M x 3 x 3
可选: all_scale_tokens, rays, Plucker rays, cluster_sizes, view metadata
```

其中 `pooled_points` 是 memory 中的 3D scene points，`pooled_features` 是对应点的 feature，二者行数必须一致。

### 4.2 构建 3D memory points

构建 memory point 的口径统一为：ACE-FCN memory 和 MapAnything memory 都从 COLMAP 恢复的稀疏点云出发。COLMAP 流程同时产生 sparse 3D points 和 tracks；前者用于得到 memory 的稀疏深度 / 3D 点支撑，后者用于后续 track 重投影监督。

统一流程是：

1. 对训练图像运行或读取 COLMAP reconstruction，得到 sparse 3D point cloud、每个 3D point 的 track observations，以及相机几何信息。
2. 把 COLMAP 3D points 投影 / 对齐到训练图像，形成 sparse depth 或 patch-level 3D point support。
3. 在对应图像上提取 backbone features。
4. 将 feature grid 上的 patch / pixel 与可见 COLMAP 3D point 建立对应。
5. 得到统一的 memory pair：

```text
(3D scene point p_i, visual feature f_i)
```

这一步的重点是“点云对应 + 特征提取”，而不是为不同 memory source 设定不同几何来源。

对于 ACE-FCN 与 MapAnything memory，差别主要在 feature extractor，而不是几何来源：

- ACE-FCN memory 使用 frozen ACE-FCN encoder 的 patch features；
- MapAnything memory 使用 MapAnything / foundation-feature 分支提取的 memory features；
- 二者都应表述为和 COLMAP sparse point cloud / sparse depth 建立 3D 对应后形成 memory bank。

如果没有可用的 COLMAP 稀疏点云、稀疏深度或有效 feature-point 对应，不应在论文叙事里暗示生成了可靠伪坐标。

完成 feature-point 对应后，再用 BSE / voxel / pooled 策略对 points 和 features 做空间聚合，保存为 pooled memory 文件。

### 4.3 Feature 与 point 的配对

Memory bank 的关键不是只有点云，而是每个 3D point 必须带有 feature：

```text
(p_i, f_i)
```

其中：

- `p_i` 是该 memory cell 的 3D world coordinate；
- `f_i` 是对应 view / patch / voxel 聚合出来的视觉特征；
- 如果多层 feature 被保留，`f_i` 可以是多层 concat；
- `layers_idx` 记录 feature 层来源，训练前会检查 feature dim 是否能被层数整除。

在图里，memory bank 应画成：

```text
Pooled Scene Memory Bank
{3D points p_i, features f_i}
```

不能把它画成 K 个 tokens。K tokens 是 GeoLMC 压缩后的结果。

### 4.4 BSE / pooled 聚合

Memory construction 可以对原始 points/features 做聚合。BSE 或 voxel pooling 的作用是：

- 减少 memory 点数；
- 保持局部几何边界；
- 聚合颜色、feature、ray direction、camera center 等附加信息；
- 可保存 Plucker rays 等多视角几何描述。

保存时会同时保留：

- normalized points；
- world points；
- `pooled_points` 作为 legacy alias，通常指向 world coordinates；
- `pooled_features` 作为 feature alias；
- `scene_center`，通常是相机中心均值或与训练坐标系一致的 anchor。

### 4.5 构建重建点云与 track 信息

COLMAP 恢复流程同时服务两个输出：

1. 给 memory construction 提供 sparse point cloud / sparse depth / feature-point correspondences。
2. 给多帧 track 重投影提供 track sidecar。

因此，为了支持多帧 track 重投影，memory construction 不只需要普通 memory bank，还需要来自 COLMAP 的前处理信息：

- COLMAP sparse reconstruction；
- 3D point tracks；
- 每个 track 在多个图像中的 2D observations；
- 每帧的 camera pose 和 image intrinsics；
- 每个 observation 对应到模型输入分辨率下的像素坐标；
- 每个 observation 对应到 feature grid 的 cell index；
- anchor frame 和 target frame 的对应关系。

`tools/build_colmap_keyframe_channel.py` 的作用就是把同一套 COLMAP reconstruction 中的 tracks 对齐到 SCR feature grid，生成 `keyframe_channel.npz`。

这个 sidecar 里关键字段包括：

```text
anchor_image_idx
anchor_xy_model
anchor_feature_yx
anchor_patch_center_xy
anchor_alignment_error_px
alignment_weight
point3D_id
target_image_idx
target_xy_model
track_flag
parallax_deg
colmap_reproj_error
track_length
track_xyz_world
```

这些字段用于训练时把某个 anchor feature cell 上预测出的 3D scene coordinate，投影到另一个 target frame 的像素位置，形成 inter-frame track reprojection loss。

## 5. GeoLMC global compression

GeoLMC 是方法第二核心。它把 large memory bank 压缩成固定数量的 latent scene tokens。

我们这里只关心 global 模式。

### 5.1 输入输出

输入：

```text
pooled_points:   B x N x 3
pooled_features: B x N x D
scene_center:    B x 3
optional all_scale_tokens
```

输出：

```text
latent_z:      B x K x d
latent_coords: B x K x 3
```

其中 K 是压缩后的 latent token 数，例如 64。

### 5.2 Latent coordinates

GeoLMC 不是直接平均 memory features，而是先在 `pooled_points` 上采样 K 个 latent coordinates。

当前 global 路径里：

- 如果 memory points 数量大于 K，则用 farthest point sampling；
- 默认起点策略是 `farthest_from_center`；
- 如果 memory points 数量不超过 K，则直接取前 K 个点。

这些 latent coordinates 是压缩器的空间锚点：

```text
P = {P_1, ..., P_K}
```

### 5.3 Query seed

GeoLMC 的 latent query 不是纯 learnable token，而是由几何坐标初始化：

```text
P_k - scene_center
  -> 3D positional encoding
  -> latent query seed
```

因此每个 latent token 从一开始就带有场景空间位置含义。

### 5.4 Key 和 Value 的不对称设计

GeoLMC 对 memory features 的使用是不对称的：

- key 来自 selected layer slice 或 scalar mix；
- value 来自 full concatenated memory features。

也就是说，attention 的匹配可以依赖一个更稳定或更语义化的 key feature，而 token 内容聚合时仍保留完整 feature 表达。

### 5.5 Geometry-aware attention

Global compressor 内部做 cross-attention：

```text
latent queries -> memory keys / values
```

并引入几何信息：

- latent coordinates 与 pooled points 之间的距离；
- distance-based geometric bias；
- 可选 RBF / CRPB / PointRoPE 等位置机制；
- local mode 不在当前主线中展开。

最终每个 latent token 聚合来自整个 scene memory bank 的信息，但聚合过程受 3D geometry 调制。

### 5.6 Scale token

如果 memory 文件提供 `all_scale_tokens`，GeoLMC 会把 view-level scale tokens 做平均，经过 MLP 后作为 global scene shift 加到 latent tokens 上。

这不是 token-wise 对齐，而是 scene-level 的全局调制。

## 6. Single memory-to-image fusion

Fusion 是方法第三核心。我们这里只关心 `single` 模式。

### 6.1 输入输出

输入：

```text
query_feats:  B x Nq x C
latent_z:     B x K x d
latent_p:     B x K x 3
scene_center: B x 3
```

输出：

```text
fused_feats: B x Nq x C
```

其中 `query_feats` 是当前 query image 的 dense backbone features；`latent_z, latent_p` 是压缩后的 scene tokens。

### 6.2 Fusion 机制

Single fusion 是一次 query-to-memory cross-attention：

```text
query dense features -> attend to latent scene tokens
```

具体理解：

- query 来自当前图像 dense features；
- key 来自 latent token features `Z`；
- value 默认是 `Z + PE(P - scene_center)`；
- 也就是说，latent token 的视觉内容和 3D 空间位置一起进入 value path；
- 默认 fusion geometry mode 是 value-only geometry；
- 输出与 query features 做 residual + FFN，得到 fused dense features。

因此 fusion 的作用是：让每个 query patch 根据自身视觉特征，从 scene-level compressed memory 中读取有用的场景先验。

### 6.3 Fusion 在图中的正确画法

图中应画成：

```text
Query Image -> Backbone -> Dense Query Features
                                    |
K Latent Scene Tokens --------------|
                                    v
                           Single LMC Fusion
                                    |
                             Fused Features
```

不要把 fusion 画成 retrieval，也不要画成直接输出 pose。它只是把 memory-conditioned 信息写回 dense features，后面仍然由 SCR coordinate head 预测 scene coordinates。

## 7. ACE-G 两阶段训练

ACE-G 是当前方法框架里的训练主线。它有两个阶段：

```text
S1:   train compressor + fusion + coordinate head
S2-G: freeze compressor, cache compressed memory, train coordinate head, optionally train fusion
```

我们的方法叙事应该围绕这个两阶段机制展开。

## 8. S1 阶段

S1 是训练 scene memory compressor 的主阶段。

### 8.1 S1 输入

S1 使用训练数据或 buffer 中的 raw backbone features：

- query image features；
- target pixel；
- GT pose inverse；
- intrinsics / inverse intrinsics；
- GT scene coords / valid mask，供 multi-frame 和 valid-coordinate 采样使用；
- 可选 SfM track sidecar 字段。

### 8.2 S1 前向流程

S1 的逻辑是：

```text
raw query features
  -> GeoLMC compressor(memory bank)
  -> single fusion(query features, latent memory)
  -> coordinate head
  -> predicted scene coordinates
  -> reprojection losses
```

在 S1 中：

- compressor 有梯度；
- fusion 有梯度；
- coordinate head 有梯度；
- backbone 通常冻结；
- memory bank 是固定输入，不是训练参数。

S1 的目标是让 compressor 学会把离线 memory bank 压缩成对当前 SCR 任务有效的 latent scene representation，同时让 fusion 学会把这个 representation 注入 query features。

### 8.3 S1 loss

S1 的核心监督仍然是 SCR 的 scene coordinate reprojection：

- 预测每个 sampled patch / pixel 的 3D scene coordinate；
- 用 GT pose inverse 和 intrinsics 投影回图像；
- 与 target pixel 比较；
- 对投影无效或深度异常的点加 invalid loss。

这里不是 feature-level alignment loss。即使 S1 有“memory alignment”的直觉，代码层面的主监督也是 reprojection / invalid coordinate loss。

## 9. S2-G 阶段

S2-G 是在压缩器已经通过 S1 学到有效 scene tokens 后，对 coordinate head 和最终 fused representation 做全局强化的阶段。

### 9.1 S2-G 关键行为

S2-G 开始前：

```text
compressor.eval()
no_grad compressor(memory)
cache latent scene tokens
```

也就是说：

- compressor 在 S2-G 中冻结；
- compressed memory tokens 缓存下来；
- S2-G 不再每个 batch 重新训练 compressor。

### 9.2 S2-G buffer

默认 ACE-G S2-G buffer 存的是 raw backbone features，不是 fused features。

每个 S2-G batch 中：

```text
raw backbone features
  -> single fusion with cached latent tokens
  -> coordinate head
  -> scene coordinates
  -> loss
```

如果使用 R2 设置，fusion 可以在 S2-G 中以较小学习率继续训练；但 compressor 仍然冻结。

### 9.3 S2-G 的角色

S2-G 的作用不是重新学习 memory compression，而是：

- 固定 scene-level compressed representation；
- 让 coordinate head 适应 memory-conditioned features；
- 在需要时微调 fusion；
- 通过更稳定的训练阶段提升最终 pose accuracy。

## 10. 单帧重投影 loss

单帧重投影是主 loss。

对每个训练样本：

1. 网络预测 scene coordinate：

```text
X_pred(u) in R^3
```

2. 用当前帧 GT inverse pose 把 world coordinate 变到 camera frame：

```text
x_cam = T_world_to_cam X_pred
```

3. 用 intrinsics 投影到像素平面：

```text
u_pred = K x_cam
```

4. 与当前 patch / pixel 的 target position 比较：

```text
L_reproj = rho(u_pred - u_target)
```

5. 对无效深度、非有限坐标、过远坐标等情况加入 invalid loss。

这个 loss 保持了 ACE/SCR 的核心训练范式：网络不直接回归 pose，而是回归 dense scene coordinates，并通过几何投影约束坐标质量。

## 11. 多帧 / track 重投影 loss

这里需要区分两个相关概念。

### 11.1 Multi-frame reprojection

Multi-frame reprojection 不依赖 COLMAP track 的一对一观测，而是利用训练集中已有的 GT scene coordinates 和 reference frame bank。

基本逻辑：

1. 对当前 batch 中有效的 GT scene coordinates 取样。
2. 随机选择若干 reference frames。
3. 把 predicted scene coordinate 和 GT scene coordinate 分别投影到这些 reference frames。
4. 只保留 GT 投影可见、深度有效、在图像范围内的项。
5. 比较 predicted projection 和 GT projection。

这个 loss 的含义是：当前图像预测出的 3D scene coordinate 不仅要能解释当前帧，也应能在其他训练视角下保持几何一致。

### 11.2 SfM / COLMAP track inter-frame reprojection

Track reprojection 更严格。它依赖 COLMAP track sidecar。

对某个 anchor frame 的 feature cell，sidecar 给出：

- anchor image index；
- anchor feature yx；
- anchor 像素坐标；
- 该 track 对应的 target image index；
- target frame 中的 2D observation；
- target frame pose 和 intrinsics；
- alignment weight；
- parallax、track length、COLMAP reprojection error 等质量指标。

训练时，网络在 anchor feature cell 预测一个 3D scene coordinate，然后把这个 3D 点投影到 target frame：

```text
X_pred(anchor cell)
  -> target camera using target pose
  -> target pixel using target intrinsics
  -> compare with COLMAP target observation
```

这个 loss 用 SfM tracks 作为跨帧几何约束，直接约束同一个 3D track 在不同图像中的投影一致性。

### 11.3 Guided track sampling

为了让 batch 中有足够的 track-supervised rows，可以启用 track-guided sampling。

它会从 buffer 中筛选满足条件的 rows：

- 有 source anchor pixel；
- 有有效 `point3D_id`；
- 有 target pixel；
- 有 target image index；
- 有正的 inter-frame weight。

然后按 append 或 balanced_replace 策略把这些 track rows 加入训练 batch。

这使 S2-G 训练时不仅看到普通随机 patch，也能看到带跨帧 track 监督的 patch。

## 12. 与 GLACE 的关系

在当前方法主线里，GLACE 不是我们要重点展开的模块。

我们只关心 GLACE 的原始 concat 方式：

```text
local / fused dense features
  + broadcast image-level global descriptor
  -> concat
  -> GLACE-style coordinate head
```

也就是说，GLACE 在这里体现为一种现有 SCR 方法 / head 形式。我们的方法给它提供 memory-conditioned local features，然后按 GLACE 原本的 concat 机制接入 global descriptor。

流程图里不需要展开：

- residual global head；
- FiLM head；
- global routing；
- GLACE 内部实现细节。

正确叙事是：

我们的方法是 scene memory compression + fusion；GLACE concat 只是证明这个 scene compressed representation 可以增强已有 GLACE-style SCR pipeline 的一个实例。

## 13. 推理流程

推理时：

1. 加载训练好的 LMC checkpoint。
2. 加载 scene memory bank。
3. 构建 GeoLMC compressor 和 fusion。
4. 对 memory bank 运行一次 compressor：

```text
memory bank -> cached latent scene tokens
```

5. 对每张 query image：

```text
query image
  -> backbone dense features
  -> single fusion with cached scene tokens
  -> coordinate head
  -> dense scene coordinates
  -> DSAC* / RANSAC
  -> camera pose
```

推理时不能画成每张 query 都重新构建 memory 或重新训练 compressor。

## 14. 流程图应该表达的主张

流程图的主张应该是：

我们从训练场景中构建一个带 3D 几何和视觉特征的 memory bank，并通过 GeoLMC global compressor 把它压成固定大小的 latent scene representation。这个 representation 通过 single fusion 注入 query features，使现有 SCR 方法在预测 scene coordinates 时获得场景级先验。训练上，ACE-G 的 S1 学习 compressor/fusion/head，S2-G 固定压缩器并强化最终 head/fusion；监督由单帧重投影和多帧 / track 重投影共同提供。

图中应优先画：

```text
(a) memory construction:
training images + poses + COLMAP sparse cloud/tracks -> pooled memory bank

(b) memory compression:
pooled points/features -> GeoLMC global compressor -> K latent scene tokens

(c) query fusion:
query image -> backbone -> dense features
latent scene tokens -> single fusion -> fused features

(d) SCR prediction:
fused features -> coordinate head -> scene coordinates -> DSAC*/RANSAC -> pose

(e) training:
S1 train compressor/fusion/head
S2-G freeze compressor, cache tokens, train head/fusion
single-frame reprojection + multi-frame/track reprojection
```

## 15. 流程图不应该表达的内容

不要把方法画成 GLACE 桥接论文。

不要把 residual / FiLM 作为主线。

不要把 local compressor 模式画进主图。

不要把 fusion 的 cascade / reread / dual / coord-prior 变体画进主图。

不要把 memory bank 画成 K tokens；K tokens 是压缩器输出。

不要把 S2-G 画成训练 compressor。

不要把 default S2-G buffer 画成 fused feature buffer。

不要把 loss 写成 feature alignment；主 loss 是几何重投影。

不要把 DSAC* / RANSAC 画成可学习模块。

不要把 memory construction 简化成“读取一个 .pt 文件”；论文图里应该体现这个 .pt 文件来自训练图像、pose、COLMAP sparse point cloud / sparse depth、feature extraction、feature-point correspondence 和 point cloud pooling。

## 16. 推荐图标题

中文：

基于场景压缩记忆的 SCR 增强框架

英文：

Scene-Compressed Memory for Enhanced Scene Coordinate Regression

或者：

Global Latent Scene Memory for ACE-G Relocalization

## 17. 推荐 caption 中文稿

我们首先从带位姿的训练图像和 COLMAP 恢复的稀疏点云中构建场景 memory bank。COLMAP 点云被投影 / 对齐到训练图像，形成 sparse depth 或 patch-level 3D point support；随后对图像提取 backbone features，并建立 feature 与 3D point 的对应关系，使每个 memory cell 包含 3D scene point 及其视觉特征。同一 COLMAP 流程还产生 track observations，用于构建多帧 track 重投影的 sidecar。GeoLMC 在 global 模式下从 memory points 中采样 K 个 latent anchors，并通过几何感知 cross-attention 将大规模 memory bank 压缩为固定大小的 latent scene tokens。对于每张 query image，backbone 产生 dense query features，single fusion 模块让这些 query features attend 到 cached latent scene tokens，从而得到 memory-conditioned features。随后标准 SCR coordinate head 预测 dense scene coordinates，并由 DSAC* / RANSAC 求解 6-DoF pose。训练采用 ACE-G 两阶段流程：S1 训练 compressor、fusion 和 head；S2-G 冻结并缓存 compressor 输出，继续训练 head 并可选微调 fusion。监督信号由单帧重投影和多帧 / COLMAP track 重投影组成，以约束预测坐标在当前视角和跨视角下的几何一致性。

## 18. 我当前理解中的最终方法定位

这项工作的核心定位不是“提出一个新的 pose solver”，也不是“提出一个 GLACE 变体”，而是：

提出一种场景级压缩 memory representation，使 SCR 模型能够在不显著改变下游 pose pipeline 的前提下，利用训练场景的多视角几何、重建点云和视觉特征。

这套 representation 的关键技术闭环是：

```text
scene memory construction
  -> GeoLMC global compression
  -> single query-memory fusion
  -> ACE-G S1/S2-G optimization
  -> single-frame + multi-frame/track geometric supervision
```

它可以落在不同 SCR backbone / head 上，所以实验可以覆盖 ACE、DINO-ACE、GLACE；但论文方法图应该突出通用 scene memory enhancement，而不是把注意力放在某一个下游 head 的连接细节上。
