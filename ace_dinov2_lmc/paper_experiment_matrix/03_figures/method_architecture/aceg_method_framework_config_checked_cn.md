# ACE-G 场景压缩表征方法框架理解稿（配置核验版）

日期：2026-06-29

本文档按实际训练配置反推方法图口径。目标不是罗列所有代码分支，而是抽取论文主线中两条重要实验分支的共同事实：

1. **DINO+MapAnything memory 分支**：Indoor6 早期最佳配置，核心是 `ace_dinov2 + MapAnything/BSE memory + ACE-G + GeoLMC global + single fusion + SCR head`。旧最佳没有使用 STGS。
2. **ACE-FCN/STGS/GLACE 分支**：Wayspots/Cambridge 等较新配置，核心是 `ACE-FCN memory + ACE-G + GeoLMC global + single fusion`，Stage1 可启用 STGS anchor-guided sampling，Stage2 用 GLACE 原始 `concat` head。

这两条线的共同主张是：我们设计了一种面向场景的压缩记忆表征，让不同 SCR backbone/head 可以从同一类 scene memory bank 中读取场景级几何和视觉先验。

## 1. 先给最终结论

方法图的主干应该画：

```text
Training images + poses + COLMAP sparse cloud/tracks
  -> sparse depth / patch-level 3D support
  -> backbone-specific feature extraction
  -> pooled scene memory bank {3D points, features}
  -> GeoLMC global compression
  -> K anchored latent scene tokens
  -> single query-memory fusion
  -> SCR coordinate head
  -> scene coordinates
  -> DSAC*/RANSAC pose
```

共同事实：

- 两条主线都使用 `lmc_flow=ace_g`。
- 压缩模块按论文主线只讲 `lmc_mode=global`。
- 压缩 token 数主线都是 `K=64`。
- fusion 主线只讲 `lmc_fusion_refinement_mode=single`。
- fusion geometry 默认/主线是 `lmc_fusion_geometry_mode=value_only_raw`。
- SCR head 仍预测 dense scene coordinates，pose 仍由 DSAC*/RANSAC 求解。
- 主监督是单帧 reprojection/invalid coordinate loss。

需要分支化写清楚的事实：

- `scale token`：DINO+MapAnything 旧主线启用 `use_scale_token=True`；ACE-FCN/STGS/GLACE 主线显式 `use_scale_token=False`。不要把 scale token 画成共同主图模块。
- `key/value 不对称`：多层 DINO/MapAnything memory 下成立；ACE-FCN memory 是单层 `layers_idx=["ace_fcn"]`，key/value 退化为同源特征。不要把它写成所有分支共同设计。
- `latent_coords`：代码里确实存在并使用，但论文图里更适合叫 `3D latent anchors / anchored latent scene tokens`。它不是额外监督项，也不是独立输出 head。
- `STGS`：不是 Indoor6 DINO 旧最佳的一部分。当前较新 STGS 主线主要启用 track-guided sampling + anchor self reprojection auxiliary loss；`inter-frame track reprojection loss` 路径存在，但 anchor 主配置里 `anchor_only` 且 `sfm_track_inter_frame_weight=0.0`，所以不要把它画成当前共同启用的 solid loss。

## 2. 两条主线的配置事实

### 2.1 DINO+MapAnything / Indoor6 旧最佳

证据锚点：

- `04_evaluation/paper_results_20260606/tables/indoor6_dinov2_full_baselines_4090_best_by_scene_20260606.tsv`
- `04_evaluation/paper_results_20260606/indoor6/indoor6_dinov2_train_compare_report_20260606.md`
- `scripts/run_indoor6_scene5_dino_oldbest_repro_gpu.sh`
- `dino_lmc_base/04_evaluation/.../run_config.json`

实际口径：

- `model_backend=ace_dinov2`。
- `train_preset=memory_compare_ace_g_v1` 或等价 ACE-G 配置。
- `lmc_flow=ace_g`。
- `lmc_mode=global`，但旧 scene5 最佳复现允许 visibility fallback 到 local；后续 force-global 诊断才把 fallback 关掉。
- `num_latent_tokens=64`。
- 旧最佳常见 `lmc_iterations=28`。
- `lmc_key_slice_idx=2`，`lmc_key_feature_mode=slice`。
- `use_scale_token=True`。
- memory 来自 MapAnything/BSE pooled memory，通常是多层/拼接 feature，并可带 `all_scale_tokens`。
- 旧最佳没有使用 STGS。

因此 DINO+MapAnything 分支可以支持这样的论文表述：

> 对多层 foundation features 构建的 scene memory，我们使用选定层作为 attention key，使用多层 concat feature 作为 value 内容，并可用 view-level scale tokens 做 scene-level 调制。

但这个表述不能泛化到 ACE-FCN 单层 memory。

### 2.2 ACE-FCN/STGS/GLACE 较新主线

证据锚点：

- `scripts/run_wayspots_ace_fcn_lmc_suite.sh`
- `scripts/launch_wayspots_bigbuf_it6_matrix_gpu23.sh`
- `scripts/launch_wayspots_fullflow_extra_queue_gpu23.sh`
- `scripts/launch_lawn_fullmethod_gpu0.py`

实际口径：

- Stage1 使用 `model_backend=ace_fcn_lmc`。
- `lmc_flow=ace_g`。
- `memory_path=.../memory_ace_fcn_sparse_sp_r4.pt`。
- `use_scale_token=False`。
- `ace_lmc_global_head_mode=none`。
- `lmc_fusion_refinement_mode=single`。
- `num_latent_tokens=64`。
- ACE-FCN memory 保存 `layers_idx=["ace_fcn"]`，是单层 memory。
- 较新 Stage1 可启用 STGS `anchor_v2_f010_w05`：
  - `use_sfm_track_guided_sampling=True`
  - `sfm_track_guided_sampling_strategy=balanced_replace`
  - `sfm_track_guided_fraction=0.10`
  - `sfm_track_guided_mode=anchor_only`
  - `sfm_track_guided_main_loss_mode=include`
  - `sfm_track_guided_source_target_mode=patch_center`
  - `sfm_track_guided_aux_normalizer=full_batch`
  - `sfm_track_anchor_self_weight=0.5`
  - `sfm_track_anchor_use_alignment_weight=True`
  - `sfm_track_inter_frame_weight=0.0`

Stage2 / GLACE concat：

- `ace_lmc_global_head_mode=glace_concat`。
- `ace_lmc_local_checkpoint_path=<stage1 checkpoint>`。
- `ace_lmc_freeze_local_stack=True`。
- `glace_feat_name=features.npy`。
- `lmc_fusion_refinement_mode=single`。
- `use_scale_token=False`。

因此 ACE-FCN/STGS/GLACE 分支可以支持这样的论文表述：

> 我们先用 ACE-FCN feature-space memory 训练 local LMC stack，再在 Stage2 冻结 local stack，用 GLACE 原始 concat head 接入 image-level global descriptor。STGS 只作为训练采样和几何监督增强，不改变 memory compression/fusion 的核心接口。

## 3. Memory construction 的统一口径

两条分支都应统一讲成：

```text
COLMAP sparse reconstruction
  -> sparse 3D points + tracks
  -> sparse depth / patch-level 3D support
  -> feature extraction
  -> feature-point correspondence
  -> pooled scene memory bank
```

共同输入：

- training RGB images；
- camera pose；
- camera intrinsics；
- COLMAP sparse 3D point cloud；
- COLMAP tracks / observations；
- 由 COLMAP 点云投影或对齐得到的 sparse depth / patch-level 3D support；
- backbone-specific dense/patch features。

共同输出：

```text
pooled_points:   N x 3
pooled_features: N x D
scene_center:    3
all_poses / all_intrinsics
optional metadata
```

差异只在 feature source：

- DINO+MapAnything memory：foundation / MapAnything feature，通常多层，可能带 `all_scale_tokens`。
- ACE-FCN memory：ACE-FCN encoder feature，`feature_dim=512`，`layers_idx=["ace_fcn"]`，不使用 scale token。

论文图里，memory bank 应画成：

```text
Pooled Scene Memory Bank
{3D points p_i, visual features f_i}
```

不要把 memory bank 画成 K 个 tokens；K tokens 是 GeoLMC 压缩后的结果。

## 4. Track / STGS 前处理的正确口径

COLMAP 同时给出 sparse 3D points 和 tracks。STGS sidecar 是从同一套 reconstruction 中把 tracks 对齐到 SCR feature grid。

`tools/build_colmap_keyframe_channel.py` 保存的核心字段包括：

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
target_feature_yx
target_patch_center_xy
parallax_deg
colmap_reproj_error
track_length
track_xyz_world
```

这个 sidecar 支持两类用途：

1. 当前 STGS anchor 主线：从 track rows 中采样高质量 anchor cells，用 anchor self reprojection auxiliary loss 强化训练。
2. 可选 inter-frame track reprojection：把 anchor cell 预测的 3D coordinate 投影到 target frame，与 COLMAP target observation 比较。

当前较新主干 `anchor_v2_f010_w05` 属于第 1 类。它不启用第 2 类 inter-frame loss，因为 `sfm_track_guided_mode=anchor_only` 且 `sfm_track_inter_frame_weight=0.0`。

因此流程图建议：

```text
COLMAP tracks -> STGS sidecar -> track-guided sampling / anchor self reprojection
```

如果图中要画 target-frame track reprojection，应该标成 optional 或 dashed branch，不能画成所有主线共同启用项。

## 5. GeoLMC global compression

### 5.1 输入输出

输入：

```text
pooled_points:   B x N x 3
pooled_features: B x N x D_total
scene_center:    B x 3
optional all_scale_tokens
```

输出：

```text
latent_z: B x K x d
latent_p: B x K x 3
```

代码里 `latent_p` 叫 `latent_coords`。论文图中建议叫 `3D anchors` 或 `anchored latent scene tokens`。

### 5.2 latent coordinates / 3D anchors 的真实作用

它们不是未使用项。global 路径会：

1. 从 `pooled_points` 里 FPS 采样 K 个 3D anchors。
2. 用 `latent_p - scene_center` 做 positional encoding，初始化 latent query seed。
3. 计算 `latent_p` 与所有 memory points 的距离，给 geometry-aware attention 使用。
4. 把 `(latent_z, latent_p)` 一起传给 fusion。
5. fusion 在 `value_only_raw` 下用 `latent_p - scene_center` 的 Fourier PE 注入 value path。

因此正确说法是：

> GeoLMC 输出带 3D anchor 的 latent scene tokens。

不建议说：

> 我们没有使用 latent coordinates。

也不建议在图里单独画：

> latent coordinates head / latent coordinate supervision。

它是 token 的几何锚点，不是独立预测任务。

### 5.3 key/value 设计

共同模块名可以写 `selected-key / concat-value compression`，但解释必须分支化：

- DINO+MapAnything 多层 memory：`key` 使用 selected layer slice，实际常用 `lmc_key_slice_idx=2`；`value` 使用 full concatenated memory features。这时 key/value 不对称是事实。
- ACE-FCN 单层 memory：`num_layers=1`，`resolved_key_slice_idx=0`，key slice 就是完整单层 ACE-FCN feature；value 也来自同一单层 feature 的投影。因此没有有意义的 key/value 不对称。

图中建议只写：

```text
Geometry-aware memory attention
```

或：

```text
Selected-key / concat-value
(multi-layer memory)
```

不要把 key/value 不对称画成所有分支的核心创新点。

### 5.4 scale token

`scale token` 是分支项，不是共同项。

- DINO+MapAnything 旧主线：`use_scale_token=True`，如果 memory 有 `all_scale_tokens`，GeoLMC 会对 view-level tokens 做 mean pooling，经 MLP 后加到 latent tokens。
- ACE-FCN/STGS/GLACE 主线：训练脚本显式 `use_scale_token=False`，即使代码支持，也不应画入该分支主图。

推荐论文主图不画 scale token。若需要在 supplement 或分支图里说明 DINO+MapAnything，可加小注：

```text
optional view-level scale token (DINO/MapAnything only)
```

## 6. Single memory-to-image fusion

两条主线共同使用 single fusion。

输入：

```text
query_feats: B x Nq x C
latent_z:    B x K x d
latent_p:    B x K x 3
scene_center:B x 3
```

机制：

```text
query dense features -> attend to latent scene tokens
```

在默认 `value_only_raw` 下：

- key 来自 `latent_z`；
- value 来自 `latent_z + PE(latent_p - scene_center)`；
- query 来自当前图像 dense features；
- 输出经过 residual + FFN 得到 fused dense features。

所以 fusion 的论文描述应该是：

> query features perform a single cross-attention read from anchored latent scene tokens.

不要画成 retrieval，也不要画成直接输出 pose。

## 7. ACE-G 训练流程

共同训练范式是 ACE-G 的 S1 / S2-G。

### 7.1 S1

S1 训练：

- GeoLMC compressor；
- single fusion；
- coordinate head。

输入是训练 buffer 或 online feature，对每个 sampled patch/pixel 预测 scene coordinate，再做 reprojection loss。

### 7.2 S2-G

S2-G 开始时：

```text
compressor.eval()
no_grad compressor(memory)
cache latent scene tokens
```

S2-G 训练：

- compressor 冻结；
- cached latent scene tokens 固定；
- coordinate head 继续训练；
- fusion 可按 `ace_g_fusion_in_s2=True` 以较小 LR 继续训练。

ACE-FCN/GLACE Stage2 还会：

- 加载 Stage1 local checkpoint；
- 冻结 local stack；
- 用 `glace_concat` 接 image-level global descriptor；
- 继续训练最终 GLACE-style head/fusion。

## 8. Loss 与监督口径

### 8.1 共同主 loss：单帧 reprojection

这是所有主线最稳的共同事实。

```text
pred scene coordinate
  -> current camera
  -> current pixel
  -> compare with sampled target pixel
```

无效深度、非有限坐标、过远坐标走 invalid loss contract。

### 8.2 Multi-frame reprojection

代码支持 `use_multiframe_reprojection_loss`，但默认关闭。当前我查到的主线配置和 run_config 多数是：

```text
use_multiframe_reprojection_loss=False
```

曾有 quick-fusion 中间实验启用过：

```text
--use_multiframe_reprojection_loss True
```

因此它不能写成两条主线共同启用的事实。可以写成可选几何监督分支或已探索变体。

### 8.3 SfM/COLMAP track inter-frame reprojection

代码支持 `use_sfm_track_inter_frame_loss` 和 `sfm_track_inter_frame_weight`。但当前 STGS anchor 主线实际设置是：

```text
sfm_track_guided_mode=anchor_only
sfm_track_inter_frame_weight=0.0
use_sfm_track_inter_frame_loss=False  # DINO-STGS 脚本显式如此
```

所以当前主线不是直接优化 target-frame inter-frame reprojection。

### 8.4 当前 STGS anchor 主线实际启用的 loss

当前启用的是 track-guided sampling + anchor self reprojection auxiliary loss：

```text
COLMAP track rows
  -> balanced_replace 10% batch rows
  -> source/anchor patch-center target
  -> current-frame reprojection auxiliary loss
  -> alignment-weighted, weight=0.5
```

这仍然利用了 COLMAP track 选择高质量几何监督样本，但不是 target-frame projection loss。

## 9. 与 GLACE 的关系

GLACE 不是主方法贡献本体，只是一个下游 SCR head/backbone 实例。

我们只关心原始 concat：

```text
memory-conditioned local features
  + broadcast GLACE global descriptor
  -> concat
  -> GLACE-style coordinate head
```

不要在主图展开：

- residual global head；
- FiLM；
- global routing；
- stage1_fused feature source；
- 其他桥接实验。

## 10. 方法图应采用的语义层级

推荐画成三栏或三段：

### (a) Offline Memory Construction

```text
Training RGB + poses + intrinsics
COLMAP sparse cloud + tracks
  -> sparse depth / patch 3D support
  -> feature extraction
  -> pooled memory bank {p_i, f_i}
```

### (b) Scene Memory Compression

```text
pooled memory bank
  -> FPS K 3D anchors
  -> geometry-aware GeoLMC
  -> anchored latent scene tokens
```

### (c) Query Inference / Training

```text
query image -> backbone dense features
anchored latent scene tokens -> single fusion
fused features -> SCR coordinate head
scene coordinates -> DSAC*/RANSAC pose
```

训练 band：

```text
S1: train compressor + fusion + head
S2-G: cache tokens, freeze compressor, train head/fusion
single-frame reprojection
STGS anchor self reprojection / optional track reprojection
```

## 11. 主图不要画的内容

不要画 local/hierarchical/learned LMC。

不要画 cascade/reread/dual/coord-prior fusion。

不要把 scale token 画成共同模块。

不要把 key/value 不对称画成 ACE-FCN 分支也成立。

不要把 latent coordinates 画成单独预测 head。

不要把 inter-frame track reprojection 画成当前 STGS anchor 主线已启用的 solid loss。

不要把 GLACE bridge 画成方法重点。

不要把 memory bank 画成 K tokens。

不要把 S2-G 画成继续训练 compressor。

## 12. 最终论文故事

我们的方法可以这样讲：

> We introduce a scene-compressed memory representation for enhancing scene coordinate regression. From posed training images and COLMAP reconstruction, we build a pooled scene memory bank that aligns 3D points with backbone-specific visual features. A geometry-aware GeoLMC compressor converts this large memory into a fixed number of anchored latent scene tokens. Query image features read from these tokens through a single cross-attention fusion block before the standard SCR coordinate head predicts dense scene coordinates. The same interface supports DINO+MapAnything memory and ACE-FCN/GLACE memory. STGS provides a track-aware training extension by using COLMAP tracks to sample reliable anchor rows and, when enabled, to define inter-frame geometric supervision.

中文版本：

> 我们提出一种面向场景的压缩 memory 表征，用来增强现有 scene coordinate regression 方法。方法首先从带位姿的训练图像和 COLMAP 重建中构建 pooled scene memory bank，使 3D 点与不同 backbone 的视觉特征对齐。随后 GeoLMC 在 global 模式下把大规模 memory 压缩为固定数量的、带 3D anchor 的 latent scene tokens。查询图像的 dense features 通过一次 single cross-attention 从这些 scene tokens 中读取场景先验，再由标准 SCR coordinate head 预测 dense scene coordinates，并用 DSAC*/RANSAC 求解 pose。这一接口同时适用于 DINO+MapAnything memory 与 ACE-FCN/GLACE memory；STGS 则作为基于 COLMAP tracks 的训练增强，用于采样可靠 anchor rows，并在启用时提供跨帧几何监督。

