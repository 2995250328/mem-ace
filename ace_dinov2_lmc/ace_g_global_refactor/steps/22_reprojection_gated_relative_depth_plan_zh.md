# Step 22：重投影门控相对深度监督与稀疏深度补全规划

状态：`[planning]`

本文档汇总当前相对深度损失的代码审计、文献调研和后续讨论，作为后续实现、消融和验收的统一依据。本文档描述规划，不代表相关改动已经实现或实验结论已经成立。

## 1. 目标与非目标

当前场景坐标回归会产生大量不准确的 3D 点。直接把所有预测点变换到相机坐标系，再用单目相对深度教师监督，容易让异常场景坐标主导辅助损失。

目标：

1. 先用几何一致性筛选或加权预测点；
2. 在几何可用点上施加与教师表示匹配的 relative-disparity loss；
3. 有可靠稀疏 metric-depth anchor 时，将稀疏几何和稠密单目先验融合成带置信度的稠密伪标签；
4. 评估外部 depth completion 是否比简单、可解释的教师仿射校准更有价值；
5. 保持旧行为可复现，所有新路径默认关闭或有明确 legacy 开关。

非目标：

- 不把同视图低重投影误差等同于深度正确；
- 不直接把当前学生自身生成的稀疏深度补全后作为无条件真值；
- 不在第一版同时修改主 ReproLoss、fusion、compressor 和 relative-depth loss；
- 不在没有置信度和受控消融的情况下引入大型在线 diffusion completion 模型。

## 2. 当前实现审计

主要代码：

- `relative_depth_distillation.py`
  - `SampledRelativeDepthLoss`
  - `ImageRelativeDepthLoss`
  - `RelativeDepthDistiller`
- `trainer_dinov2_lmc.py`
  - `_compute_reprojection_invalid_loss_contract(...)`
  - `_build_relative_depth_image_dataloader()`
  - `_compute_image_relative_depth_loss(...)`
- `options_dinov2_lmc.py`
  - `--use_relative_depth_loss`
  - `--relative_depth_*`

当前完整图像路径为：

```text
独立无增强 RGB batch
  -> frozen encoder features
  -> LMC / GLACE fusion
  -> scene-coordinate head
  -> predicted world coordinates
  -> GT world-to-camera pose
  -> camera-space z
  -> Depth Anything V2 dense relative-depth teacher
  -> per-image robust normalization
  -> SSI + single-scale gradient + random pair difference
```

当前训练策略：

- relative-depth 默认关闭；
- 默认只在 S2 / S2-G 使用；
- 默认从当前 S2 phase 的 30% 位置开始线性 ramp；
- 默认每 10 个 S2 buffer step 运行一次完整图像辅助 batch；
- 支持从 checkpoint 进入 `relative_depth_only` 微调；
- teacher depth 使用 CPU half LRU cache。

## 3. 已识别问题

### 3.1 学生和教师不在同一表示空间

Depth Anything V2 relative model 提供的是近大远小的相对 disparity / proximity 表示。当前实现把相机深度 `z` 取负后与教师归一化值比较：

```text
student_rel = normalize(-z)
teacher_rel = normalize(D_teacher)
```

负号只修正单调方向，不能修正 `z` 与 inverse depth 之间的非线性差异。优先候选应为：

```text
d_student = 1 / clamp(z, z_min, z_max)
```

然后在 disparity 空间做 scale-and-shift invariant 对齐。

### 3.2 学生输出参与有效掩码，存在 mask collapse 风险

当前相对深度有效点要求学生深度为正且小于上限。错误的负深度或极大深度会直接离开辅助损失集合。在 `relative_depth_only` 阶段尤其可能出现“减少有效点即可降低辅助损失”的退化路径。

必须把以下概念分开：

- `image_valid`：由输入 mask、teacher finite 等外部条件定义；
- `ray_valid`：由投影一致性和相机空间合法性定义；
- `depth_anchor_valid`：由独立 metric / multi-view 几何证据定义；
- `student_depth_valid`：仅用于诊断或 invalid-depth penalty，不能静默决定全部监督是否消失。

### 3.3 当前梯度项只有单尺度

当前只比较相邻输出单元的横纵差分。MiDaS 类 SSI 方法在 disparity residual 上使用多尺度梯度匹配，以同时约束大尺度结构和边界。后续应至少支持 1/2/4/8 四个尺度的受控消融。

### 3.4 当前 pair loss 不是 ordinal ranking

当前随机点对项回归归一化 disparity difference 的数值。对于单目教师，差值幅度未必可靠；Depth Anything V2 的 DA-2K 相对深度评估本身采用“哪个点更近”的 pairwise ordinal annotation。

候选改法：

- baseline 先关闭 pair 项，仅保留 robust SSI；
- 或只对教师差值超过置信阈值的点对使用 ordinal logistic loss；
- 对局部点对和远距离点对分层采样，避免全局随机点对被大平面主导。

### 3.5 辅助损失 duty cycle 与权重含义不清

默认每 10 step 执行一次、执行时权重为 0.05。按长期更新频率估算，平均影响接近逐步权重 0.005，但瞬时梯度仍是 0.05。

后续配置必须明确 `relative_depth_loss_weight` 是 active-step weight，还是希望保持的 time-averaged task weight，两种语义不能混用。

### 3.6 图像变换对齐仍需验证

teacher 路径按目标高度保持宽高比 resize 原始 BGR，学生路径由 dataset 的 image height/width、可能的 crop/pad 和输出 stride 决定。不能只根据最终 `H x W` 插值就假设二者像素坐标严格一致。

实现前必须增加一次可视化或数值检查：RGB 边缘、teacher depth 边缘、输出网格中心、image mask 和重投影有效点。

## 4. 关键几何边界：同视图重投影不能验证深度

对预测世界点 `X_i`、GT world-to-camera pose `T_i`、内参 `K_i` 和目标像素 `p_i`，定义：

```text
r_i = || project(K_i T_i X_i) - p_i ||_2
```

低 `r_i` 只说明 `X_i` 接近像素 `p_i` 对应的相机射线。沿该射线移动 `X_i` 不会显著改变投影，因此错误深度也可能得到接近零的同视图重投影误差。

```text
low same-view reprojection error
    => ray consistency
    != depth correctness
```

同视图重投影可以做 `ray-valid` gate，但不能单独产生 `depth_anchor_valid`。

更强的深度证据包括：

1. RGB-D / LiDAR / 数据集稀疏深度；
2. SfM/MVS scene point 或 track；
3. 同一 3D 点在另一已知位姿图像上的 cross-view reprojection；
4. 与冻结参考模型、多视图模型或独立几何来源的一致性；
5. 在 teacher disparity 仿射对齐中的 robust inlier 身份。

其中 cross-view reprojection 能通过第二条射线约束点在第一条射线上的位置，明显强于同视图 gate。

## 5. 推荐总体设计

### 5.1 Level A：外部图像有效性

```text
M_image = image_mask
          & finite(teacher)
          & teacher_in_supported_range
```

该 mask 与学生预测无关。

### 5.2 Level B：同视图 ray-consistency gate

复用 `_compute_reprojection_invalid_loss_contract(...)` 中的坐标变换和投影约定，但不要直接复用主损失的 hard-clamp policy 作为唯一阈值。

硬门控候选：

```text
M_ray = M_image
        & finite(pred_cam)
        & z_min < z < z_max
        & reprojection_error_l2 < tau_px
```

更平滑的候选为 detached soft confidence：

```text
w_repro = exp(-(reprojection_error_l2 / sigma_px) ** 2).detach()
```

约束：

- hard mask 和 soft weight 均停止梯度；
- 记录 mask 前后覆盖率；
- coverage 低于最小阈值时跳过该图，而不是返回看似正常的零损失；
- 可提供 top-quantile fallback，但必须记录触发次数；
- 阈值单位统一为输入 RGB 像素，不使用输出 feature-grid 单元。

建议第一轮从 `tau_px = 2` 或 `4` 的单变量消融开始，不直接写死最终值。

### 5.3 Level C：可靠 metric-depth anchor

```text
M_anchor = M_ray & M_independent_depth_evidence
```

优先级建议：

1. dataset GT/RGB-D depth；
2. SfM/MVS / cross-view track depth；
3. 已有 RIO10 sparse depth；
4. 冻结参考 checkpoint 与教师共同筛出的 robust pseudo anchor；
5. 仅同视图低重投影预测点。

第 5 类不能独立用于生成无条件 dense metric target，只能作为低置信度候选或 relative ordinal 样本。

### 5.4 Level D：相对深度损失

第一版推荐：

```text
d_student = 1 / clamp(z_student, z_min, z_max)
d_teacher = raw Depth Anything V2 output
```

对 `M_ray` 或其 detached confidence 加权后，分别做 robust scale/shift normalization：

```text
center(d) = median(d)
scale(d)  = mean(abs(d - center(d)))
d_norm    = (d - center(d)) / max(scale(d), eps)
```

损失候选：

```text
L_rel = L_ssi_trimmed
      + lambda_grad * L_multiscale_gradient
      + lambda_ord  * L_confident_ordinal
      + lambda_inv  * L_invalid_depth
```

- `L_ssi_trimmed`：丢弃 residual 最大的一定比例，减弱教师错误和动态/反射区域影响；
- `L_multiscale_gradient`：在 disparity residual 的多个尺度上计算梯度；
- `L_confident_ordinal`：只使用教师 disparity 差值超过阈值的点对；
- `L_invalid_depth`：在外部有效区域内惩罚负深度、过近和过远预测，防止 mask collapse。

不得让 `M_ray` 对主网络反向传播。

## 6. 稀疏点到稠密深度的推荐路线

### 6.1 首选：教师 disparity 的稀疏 metric 校准

Depth Anything 已经提供稠密、边缘感知的 relative disparity。当前任务通常不需要再训练一个网络“补形状”，而是需要用可靠稀疏点为稠密相对表示提供 metric anchor。

在 `M_anchor` 上拟合：

```text
a * D_teacher(i) + b ~= 1 / z_anchor(i)
```

要求：

- 约束 `a > 0`；
- 使用 RANSAC、Huber IRLS 或 trimmed least squares；
- 检查 anchor 数量、空间覆盖和拟合残差；
- 不足时退回 sparse relative loss，不生成 dense pseudo target。

然后：

```text
dense_disparity = clamp(a * D_teacher + b, d_min, d_max)
dense_depth     = 1 / dense_disparity
```

稠密置信度应由 anchor fitting residual、anchor 空间支撑、teacher 边缘/不确定区域和 completion/alignment 模型自身置信度共同决定。

这一方案计算便宜、容易离线缓存、与 relative-depth teacher 的表示一致，优先于直接引入第二个大模型。

### 6.2 可选：外部 RGB-guided depth completion

候选方法：

- NLSPN：非局部空间传播并显式预测 confidence；
- CompletionFormer：CNN + Transformer 的 RGB/sparse-depth completion；
- Marigold-DC：zero-shot diffusion depth completion，适合离线生成而非训练内环。

使用前提：

- 输入必须是可靠 metric anchor，而不是仅低同视图重投影的学生预测；
- completion 网络冻结；
- completion 输出和 confidence 停止梯度；
- 优先离线或每个 checkpoint/cycle 刷新，不在每 10 个训练 step 在线运行；
- completion 结果必须保留原始 anchor，并对无 anchor 支撑区域降低权重；
- 单独报告 completion 域差异：KITTI/NYUv2 训练的模型未必适合 Indoor6、Wayspots 或 RIO10。

禁止的闭环：

```text
current student prediction
  -> same-view reprojection filter
  -> unconditional depth completion
  -> dense pseudo ground truth
  -> supervise the same current student
```

如必须使用学生预测作为 sparse input，至少需要使用冻结历史 checkpoint、离线固定结果、hold-out anchor 验证、confidence-weighted loss，并与 sparse-only baseline 对照。

## 7. 训练接线方案

### 7.1 完整图像 batch 新增输入

`_compute_image_relative_depth_loss(...)` 需要明确取得：

- `K_B33` 和 `K_inv_B33`；
- 与输出 `H x W` 对齐的目标像素中心网格；
- `gt_pose_inv_B34`；
- `image_mask_B1HW`；
- 可选 sparse/GT depth 与其 mask；
- image identity，用于 teacher/completion cache。

目标像素网格必须复用 ACE dataset / regressor 的 output-subsample 和 pixel-center 约定，不能自行假设 offset。

### 7.2 与主 ReproLoss 的关系

主 ReproLoss 继续优化全部 sampled/buffer 点，并保留其 invalid proxy policy。relative-depth gate 只是辅助损失的样本选择，不改变主损失行为。

```text
L_total = L_reprojection_existing
        + lambda_rel(t) * L_relative_depth_gated
        + other_existing_losses
```

relative-depth gate 应复用投影数学，但拥有独立、可审计的阈值与统计，避免修改主 ReproLoss hard clamp 时悄悄改变辅助监督集合。

### 7.3 调度

建议同时满足以下条件后才启用：

```text
phase_ratio >= configured_start_ratio
and reprojection_valid_coverage >= min_coverage
and valid_image_count > 0
```

可选 curriculum：先启用 inverse-depth SSI，稳定后加入多尺度梯度，最后加入 confident ordinal 或 dense completion target。不要在同一次实验中同时改变表示、gate、completion 和 loss 权重。

## 8. 建议 CLI

以下为规划名称，实施时应与现有命名保持一致：

```text
--relative_depth_student_space neg_z_legacy|inverse_depth
--relative_depth_reprojection_gate none|hard|soft
--relative_depth_reprojection_threshold_px 2.0
--relative_depth_reprojection_sigma_px 2.0
--relative_depth_min_coverage 0.05
--relative_depth_coverage_fallback skip|top_quantile
--relative_depth_trim_ratio 0.0
--relative_depth_gradient_scales 1
--relative_depth_gradient_weight 1.0
--relative_depth_pair_mode difference_legacy|none|ordinal
--relative_depth_ordinal_margin 0.0
--relative_depth_invalid_weight 0.0
--relative_depth_dense_target none|teacher_affine|completion_offline
--relative_depth_anchor_kind none|gt_depth|sparse_depth|sfm|cross_view|frozen_student
--relative_depth_completion_root /path/to/cache
```

兼容要求：

- legacy checkpoint / recipe 能显式选择旧 `neg_z + single-scale + pair-difference` 行为；
- 新默认值只在 `--use_relative_depth_loss True` 时生效；
- checkpoint metadata 保存所有 relative-depth 配置；
- resume 时检查 teacher、gate、completion cache contract 是否一致。

## 9. 必须新增的日志

```text
relRaw             raw relative-depth loss
relW               active-step task weight
relSSI             SSI component
relGrad            multiscale gradient component
relOrd             ordinal component
relInvalid         invalid-depth component
relImageCoverage   external image-valid coverage
relRayCoverage     reprojection-gated coverage
relAnchorCoverage  independent metric-anchor coverage
relReproMean       selected-point reprojection error
relReproP90        selected-point reprojection p90
relDepthInvalid    invalid student-depth ratio
relGroups          contributing image count
relSkipped         skipped image/update count and reason
relCompletionConf  mean completion confidence when enabled
relAffineResidual  teacher-to-anchor fit residual
```

必须区分 `loss=0 because perfect` 与 `loss=0 because no valid points`。

## 10. 实施阶段

### Phase 0：测量和对齐验证

- 增加不改变 loss 的 reprojection/coverage 诊断；
- 验证完整图输出网格中心、内参、mask 和 teacher resize 对齐；
- 导出若干 overlay；
- 统计不同阈值下的 coverage、深度分布和 teacher ordinal agreement。

退出条件：确认像素坐标 contract，无 silent misalignment。

### Phase 1：表示空间修正

- 增加 `inverse_depth`；
- 保留 `neg_z_legacy`；
- 先使用 SSI-only 或关闭 legacy pair；
- 增加 invalid-depth/coverage 日志。

退出条件：synthetic test 验证 affine disparity invariance、方向和梯度有限。

### Phase 2：reprojection-gated sparse relative loss

- 完整图网格上计算同视图 reprojection；
- hard/soft detached gate；
- coverage guard；
- pair 只从 gate 内采样。

退出条件：gate 不参与反向传播，低 coverage 不伪装为正常零损失。

### Phase 3：多尺度与 ordinal 消融

- 1/2/4/8-scale residual gradient；
- teacher-confidence ordinal pair；
- local/global stratified pair sampling。

退出条件：各分量量级和梯度范数可解释，无单项长期主导。

### Phase 4：可靠 anchor 与 teacher affine densification

- 接入 GT/sparse/SfM/cross-view anchor；
- robust fit `a,b`；
- dense pseudo disparity + confidence cache；
- sparse-only fallback。

退出条件：held-out anchor 上的拟合误差和方向正确，cache provenance 完整。

### Phase 5：外部 completion 对照

- 优先离线 Marigold-DC 或与数据域匹配的 CompletionFormer/NLSPN；
- 输出 dense depth 和 confidence；
- 与 teacher-affine 和 sparse-only 做单变量比较。

只有在外部 completion 明显优于更简单方案时，才考虑长期集成。

## 11. 测试计划

### 11.1 单元/合成测试

1. 完美射线、错误深度：同视图 reprojection 应接近零，用于证明 gate 的能力边界。
2. 像素横向偏移：reprojection gate 应拒绝。
3. 负深度、极大深度、NaN/Inf：loss 和 stats 必须有限或显式跳过。
4. teacher disparity 做正 scale + shift：SSI loss 应保持不变或接近不变。
5. 学生深度整体 metric scale 改变：inverse-disparity SSI 应保持预期不变。
6. mask 全空和 coverage 过低：必须返回明确 skip reason。
7. gate tensor 不得要求梯度；mask 选择不得反传。
8. affine densification：已知 `a,b` 的合成数据应能恢复。
9. completion cache image identity、shape 和 preprocessing hash 必须匹配。

### 11.2 最小 smoke

- 单图 forward/backward；
- relative-depth-only 10--50 steps；
- 确认 head/fusion 梯度存在且 teacher/completion 无梯度；
- 确认 GPU memory 不随 cache hit/miss 增长；
- 分别覆盖 ACE、ACE-G 和实际使用的 GLACE backend。

### 11.3 实验矩阵

保持相同数据、checkpoint、seed、S2 steps 和 optimizer：

| ID | Student space | Gate | Dense target | Loss |
|---|---|---|---|---|
| A0 | `neg_z_legacy` | none | none | current legacy |
| A1 | inverse depth | none | none | robust SSI |
| A2 | inverse depth | repro hard | none | robust SSI |
| A3 | inverse depth | repro soft | none | SSI + multiscale grad |
| A4 | inverse depth | repro soft | none | A3 + confident ordinal |
| B1 | inverse depth | repro soft | teacher affine | confidence-weighted dense SSI |
| B2 | inverse depth | repro soft | offline completion | confidence-weighted dense SSI |

如果没有独立 metric anchor，不运行 B1/B2，不能用同视图低重投影学生点冒充可靠 anchor。

## 12. 评估指标和验收

主指标仍是相机重定位：

- median translation / rotation error；
- 5 cm / 5 deg、10 cm / 5 deg 等既有成功率；
- DSAC/RANSAC inlier 数与 pose failure rate。

辅助诊断：

- teacher pair WHDR / ordinal agreement；
- reprojection-gated coverage；
- invalid-depth ratio；
- selected vs rejected 点的 reprojection/depth 分布；
- sparse anchor held-out error；
- dense pseudo target 的 confidence-calibrated error；
- relative loss 各分量和梯度范数；
- 训练时间、GPU 显存和 teacher/completion 开销。

接受原则：

- 不以 relative loss 下降作为成功标准；
- pose 指标不得因更多伪监督而系统性退化；
- 改进必须跨至少多个场景或明确给出适用边界；
- completion 必须相对 sparse-only 和 teacher-affine 提供额外收益，才能证明其复杂度合理。

## 13. 风险与回退

### 风险：同视图 gate 选中沿射线错误点

回退：把 gate 仅用于 ray consistency；metric densification 必须依赖独立 anchor。

### 风险：早期 coverage 太低

回退：延后 relative loss、使用 soft confidence，或从稳定 checkpoint 做 relative-depth-only 微调。

### 风险：teacher 在反射、透明、动态区域错误

回退：trimmed SSI、teacher confident ordinal、语义/边缘 confidence 和 completion confidence。

### 风险：completion 传播错误 anchor

回退：保留 anchor、使用冻结离线模型、hold-out validation，并退回 teacher-affine 或 sparse-only。

### 风险：辅助 loss 与主 ReproLoss 冲突

回退：降低 active-step/time-averaged weight，使用 gradient norm 诊断，最后才考虑 PCGrad/GradNorm 等复杂多任务方法。

### 风险：teacher/student preprocessing 错位

回退：Phase 0 未通过时禁止启动训练实验。

## 14. 推荐执行顺序

```text
P0  像素网格与 preprocessing 对齐验证
P1  完整图 reprojection/coverage 只读诊断
P2  inverse-disparity SSI
P3  detached reprojection gate
P4  multi-scale gradient / confident ordinal ablation
P5  independent sparse/cross-view metric anchors
P6  teacher affine densification
P7  external completion comparison
```

在 P0--P4 得到清晰结果前，不进入外部 depth completion 集成。

## 15. 参考资料

- Depth Anything V2：<https://github.com/DepthAnything/Depth-Anything-V2>
- DA-2K pairwise relative-depth benchmark：<https://github.com/DepthAnything/Depth-Anything-V2/blob/main/DA-2K.md>
- MiDaS / scale-and-shift invariant disparity loss：<https://openreview.net/pdf?id=D5FDlNsVU>
- NLSPN：<https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123580120.pdf>
- CompletionFormer：<https://github.com/youmi-zym/CompletionFormer>
- Marigold-DC：<https://github.com/prs-eth/Marigold-DC>

