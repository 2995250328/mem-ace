# ACE-G / LMC 重构中文总览

本文档是 `ace_g_global_refactor/` 下英文规划文档的中文总览版，目标不是替代
`PLAN.md`、`steps/` 和 `LONG_TERM_RESEARCH_PLAN.md`，而是用中文明确三件事：

1. 现阶段已经完成了什么；
2. 现阶段真正应该进入 LMC 架构修改主线的是什么；
3. 哪些想法重要，但应当归入长期系统规划，而不是混进当前 baseline 改动。

相关原始文档：

- 高层计划：`ace_g_global_refactor/PLAN.md`
- 近期架构步骤：`ace_g_global_refactor/steps/`
- 长期系统规划：`ace_g_global_refactor/LONG_TERM_RESEARCH_PLAN.md`

## 1. 当前基线的定位

当前 ACE-G 训练链条，本质上仍然是：

- 单场景训练；
- 单 memory 条件化；
- 单模型；
- ACE-G 两阶段训练；
- 重点是先把 `true global` baseline 变得可复现、可解释、可审计。

这意味着，当前阶段的主任务不是一下子把系统推进到“通用多场景 LMC”，而是先把：

- global / local 语义；
- S1 / S2 行为；
- compressor / fusion 的关键结构配置；
- 日志和 checkpoint 元数据

全部固定住。只有在这个 baseline 足够清楚后，后面的多场景、sub-memory、归一化、MoE 才有可解释性。

## 2. 已经完成的事情

以下内容已经进入当前基线合同，属于“已做完的稳定化工作”：

### 2.1 global baseline 语义稳定化

- `--lmc_mode global` 不再默认因为 visibility fallback 静默退化成 `local`。
- 现在默认要求：命令请求什么模式，运行时就是什么模式；除非显式打开
  `lmc_auto_mode_by_visibility=True`。
- 日志和 checkpoint 会同时记录：
  - `requested_lmc_mode`
  - `effective_lmc_mode`

这件事非常关键，因为旧的很多 “global” baseline，实际上是
`requested-global + effective-local`。

### 2.2 deterministic FPS

- GeoLMC 全局压缩的 FPS 起点已经显式化。
- 默认策略改为 `farthest_from_center`，从而保证同一份 memory 和同一份 checkpoint
  在训练、S2、测试时尽量压出同一组 latent coordinates。
- 旧随机行为保留为显式 opt-in：`legacy_random`。

### 2.3 key layer 显式化

- 过去 compressor 的 key layer 选择带有隐含 helper 语义，容易误读。
- 现在通过 `lmc_key_slice_idx` 显式指定。
- 当前兼容默认仍然保持旧实际行为：
  - 多层 memory 默认取 `slice 2`
  - 对当前 memory 即 `layer 12`

### 2.4 LMC 结构元数据保存

- checkpoint 和日志中现在会记录结构相关的关键字段，例如：
  - requested/effective mode
  - `layers_idx`
  - `lmc_key_slice_idx`
  - `lmc_key_layer_label`
  - `lmc_fps_start_policy`
  - `s1_loss_step_mode`

这一步的意义是：训练和测试重建模型时，不再依赖 parser 的后来默认值。

### 2.5 S2 compressor freeze 显式化

- ACE-G 的 S2 阶段不应训练 compressor，这个边界已经显式写进代码。
- 现在 S1 / S2 的 compressor trainability 有明确边界，不再依赖隐式 no-grad 和
  optimizer 参数组副作用。

### 2.6 S1 sampled loss time-axis 显式化

- `s1_loss_step_mode` 已显式支持：
  - `fixed_zero`
  - `per_iter`
  - `global_monotonic`
- 旧行为没有被直接破坏，而是被显式映射出来。

## 3. 现阶段真正应该进入 LMC 架构修改主线的内容

这部分是“近线任务”。它们仍然围绕当前单场景 ACE-G baseline，不改系统总问题定义。

## 3.1 先完成 true-global baseline 对比

在继续改架构前，当前最重要的工作仍然是把以下对比做扎实：

- 旧的 requested-global/effective-local baseline
- true-global + `fixed_zero`
- true-global + `per_iter`

这是后续所有判断的起点。没有这一步，就无法回答：

- true global 本身到底有没有收益；
- `per_iter` 是否比 `fixed_zero` 更稳定；
- 之后的 GeoMatch / geometry / MoE 改进，应该跟谁比。

## 3.2 统一 loss contract

当前 S1 sampled、S1 full-map、S2、S2-G 之间仍然存在重复但不完全一致的
reprojection / invalid loss 逻辑。

这是现阶段最值得优先做的工程清理，因为它：

- 不改变问题定义；
- 能降低后续实验解释混乱；
- 能避免把 loss path 差异误判成架构收益。

原则是：

- 先统一实现；
- 先保持行为兼容；
- 不要在同一次提交里一边抽 helper，一边改 loss 数学定义。

## 3.3 先加 observability，再加新 loss

在 fusion 或 head 上增加新目标前，当前最缺的是可观测性，而不是更复杂的正则项。

近期最合理的加法是 diagnostic-only：

- attention entropy
- token usage
- effective token count
- avg max attention
- raw / fused feature norm
- attention output norm

这样才能回答：当前 K=64 token 是充分使用了，还是塌缩了，还是过于平均。

## 3.4 GeoMatch Fusion v1 属于近线架构改动

当前 fusion 的几何信息只进入 value path，不进入 key logits。

这意味着：

- 几何信息会影响“读出什么”；
- 但不会直接影响“选中哪个 token”。

因此，GeoMatch Fusion v1 是当前最合理的近期架构改动之一：

- 保留当前 `value_only` 作为精确 baseline；
- 再做 `gated_key_value` 作为单变量 ablation；
- 其中 key geometry 初始值应当很弱，value geometry 应尽量保持当前行为。

这件事应放在 observability 之后，而不是之前。

## 3.5 compressor geometry contract 也属于近线，但要和 GeoMatch 分开

compressor 侧现在还有两类近期值得做的事情：

1. key layer ablation
   - 先比较 `layer 12 / 18 / 6 / final`
   - 再决定是否要 learned scalar mix

2. geometry scene-scale contract
   - 统一 compressor PE、fusion PE、distance bias 的几何尺度编码
   - 例如 `coords_norm = coords / scene_scale`

但这里要明确：

- 这是“几何编码输入尺度”的合同；
- 不是“完整坐标归一化训练合同”；
- 更不是“参考坐标系 + 归一化 + 多场景”那条长期路线。

因此，compressor geometry contract 是近线任务，但必须和以下内容分开：

- GeoMatch Fusion v1
- full normalized target training
- reference-frame memory contract

## 4. 重要但不应混入当前架构主线的长期计划

下面这些方向都重要，而且你这次提到的很多点都已经进入这个类别。但它们改变的是更高层系统合同，不能混在当前 baseline 架构小改里。

## 4.1 多场景联合训练，得到真正通用的 LMC

这是长期核心目标之一。

目标不是继续做“单场景单 memory 调参”，而是：

- 一个 LMC 可以联合吸收多个 scene 的 memory；
- 学到更通用的 compressor / fusion prior；
- 成为真正可迁移的 memory-conditioned localizer。

但它要成立，前提是：

- scene identity 明确；
- coordinate frame 明确；
- normalization policy 明确；
- checkpoint 能说清楚当前 memory shard / scene shard 假设。

所以，多场景联合训练现在应该放在长期系统设计中，而不是当前 global baseline 改造中。

## 4.2 reference-frame memory coordinate contract

你提到的“特征在参考坐标系下，点云却在世界坐标系下，存在 gap”是完全成立的。

长期上，更合理的合同确实应当是：

- memory feature 在参考坐标系；
- point cloud 也变换到参考坐标系；
- 模型在参考坐标系下预测；
- 最后再把结果恢复到世界坐标系用于 DSAC / PnP。

即：

```text
world -> reference -> model prediction -> reference -> world
```

但这里必须保留一个明确约束：

- `scene_center` 仍然沿用当前经验上有效的定义：
  所有相机中心的均值；
- 不能因为引入 reference frame，就静默把 `scene_center` 改成参考原点或点云质心。

这是长期系统合同，不应和当前 GeoMatch / fusion / key-layer 改动混在一起。

## 4.3 完整坐标归一化路线

你强调归一化路线是未来多场景联合训练的关键，这个判断是对的。

长期上，更合理的训练空间很可能是：

```text
P_world -> P_ref -> P_ref_norm
```

其中：

- `world -> reference` 解决坐标系对应关系；
- `reference -> normalized` 解决跨场景尺度不一致和 loss 被大场景主导的问题。

这条路线的核心价值：

- 消除不同场景尺度差异；
- 让多场景 joint training 更可控；
- 与 VGGT / MapAnything 类 feed-forward 3D 系统的思路一致。

但注意不能混淆两类“归一化”：

1. 近期的 geometry scene-scale encoding
   - 只作用于 PE / distance bias 输入

2. 长期的 full normalized target contract
   - 改变模型输出空间与训练目标空间

这两件事不能在一个实验里一起改。

## 4.4 大场景下的 sub-memory 构建与路由

这也是长期系统层问题，而不是当前 LMC block 小改。

你提到 ACE 在 Cambridge 上先切数据集、再分别处理的思路值得借鉴。对我们这里，更合理的长期目标不是：

- 每个 cluster 单独训练一个模型

而是：

- 先对场景或参考视图做聚类；
- 每个 cluster 构建一个 sub-memory；
- 仍坚持一个统一模型；
- query 时对 sub-memory 做路由 / 候选检索 / rerank。

也就是说，ACE 的 cluster 思路可以借鉴到：

- 子数据集划分；
- sub-memory 构建；
- 更清晰的局部几何隔离；

但训练合同要升级为“一个模型处理多个 memory shard”，而不是多模型系统。

## 4.5 sub-memory 的路由与融合

这是长期设计里必须单独认真做的问题，不能顺手拍脑袋决定。

核心分歧有两类：

1. 预路由
   - 先选择最可能的 top-k sub-memory，再做推理

2. 后融合 / 后 rerank
   - 对多个候选 sub-memory 都跑，再按最终 pose 证据选

两者权衡很明确：

- 预路由更快，更适合超大场景；
- 但路由错了，可能整张 query 直接失败；
- 后 rerank 更稳，但更贵。

当前长期文档中的倾向是：

- first design 先做 top-2 / top-3 pre-routing
- 再以 DSAC / RANSAC inlier 或 reprojection score rerank
- 不建议直接对不共享严格坐标合同的 sub-memory 结果做坐标平均

这部分现在应当记入长期规划，而不是当前 GeoMatch/Fusion 代码主线。

## 4.6 MapAnything / VGGT 的 training-free 加速

这是一个重要的系统效率分支，但不属于当前 LMC 架构修改主线。

原因很简单：

- 它作用在 memory extraction / feature construction；
- 不直接作用在当前 ACE-G compressor / fusion / head 合同；
- 如果加速方法改变了 feature 数值分布，它就不再是“同一种 memory 条件”。

因此，这条线应该独立追踪：

- `AVGGT`
- `Faster VGGT with Block-Sparse Global Attention`
- MapAnything 后续 inference acceleration

只有在明确“数值是否等价”之后，才决定它是同条件加速，还是新 memory 条件。

## 4.7 ray / Plucker geometry 的使用

当前 memory 中保存了射线、Plucker ray map 等几何量，这些确实是后续很重要的潜在增益点。

但现阶段不建议直接把它们塞进模型，原因是：

- 先要搞清楚 frame convention；
- 先要确认它们与 points、poses、reference frame 的对应关系；
- 先要知道它们是在帮助 routing、attention bias，还是 residual head。

所以更合理的长期顺序是：

1. 先把它们作为 diagnostic / consistency check 资产；
2. 再考虑 ray-aware attention、visibility prior、anchor residual conditioning 等。

## 4.8 回归头用 MoE 是否有说法

有说法，但它应该明确归入长期研究项，而不是当前 baseline 主线。

合理之处在于：

- 不同区域、不同物体、不同几何状态的特征分布可能不一致；
- 单一 head 可能在做“平均映射”；
- token-routed 或 anchor-routed expert head 在理论上是有吸引力的。

但风险同样明确：

- 我们当前 supervision 是 scene-coordinate reprojection，不是 semantic label；
- “按物体分专家”只是直觉，不是现成训练合同；
- foreground/background MoE 也需要可靠 mask 或 routing 信号；
- 如果先上 MoE，很容易把 head、routing、loss、normalization、memory contract 一起搅乱。

因此，长期上更合理的 MoE 路线是：

- 优先考虑 token-routed / anchor-routed experts；
- 而不是直接假设 object-category MoE；
- 必须建立在 attention diagnostics、routing stability、coordinate contract 都比较清楚之后。

## 5. 一个清晰的阶段划分

为了后续不混乱，建议按下面的阶段顺序推进。

### 第一阶段：把当前 true-global baseline 做扎实

- 完成 `global + fixed_zero` / `global + per_iter` 对比
- 明确 old effective-local baseline 与 true-global baseline 的差异
- 清理 loss contract
- 加 observability

### 第二阶段：做近线 LMC 架构 ablation

- GeoMatch Fusion v1
- key-layer ablation
- geometry scene-scale encoding

### 第三阶段：建立长期系统合同

- reference-frame point/memory contract
- full normalized target contract
- multi-scene joint training

### 第四阶段：解决大场景与专家化问题

- sub-memory construction
- routing / reranking
- anchor-assisted head
- MoE head
- ray / Plucker usage

### 第五阶段：系统效率优化

- MapAnything / VGGT acceleration
- memory extraction cost reduction

## 6. 当前结论

一句话概括：

- 现在真正该进入当前 LMC 架构主线的，是 baseline 对比、loss contract、diagnostics、GeoMatch Fusion v1、compressor geometry contract。
- 多场景联合训练、reference-frame 点云对齐、full normalization、sub-memory、MapAnything/VGGT 加速、ray/Plucker、MoE 都很重要，但它们属于长期系统设计，不应混进当前 baseline 架构修改。
- 已完成的部分主要是 baseline 语义稳定化与关键元数据显式化，这些已经为后续研究打下了必要基础。
