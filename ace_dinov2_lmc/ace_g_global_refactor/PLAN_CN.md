# ACE-G 全局重构计划

范围：`--lmc_flow ace_g`，且实际 `lmc_mode=global`。

维护规则：`PLAN.md` 是计划的 source of truth。对 `PLAN.md` 做任何结构性修改时，
必须同步更新 `PLAN_CN.md` 作为中文翻译。不要再维护单独的 `ROADMAP_CN.md`。

本文件是高层跟踪器、优先级地图以及给新 agent 的入门摘要。它应当足够详细，使读者无需立刻阅读源代码，也能理解问题和预期方案。具体实现说明仍然分别放在 `steps/` 下的各个步骤文档中。

## 状态图例

- `[todo]`：尚未开始。
- `[designing]`：目标已接受，但具体实现方案仍在讨论中。
- `[ready]`：已经就具体实现措施达成一致。
- `[done]`：该步骤的代码/文档已经验证完成。
- `[deferred]`：已移出近期 ACE-G global 主线。

## 当前问题地图

### P0 - 实验语义过去并不可靠

现象：

- 旧命令可能写着 `--lmc_mode global`，但运行时会静默切换到 `local`。
- global FPS 使用随机起点，所以同一份 memory / checkpoint 可能产生不同的 latent 坐标。
- 一些会影响结构的配置是隐含的，或者没有保存在 checkpoint metadata 中。

风险：

- 一个结果可能被标成 global，但实际上是 effective-local。
- S1 / S2 / test 可能看到不同的压缩 memory 分布。
- layer 和 geometry 的 ablation 会变得无法解释。

状态：

- 大部分已经修复。见 steps 01、02、03、04。

### P1 - loss 合同过去部分重复

现象：

- S1 sampled、S1 full-map、S2、S2-G 之间存在重叠的 reprojection / invalid loss 逻辑。
- 边界行为可能不同：dyntanh step、invalid clamp、mask 以及 sampled / full-map 处理。

风险：

- 我们可能以为 S1 / S2 的监督是等价的，但实际上并非如此。
- 后续架构改动可能被错误归因，而真实原因是 loss path 差异。

状态：

- 当前兼容行为保持路径已经修复。见 step 07。

### P2 - compressor 几何难以解释

现象：

- key feature 选择过去是隐含的；现在虽然已显式化，但尚未做 ablation。
- 当前默认实际上是 slice 2 / layer 12。
- 当前 multi-layer memory contract 仍然是保守版本：key 使用一个选定 layer，value 保留 all-layer concatenation。后续 LMC 应该用 DPT-style fused memory feature，还是保留显式 multi-level compression / fusion，目前尚未决定。
- PE 和 distance bias 使用原始 scene 坐标，因此不同场景的几何频率 / 尺度不同。
- 当前的 distance bias 使用的是一种标量 `log(dist_sq)` 风格信号。

风险：

- key layer 的选择可能是隐藏瓶颈。
- 几何特征在不同场景尺度下可能代表不同含义。
- compressor attention 可能过拟合于原始坐标尺度，而不是有用的相对几何。

状态：

- 合同部分已经修复；但 ablation 和 scene-scale contract 还没做。见 `steps/06_compressor_geometry_contract.md`。

### P3 - fusion 的几何信息不会影响 matching logits

现象：

- 当前 fusion 只在 value path 中使用 geometry：
  `k = k_proj(memory_z)`，`v = v_proj(memory_z + PE(memory_p))`。
- query-to-token 的 matching logits 不会直接看到 `memory_p`。

风险：

- 模型只能在 token 已经被选中之后才利用几何信息。
- global token 可能纯靠 feature similarity 被选中，即使几何本应提供消歧作用。

状态：

- 已实现为 Progressive Geometry Injection / GeoKey v0。第一刀是可证伪的
  memory-side geometry probe，不是完整 GeoMatch。A0/A1/A2 实验待跑。见 step 05。

### P4 - token usage 不可观测

现象：

- 我们不知道 global K=64 tokens 是否被广泛使用、是否塌缩到少数 token，还是过于分散。
- 我们没有记录 attention entropy、effective token count、avg max attention 或 raw / fused feature norm 变化。

风险：

- 可能因为错误原因加入 usage loss 或 MoE-style routing。
- fusion 改进无法在最终 pose metric 之外得到解释。

状态：

- 仅诊断用途的 runtime observability 已在显式 flag 后实现。见 step 10。

### P5 - 未来多场景 LMC 需要新的系统合同

现象：

- 当前训练仍然是单场景、单 memory、基于 FPS 的 ACE-G。
- 未来目标是真正可复用、跨场景联合训练的 LMC。
- 当前 memory feature 来自 MapAnything / 参考坐标系特征路径，而点坐标仍主要按世界坐标处理。
- 大场景可能需要 sub-memory 构建与 routing。
- memory 文件中已经包含 ray / Plucker-style 几何，但当前 compressor / fusion / head 基本没有用到。

风险：

- 如果没有显式的 coordinate normalization，多场景训练会被大尺度场景主导。
- 参考系 feature 与世界系 point 之间会形成一个可避免的坐标系 gap。
- 一旦 sub-memory routing 选错 memory shard，整张 query 可能失败。
- 在 baseline 还不够可观测时引入 MoE head 或 geometry / ray 特征，会让复杂度过早上升。

状态：

- 这是长期系统设计项。不要混入当前 true-global baseline 改动。见 `LONG_TERM_RESEARCH_PLAN.md`。

## 优先级顺序

## 分类快照

本节将当前设计想法划分为：已完成工作、近期 ACE-G 架构工作、长期系统工作。目的在于避免把 baseline 清理和更大的研究改动混在一起。

### 已完成 / 当前 baseline 合同

- deterministic FPS 已经实现，并通过 `lmc_fps_start_policy` 记录在日志中。
- requested / effective LMC mode 已显式化；除非显式打开 visibility fallback，否则 requested mode 默认具有权威性。
- 旧的 key-layer 歧义已经通过 `lmc_key_slice_idx` 修复；五层 memory 的兼容默认值仍为 slice 2 / layer 12。
- checkpoint 现在会记录重建 compressor / fusion path 所需的结构相关 LMC metadata。
- ACE-G 的 S2 compressor freeze 已在 parameter trainability 边界上显式化。
- sampled S1 的 ReproLoss time-axis 行为已通过 `s1_loss_step_mode` 显式化。

### 近期 LMC 架构工作

这些项在当前 true-global baseline 对比完成后相关，但应当作为带有诊断日志的受控 ablation 加入：

- 仅诊断用途的 fusion / token 可观测性已经实现，并继续作为新 fusion ablation 的解释层。
- Progressive Geometry Injection / GeoKey v0 已通过显式 fusion flags 实现：先测试 scene-scale-normalized
  value geometry，再用 zero-init scalar gate 让 memory geometry 进入 fusion key。它不使用 query 3D 坐标，也不应被视为完整 GeoMatch。
- Detached token-routing prior 现在应排在任何 coordinate-based coarse GeoMatch 之前。它预测
  memory token 分布，可以保留多峰不确定性，而不是强迫 pre-fusion raw feature 先落到单个 3D 坐标。
- compressor key-layer ablation，从当前 layer 12 baseline 开始。
- basic key-layer 结果之后再做 memory feature hierarchy ablation：当前 selected-layer key / all-layer value、key scalar mix、DPT-style fused memory feature，或显式 multi-level compressor / fusion。
- compressor / fusion 的 geometry scale 诊断，以及 PE / distance bias 的共享 scene-scale encoding contract。
- shared reprojection / invalid-loss helper 已在当前兼容行为保持路径完成。

### 长期系统工作

这些项很重要，但它们改变的合同范围比当前单场景 ACE-G baseline 更大，不应混入首轮架构 ablation：

- 多场景联合 LMC 训练。
- reference-frame point / memory coordinate contract。
- 面向多场景 loss 平衡的完整 coordinate normalization。
- 面向大场景的 sub-memory 构建、routing 与候选 reranking。
- training-free 的 MapAnything / VGGT memory extraction 加速。
- ray / Plucker 几何使用。
- anchor-assisted coordinate head 与 MoE / expert head。
- DSD-style local / deformable compressor 以及更大范围的 head 替换。

### Priority 0 - 已完成的 baseline 稳定化

1. `[done]` Deterministic FPS。
   - 问题：随机 FPS 起点会改变 latent 坐标。
   - 修复：确定性起点策略，默认 `farthest_from_center`。
   - 详情：`steps/01_deterministic_fps.md`

2. `[done]` 显式的低风险 LMC 合同。
   - 问题：key layer、config 和 S2 compressor freeze 过去是隐含的。
   - 修复：显式 key slice、config metadata、S2 trainability 边界。
   - 详情：`steps/02_low_risk_contract.md`

3. `[done]` 显式的 S1 sampled loss time-axis。
   - 问题：sampled S1 实际上使用了隐含的 fixed-zero dyntanh 行为。
   - 修复：`fixed_zero / per_iter / global_monotonic` mode 已显式化。
   - 详情：`steps/03_s1_sampled_loss_step_mode.md`

4. `[done]` requested LMC mode 默认具有权威性。
   - 问题：visibility fallback 会静默把 global 改成 local。
   - 修复：`lmc_auto_mode_by_visibility=False` 成为默认值。
   - 详情：`steps/04_lmc_mode_authority.md`

### Priority 1 - 完成当前 true-global 对比

5. `[ready]` 在做更多架构改动前先比较 true-global baseline。
   - 必需运行：`global + fixed_zero`、`global + per_iter`。
   - 要与旧的 requested-global / effective-local baseline 对比。
   - 成功指标：post-train multi-seed pose metrics，加上 best iteration 和 training stability logs。
   - 详情：`steps/05_geomatch_near_term.md`

### Priority 2 - 清理共享监督合同

6. `[done]` 统一 reprojection 和 invalid-loss 行为。
   - 问题：重复的 loss 实现可能掩盖行为差异。
   - 范围：S1 sampled、S1 full-map、S2、S2-G。
   - 约束：首先保持当前行为；只在 tests / smoke checks 支持下重构成 shared helper。
   - 优先原因：它可以在改 fusion / compressor 架构前降低风险。
   - 详情：`steps/07_loss_contract.md`

7. `[done]` 非架构 hygiene 与 fail-fast 检查。
   - 问题：架构 ablation 前仍有一些工程风险：隐含的 head-grid trimming、延迟暴露的 memory feature-dim 错误，以及 preset 选项与 preset 定义漂移。
   - 范围：不改变 loss、fusion、compressor 或 head architecture 行为。
   - 详情：`steps/08_non_arch_hygiene.md`

8. `[done]` Module train/eval mode restoration contract。
   - 问题：helper 函数会临时切换 module 的 eval/train mode，并可能恢复出比调用方原始状态更宽泛的 train 状态。
   - 范围：compressor memory compression、fused-buffer construction、ACE-G raw buffer construction，以及 regressor / encoder / heads 的精确 mode restoration。
   - 详情：`steps/09_module_mode_contract.md`

### Priority 3 - 在引入新 loss 前先加可观测性

9. `[done]` 增加仅诊断用途的 compressor / fusion 可观测性与 runtime semantics。
   - 问题：token usage 和 fusion 行为不可见。
   - 跟踪项：attention entropy、effective token count、avg max attention、token usage、raw / fused feature norms、compressor geometry stats，以及 summary 中的 experiment semantics。
   - 不引入新 loss；关闭时不改变行为。
   - 详情：`steps/10_runtime_observability_and_semantics.md`

10. `[designing]` 增加 compressor geometry contract 和 ablation。
   - 问题：key layer 和 geometry scale 可能是隐藏瓶颈。
   - 首轮 ablation：layer 12 baseline、layer 18、layer 6、final、learned scalar mix。
   - 后续 ablation 应决定 memory layers 是保持 selected-key / all-layer-value，通过 DPT-style feature neck 折叠成单个特征，还是以 multi-level compression / fusion 的方式显式保留。
   - scene-scale contract 应先于 PE / distance-bias redesign。
   - 详情：`steps/06_compressor_geometry_contract.md`

### Priority 4 - 受控架构 ablation

11. `[done]` Progressive Geometry Injection / GeoKey v0。
   - 问题：geometry 不影响 query-memory matching logits。
   - 已实现 ablation modes：A0 当前 value-only raw PE，A1 value-only normalized PE，A2 normalized PE 加 memory-side GeoKey。
   - 将 A1/A2 视为 probe。单凭这组结果不能证明或否定完整 query-memory GeoMatch。
   - 硬约束：A0 必须精确兼容旧行为，GeoKey scalar gate 初始化为 0，首轮固定 `lmc_key_slice_idx=2`，并在日志和 checkpoint metadata 中记录 scene-scale source/value。
   - 详情：`steps/05_geomatch_near_term.md`

12. `[todo]` 仅当 diagnostics 能证明有必要时才加 usage regularization。
    - 问题：可能存在 token collapse。
    - 只有在测得 token usage 后才加入。
    - 首先做 S1-only 的 `lambda_usage` ablation。
    - 详情：`steps/05_geomatch_near_term.md`

13. `[todo]` fusion residual gate 作为低优先级 ablation。
    - 问题：`LayerNorm(query_feats + attention_out)` 可能让 fusion 过度重参数化 raw features。
    - 候选形式：`query_feats + residual_gate * attention_out`。
    - 优先级低于 key / value geometry 与 diagnostics。
    - 详情：`steps/05_geomatch_near_term.md`

### Priority 5 - 中长期研究

14. `[designing]` Detached token-routing prior 优先于 coordinate GeoMatch。
    - 保持 ACE head 为主输出。
    - 优先预测 detached memory-token 分布 `r(token | query)`，并以
      `g * log(r + eps)` 的形式作为弱 residual logit prior 加入，而不是先加 coarse-coordinate distance bias。
    - 这比单点 coarse 3D 更安全，因为它能让重复纹理或歧义 patch 保持多峰。
    - 只有在 A0/A1/A2 和 attention diagnostics 都理解清楚之后才尝试。
    - 详情：`steps/05_geomatch_near_term.md`

15. `[designing]` Confidence-gated coarse-coordinate GeoMatch 作为更晚的弱先验。
    - 不要把 detached coarse coordinate 当作 GeoKey 之后的默认下一步。错误的 coarse coordinate 会把 attention 拉向错误 memory 区域，refinement 可能救不回来。
    - 如果实现，必须是具备 fallback 的 residual：zero / warmup global gate、per-query confidence、normalized coordinates、nearby reward 而不是远处 token punishment，并且 image feature logits 仍是主路径。
    - anchor-relative branch 仍只作为辅助分支。
    - 详情：`steps/05_geomatch_near_term.md`

16. `[deferred]` 更大的研究改动。
    - 完整 normalized-coordinate target training。
    - reference-frame memory coordinate contract。
    - DSD / local / deformable compressor。
    - sub-memory routing。
    - 多场景联合 LMC 训练。
    - ACE head replacement / MoE architectures。
    - ray / Plucker geometry 使用。
    - P4Pf / uncalibrated-query solver 改动。
    - 详情：`LONG_TERM_RESEARCH_PLAN.md`

## 详细任务卡

### 1. Deterministic FPS

状态：`[done]`

问题：

- global compressor 使用 FPS 选择 latent 坐标。
- 旧的 FPS 起点来自随机性。
- 因此同一份 memory 和 checkpoint 在 S1、S2、test 时可能得到不同的 latent token 坐标。

为什么重要：

- latent 坐标会影响 compressor query PE。
- 它们会影响 global attention distance bias。
- 它们会影响来自 `memory_p` 的 fusion PE。
- 它们会改变 ACE head 看到的 fused feature 分布。

已实现方向：

- 默认使用确定性的 FPS 起点策略。
- 当前默认值为 `farthest_from_center`。
- 仅保留 `legacy_random` 作为对旧行为的显式 opt-in。

验证：

- 对同一份 memory 重新压缩，应当选择相同的 latent 坐标。
- checkpoint / 日志 metadata 应记录 `lmc_fps_start_policy`。

详情：

- `steps/01_deterministic_fps.md`

### 2. 显式 key layer 与低风险 LMC 合同

状态：`[done]`

问题：

- memory feature 包含多个 layer slice：
  `0, 6, 12, 18, final`。
- 旧 compressor 的 key path 看起来像是在选 layer index `1`，但由于 helper 语义，实际选中的是 slice `2`，即 layer `12`。
- 这非常容易误读，也让 layer ablation 变得含糊。

为什么重要：

- key feature 决定 memory attention matching。
- 一个静默的 key-layer 不匹配会改变 compressor 行为，而命令名和 checkpoint 名保持不变。

已实现方向：

- 增加显式的 `lmc_key_slice_idx`。
- 多层 memory 的默认行为保持兼容：slice `2`。
- 保存 / 记录 `lmc_key_slice_idx`、`key_layer_label`、`layers_idx`。
- 在 ablation 证明更好的 key layer 之前，保留旧行为。

验证：

- 日志会显示选中的 key slice 和 label。
- checkpoint config 包含选中的 key metadata。
- test-time reconstruction 使用 checkpoint config，而不是意外继承 parser 默认值。

详情：

- `steps/02_low_risk_contract.md`

### 3. 保存会影响结构的 LMC 配置

状态：`[done]`

问题：

- 一些会影响模型结构或行为的字段过去没有完整记录在 checkpoint 中。
- 如果 parser 默认值之后变化，训练和测试可能会重建出不同结构。

关注字段：

- requested / effective LMC mode
- `layers_idx`
- selected key slice / key layer label
- `pe_normalize_input`
- `geo_sigma`
- `num_fine`、`num_coarse`
- `backbone_feature_dim`
- FPS start policy
- S1 sampled loss step mode
- future fusion geometry mode

已实现方向：

- 将低风险的结构 metadata 保存到 `lmc_config` 中。
- 记录 requested / effective mode 和 key layer metadata。

验证：

- 检查 checkpoint 时，应能看到重建 compressor / fusion path 所需的相同配置。
- 旧 checkpoint 仍可通过兼容默认值加载。

详情：

- `steps/02_low_risk_contract.md`

### 4. 显式的 S2 compressor freeze

状态：`[done]`

问题：

- ACE-G 的 S2 不应训练 compressor。
- 旧行为依赖 `no_grad()` 和 optimizer membership 的组合。
- 未来代码可能不小心在 S2 中 live-forward compressor，从而泄漏梯度。

已实现方向：

- 增加显式的 trainability 边界：
  S1 启用 compressor 训练，S2 禁用。
- 在保持当前行为的同时，让 stage 合同可见。

验证：

- S1 日志显示 compressor 可训练。
- S2 日志显示 compressor 已冻结。
- S2 的 optimizer parameter groups 不应包含 compressor 参数。

详情：

- `steps/02_low_risk_contract.md`

### 5. S1 sampled loss step mode

状态：`[done]`

问题：

- legacy mode 下的 sampled S1 使用了隐含的 `iter_for_loss=0` 路径。
- 这意味着 dyntanh 在整个 sampled S1 中都停留在最宽松的 step。
- 很难判断这究竟是预期 curriculum，还是历史遗留行为。

为什么重要：

- `fixed_zero` 更宽松，可能有利于 compressor 早期学习。
- `per_iter` 更严格，可能更贴近正常训练日程。
- 如果不记录，实验无法说明使用的是哪种行为。

已实现方向：

- 增加显式的 `s1_loss_step_mode`：
  - `fixed_zero`
  - `per_iter`
  - `global_monotonic`
- 保持兼容默认值：
  - legacy profile -> `fixed_zero`
  - `mapany_flow_v1` -> `global_monotonic`

验证：

- 日志显示 requested / resolved 的 S1 sampled loss step mode。
- 当前对比运行应隔离 `fixed_zero` 与 `per_iter`。

详情：

- `steps/03_s1_sampled_loss_step_mode.md`

### 6. LMC mode 权威性

状态：`[done]`

问题：

- 旧代码默认开启 visibility-based fallback。
- 命令中的 `--lmc_mode global` 可能静默变成 effective `local`。
- fallback statistic 只检查采样 memory points 是否在各相机前方（相机坐标系下 `z > 0`），对 multi-view scene point cloud 来说过于粗糙。

为什么重要：

- 旧的 “global” baseline 可能实际上是 effective-local。
- 结果目录和命令行会因此被误读。

已实现方向：

- `lmc_auto_mode_by_visibility` 默认改为 `False`。
- 除非显式启用 fallback，否则 requested `lmc_mode` 具有权威性。
- 日志 / checkpoint 记录 requested 和 effective mode。

验证：

- true-global 运行必须显示：
  `requested=global effective=global auto_by_visibility=False`。
- 旧 fallback 运行应被视为 requested-global / effective-local，除非日志证明并非如此。

详情：

- `steps/04_lmc_mode_authority.md`

### 7. 当前 true-global baseline 对比

状态：`[ready]`

问题：

- 旧的强 scene2a baseline 并不是真正 global；它是 requested-global 但 effective-local。
- 在做架构改动之前，我们需要真正的 true-global 参考点。

必需对比：

- 旧的 requested-global / effective-local baseline
- true-global + `fixed_zero`
- true-global + `per_iter`

为什么重要：

- 如果 true global 本身已经更好，那么 fusion / compressor 改动应与 true global 对比，而不是与旧的 effective-local 对比。
- 如果 `per_iter` 表现更差，未来工作就应继续把 `fixed_zero` 作为稳定的 S1 baseline。

验证：

- 对比 best iteration、best score、post-train multi-seed median，以及 25 / 10 / 5 / 2 / 1 cm recall。
- 确认日志中有 effective mode 和 S1 step mode。

详情：

- `steps/05_geomatch_near_term.md`

### 8. 统一的 reprojection / invalid loss 合同

状态：`[done]`

问题：

- S1 sampled、S1 full-map、S2、S2-G 中都存在重叠的 reprojection 和 invalid-loss 逻辑。
- dyntanh step handling、invalid clamp、mask、sampled / full-map 行为之间可能存在小差异。

为什么重要：

- 架构改动可能会因为 loss path mismatch 而被错误归因为 metric shift。
- 很难证明 S1 / S2 的监督是一致的。

已实现方向：

- 已抽出保持兼容行为的 shared helper。
- 当前路径不故意修改数学定义。
- 后续架构改动可以基于同一套 reprojection / invalid-loss contract。

验证：

- 对 synthetic data 运行轻量 helper checks。
- 运行短 smoke command 或 compile check。
- 日志应标明当前启用的是哪种 loss contract / mode。

详情：

- `steps/07_loss_contract.md`

### 9. 非架构 hygiene 与 fail-fast 检查

状态：`[done]`

问题：

- sampled rows 的 head-grid packing 过去被命名成 trimming，而不是实际的 fake-grid contract。
- tail trimming 只警告一次，没有累计的 stage statistics。
- LMC memory feature-dim mismatch 只是 warning，但后续会导致 shape error。
- `memory_compare_ace_g_v2` 存在于 preset defaults 中，但 `--train_preset` 不接受它。

决策：

- 保持当前训练数学和普通实验行为不变。
- 只把无效 memory shape contract 提升为立即失败。
- 暴露已经定义好的 preset，但不改变旧 preset defaults。
- 将这个改动记录在 `steps/08_non_arch_hygiene.md`；后续 code/config 改动必须更新 refactor docs，才算完成。

验证：

- `python -m py_compile options_dinov2_lmc.py trainer_dinov2_lmc.py train_ace_dinov2_lmc.py test_ace_dinov2_lmc.py /home/xwh/project/ace_depth/ace_compressor.py`
- CLI parser 接受 `--train_preset memory_compare_ace_g_v2`。
- Synthetic fail-fast check 确认不可整除的 memory feature dimensions 会带着 config context 抛出 `ValueError`。

详情：

- `steps/08_non_arch_hygiene.md`

### 10. Module train/eval mode restoration contract

状态：`[done]`

问题：

- 一些 helper 函数会为了确定性 feature extraction 临时切到 eval mode，然后用 `regressor.train()` 这类宽泛调用恢复。
- 这可能把原本 eval-only 的 child module，尤其是 frozen DINO encoder，意外切回 train mode。
- `_compress_memory()` 如果 compression 失败，没有通过 `finally` block 恢复 compressor mode。

决策：

- 增加小型 helper 来捕获和恢复精确的 module `.training` flags。
- 在 compressor memory compression、fused-buffer construction 和 ACE-G raw-buffer construction 周围使用这些 helper。
- 保持 output math、optimizer membership、loss contracts 和 trainability boundaries 不变。

验证：

- `python -m py_compile trainer_dinov2_lmc.py`
- Synthetic smoke check 能捕获 / 恢复混合 parent/child module modes。
- 既有 full sanity run 仍作为 metric-level check；这个仅状态恢复的改动不需要新的 baseline training。

详情：

- `steps/09_module_mode_contract.md`

### 11. 仅诊断用途的可观测性

状态：`[done]`

问题：

- 我们目前看不出来 K=64 的 global tokens 是否被良好利用。
- attention usage、entropy、effective token count、raw / fused norm 等过去没有日志。

为什么重要：

- usage loss 只有在确实存在 collapse 时才有意义。
- Progressive Geometry Injection 的改进应能通过 routing / token 行为来解释，而不仅仅是 pose metrics。

已实现方向：

- 已增加默认关闭的可选 runtime diagnostics。
- 实现不加入 loss，关闭时不改变输出。
- eval summary 记录 requested/effective mode、key slice、FPS policy、S1 step mode 与 ACE-G fusion mode。

指标：

- attention entropy
- token usage
- effective token count
- average max attention
- raw feature norm
- attention output norm
- fused feature norm
- GeoKey 启用时的 geometry gate / scale values

验证：

- diagnostics 可在指定运行中启用。
- diagnostics 关闭时，普通运行保持不变。

详情：

- `steps/10_runtime_observability_and_semantics.md`

### 12. Compressor geometry contract 与 ablation

状态：`[designing]`

问题：

- key layer 已显式化，但还没做 ablation。
- multi-layer memory usage 目前只确定了 selected-layer key + all-layer-concat value 的兼容合同，后续是否改为 DPT-style fused feature 或 explicit multi-level fusion 尚未决定。
- PE 和 distance bias 仍然使用原始 scene 坐标。
- distance-bias geometry 过于狭窄：本质上仍是 scalar distance / log-distance。

为什么重要：

- layer 12 可能不是最优选择。
- geometry encoding 在不同场景中可能对应不同的物理尺度。
- compressor attention 可能更敏感于 scene size，而不是真正有用的 geometry。

建议方向：

- 从 key-layer ablation 开始：
  layer 12、layer 18、layer 6、final、learned scalar mix。
- 然后定义 scene-scale policy。
- 在 key-layer 和 scene-scale 基础明确之后，再测试 memory feature hierarchy variants：
  当前 selected-layer key / all-layer value、key scalar mix、DPT-style fused memory feature、explicit multi-level compressor / fusion。
- 只有在 scene-scale contract 建立之后，才去测试 PE 和 distance-bias 变体。

不要混在一起：

- key-layer ablation
- Progressive Geometry Injection / GeoKey v0
- scene-scale PE changes
- DPT-style 或 explicit multi-level memory feature fusion

每一项都应是独立 ablation。

详情：

- `steps/06_compressor_geometry_contract.md`

### 13. Progressive Geometry Injection / GeoKey v0

状态：`[done]`

问题：

- 当前 fusion 的 geometry 只影响 values，不影响 query-memory matching logits。
- fusion 前的 query-side 3D 坐标不可靠，所以第一阶段 geometry-logit 改动不能依赖 query geometry。

建议方向：

- A0：保留当前 value-only raw centered PE 作为精确 baseline。
- A1：value-only，但使用显式 scene-scale-normalized memory geometry。
- A2：A1 加 memory-side GeoKey：
  `k_input = memory_z + key_geo_scale * pe_mem`。
- 首轮 A0/A1/A2 固定 `lmc_key_slice_idx=2`，避免 key-layer ablation 污染 fusion geometry 结果。
- scene scale 应来自稳定的 memory metadata 或预先计算的 memory-point statistic，不要来自每次 forward 的 latent coordinates。

重要兼容性说明：

- 当前 value path 等价于 `value_geo_scale=1.0`。
- 如果把 sigmoid gate 初始化为零，其输出会是 `0.5`，这并不等于旧行为。
- GeoKey 更适合使用直接 scalar parameter，并初始化为 `0.0`，让 A2 初始严格等价于 A1。
- A1/A2 是 probe。A2 优于 A1 支持 memory-side geometry 进入 key；A2 与 A1 持平并不能否定后续 query-side priors，但也不能作为直接跳到 coordinate-distance bias 的理由。
- 已实现 CLI：
  `--lmc_fusion_geometry_mode {value_only_raw,value_only_norm,geokey_norm}`、
  `--lmc_fusion_key_geo_init`、
  `--lmc_fusion_scene_scale_source {memory_points_p95,fixed}` 和
  `--lmc_fusion_scene_scale_value`。

验证：

- 旧 checkpoint 能加载。
- 日志 / checkpoint 能标明 fusion geometry mode、scene-scale source/value、learned key geometry scale。
- A0 能精确重建当前 baseline。
- 与 true-global `forceglobal_s1_periter` value-only baseline 对比，同时检查 routing diagnostics，而不只看 pose metrics。

详情：

- `steps/05_geomatch_near_term.md`

### 14. 条件式 usage regularization

状态：`[todo]`

问题：

- token collapse 可能存在，但目前还没有测到。

风险：

- 如果一个 scene 自然会更多使用某些 token，那么强行把 token usage 拉成均匀分布反而可能是错的。

建议方向：

- 只有在 diagnostics 能证明 collapse 存在之后才加。
- 先做 S1-only。
- 尝试较小权重：`0.001`、`0.003`、`0.01`。

验证：

- effective token count 提升。
- pose metrics 不退化。
- attention maps 变得更有意义，而不是仅仅更平均。

详情：

- `steps/05_geomatch_near_term.md`

### 15. Fusion residual gate

状态：`[todo]`

问题：

- 当前 fusion 使用 `LayerNorm(query_feats + attention_out)`。
- 这可能让 fusion 变成一种大幅度的 feature reparameterization，而不是受控的 memory correction。

建议方向：

- 考虑使用 `query_feats + residual_gate * attention_out`。
- 其优先级低于 diagnostics 和 GeoKey geometry。

验证：

- 记录 residual gate 值以及 raw / fused norm ratio。
- 只有在 diagnostics 存在后才测试。

详情：

- `steps/05_geomatch_near_term.md`

### 16. Detached token-routing prior

状态：`[designing]`

问题：

- 完整 query-memory geometry bias 需要 query-side 信息，但由 raw feature 得到的 pre-fusion query 3D 不可靠。
- 在重复纹理、弱纹理或房间 / 区域混淆时，单个 coarse coordinate 可能发生结构性错误。
- 如果这个错误坐标被用于 distance bias，它会奖励错误 memory 区域，并让 refined head 很难恢复。

建议方向：

- 在 coordinate-based GeoMatch 前，优先做 detached token-routing prior。
- 预测 memory token 分布：
  `r_ik = P(memory token k | query i)`。
- 只把它作为弱 residual prior 加到 fusion logits：
  `logits = image_logits + g * log(r_ik + eps)`。
- `g` 初始化为 0，并在 diagnostics 证明有用之前限制或 warm up。
- 保持 ACE head 作为主输出。

为什么它应排在 coarse coordinate 前面：

- 分布可以保持多峰，这对歧义 patch 很重要。
- 它不要求在 fusion 前先选择一个可能错误的 3D 点。
- 它更像 coarse matching prior，而不是 hard coordinate refinement。

验证：

- 在启用 attention diagnostics 的情况下与 A1/A2 对比。
- 检查 routing entropy、top-token maps 和 pose metrics 是否一起改善。
- 如果它只是让错误 attention 更尖锐，应拒绝或仅保留为诊断工具。

详情：

- `steps/05_geomatch_near_term.md`

### 17. Confidence-gated coarse-coordinate GeoMatch / anchor-assisted branches

状态：`[designing]`

问题：

- Coordinate-based GeoMatch 需要 query-side 3D，但 pre-fusion query 3D 不可靠。
- 绝对坐标回归可能比 anchor-relative residual prediction 更难。

建议方向：

- 保持 ACE head 作为主输出。
- 只有在 A1/A2 diagnostics 和 token-routing prior 说明 memory-side geometry / routing 有用或不足之后，才考虑 detached coarse query-coordinate bias。
- 把 coarse-coordinate GeoMatch 视为 confidence-gated weak prior，而不是 hard coarse-to-fine refinement 假设。
- 使用 zero-init 或 warmup global gate，detach coarse coordinate，复用 A1/A2 的 scene-scale contract 做归一化，并保持 image feature logits 作为主 matching path。
- 优先使用 nearby reward，例如 `alpha * exp(-d^2 / tau)`，不要使用会压制所有远处 token 的全局 negative-distance penalty。
- 加入 bias 前必须有 per-query confidence。第一版 inference-compatible 信号可以用 coarse-to-memory nearest distance 和 / 或 predicted uncertainty；training-only reprojection validity 可用于诊断，但不能作为推理时的 gate。
- 只把 anchor-relative branch 当作辅助分支加入。
- 从较小 loss weight 开始，例如 `0.05` 或 `0.1`。

暂时不要做：

- 不要替换 ACE head。
- 在辅助分支证明有效之前，不要让 anchor-only 成为主路径。
- 不要让 coarse coordinate 主导 attention logits。
- 不要在没有 image logits fallback 的情况下加入 hard distance penalty。

详情：

- `steps/05_geomatch_near_term.md`
- `LONG_TERM_RESEARCH_PLAN.md`
