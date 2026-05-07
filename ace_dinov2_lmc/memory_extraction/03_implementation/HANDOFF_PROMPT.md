# Handoff Prompt

请先从项目根目录进入：

`cd /home/xwh/project/ace_depth/ace_dinov2_lmc`

先阅读这三个文档恢复上下文：

1. `memory_extraction/03_implementation/EXECUTION_PLAN_reference_consistent_memory.md`
2. `memory_extraction/03_implementation/PROGRESS_reference_consistent_memory.md`
3. `memory_extraction/03_implementation/HANDOFF_PROMPT.md`

注意：

- `IMPLEMENTATION_NOTES.md` 和 `IMPLEMENTATION_NOTES_zh.md` 主要是早期 BSE extraction 实现说明，不代表当前 Phase 3/4/5 的真实状态。
- 当前真正需要跟进的是 reference-consistent memory、cluster fallback、query-time ensemble、以及后续 shared-model multi-memory 路线。

## 当前总体状态

我们已经完成并验证了这些部分：

1. Phase 3 minimal reference gate 已落地。
2. post-repair 不再只盯单个最差洞，而是平衡 top-K uncovered clusters；repair 结果会写入 `memory_policy_report.json`。
3. Phase 4 第一版 offline cluster fallback 已实现并验证：
   - `cluster_fallback_plan.json`
   - `memory_bse.clustered.pt`
   - `memory_bse.cluster_*.pt`
4. per-cluster 单独训练/评估路径已打通。
5. query-time dual-checkpoint ensemble 已实现并验证：
   - 新脚本：`test_ace_dinov2_lmc_ensemble.py`
   - 主入口：`test_ace_dinov2_lmc.py --ensemble_networks ...`

但以下事情还没有完成：

1. `C1` 还没有完成真正的 end-to-end 训练验证。
2. 当前 cluster 验证仍是“两个独立 checkpoint + 后融合评测”，还不是“一个共享模型 + 多个 memory 文件”。
3. `clustered package` 还没有作为训练输入被原生消费；现在训练仍然是每个 `cluster_XX.pt` 单独跑。
4. query-time ensemble 现在是 independent prediction + DSAC inlier count selection，尚未有 learned router，也没有 shared-model multi-memory 版。

## 目前最可信的实验结论

### scene2a

`scene2a` 的 repaired Phase 4 C0 run 仍然是当前最稳定的 single-memory baseline。

关键 extraction：

`/home/xwh/project/ace_depth/ace_dinov2_lmc/memory_extraction/04_evaluation/memory_extract/scene2a/40v_v0.05_bilinear_bse_ut0.02_noaug_asb_adaptive_pool1.0_rgate4_prepair8_sor_gm_l2/20260423_103144`

关键信息：

- repair 执行了真实 swap：`3909 -> 4294`
- selected reference = `2850`
- coverage mean/p95/max = `0.6089 / 1.5291 / 2.3385`
- probe translation mean/q90/max = `0.0641 / 0.0984 / 0.1385`

当前最好训练结果：

`/home/xwh/project/ace_depth/ace_dinov2_lmc/04_evaluation/train_compare/memory_pooled_vs_asb/indoor6_ace/scene2a/dino_ace_lmc_ace_g/scene2a_asb40_phase4_c0_20260422_122243_aceg_fS2cie_global_res518_buf2.6M_F7.7M_K64_it28_ep24_bs5120_spi_s1buf_sp384_onecycle_improved`

结果：

- `5cm/5deg = 76.65%`
- `2cm/2deg = 26.07%`
- median = `0.3266 deg / 3.015 cm`

### scene3

`scene3` 是当前关键场景。单一 memory 路线会稳定触发 `cluster_branch`。

已经验证通过的 cluster fallback extraction：

`/home/xwh/project/ace_depth/ace_dinov2_lmc/memory_extraction/04_evaluation/memory_extract/scene3/60v_v0.05_bilinear_bse_ut0.02_noaug_asb_adaptive_pool1.0_rgate6_prepair12_cfb2_sor_gm_l2/20260425_130557`

关键产物：

- `cluster_fallback_plan.json`
- `memory_bse.clustered.pt`
- `memory_bse.cluster_01.pt`
- `memory_bse.cluster_02.pt`

已知事实：

- policy decision = `cluster_branch`
- 生成了两个 `cluster_local` memory
- 两个 cluster 当前都标记为 `low_confidence=true`
- failure reason = `disconnected_covis_graph`
- 这表示代码路径是对的，但 scene3 的结构确实复杂

## per-cluster 训练结果

cluster01 run：

`/home/xwh/project/ace_depth/ace_dinov2_lmc/04_evaluation/train_compare/memory_pooled_vs_asb/indoor6_ace/scene3/dino_ace_lmc_ace_g/20260425_162835_aceg_fS2cie_global_res518_buf2.6M_F7.7M_K64_it28_ep24_bs5120_spi_s1buf_sp384_onecycle_improved`

best：

- `5cm/5deg = 66.67%`
- `10cm/5deg = 86.98%`
- median = `0.7083 deg / 3.6337 cm`

cluster02 run：

`/home/xwh/project/ace_depth/ace_dinov2_lmc/04_evaluation/train_compare/memory_pooled_vs_asb/indoor6_ace/scene3/dino_ace_lmc_ace_g/20260425_162840_aceg_fS2cie_global_res518_buf2.6M_F7.7M_K64_it28_ep24_bs5120_spi_s1buf_sp384_onecycle_improved`

best：

- `5cm/5deg = 67.94%`
- `10cm/5deg = 86.03%`
- median = `0.6748 deg / 3.3586 cm`

注意：

- 单次 manual eval 会受到 DSAC 随机性影响。
- 评测时没有 test augmentation；波动主要来自 DSAC hypotheses sampling 和 winner sampling。
- 因此比较时尽量使用同一口径，优先考虑 `--hypotheses 256`。

## 联合测评结果

已经实现并验证 “逐帧双分支独立推理，按 DSAC inlier count 选最终 pose”。

命令入口有两种：

1. `test_ace_dinov2_lmc_ensemble.py`
2. `test_ace_dinov2_lmc.py --ensemble_networks ...`

最新联合评测结果：

输出目录：

`/home/xwh/project/ace_depth/ace_dinov2_lmc/04_evaluation/train_compare/memory_pooled_vs_asb/indoor6_ace/scene3/dino_ace_lmc_ace_g/cluster_ensemble_eval`

指标：

- `25cm/5deg = 99.05%`
- `10cm/5deg = 92.06%`
- `5cm/5deg = 68.89%`
- `2cm/2deg = 29.84%`
- median = `0.63 deg / 3.08 cm`

选择分布：

- `cluster01`: 106 frames
- `cluster02`: 209 frames

这说明：

1. `sub memory` 不是伪改动，确实有互补性。
2. scene3 上 “切分后独立训练，再聚合评测” 已经被验证有效。
3. 当前 cluster 路线已经具备研究意义，不应再回退到“只看单一 fallback memory”。

## 关于参考坐标系 gap 的最新判断

当前最重要的概念结论：

1. MapAnything 的 memory feature 是 reference-conditioned。
2. 当前主训练大多还是 `C0`，监督点云在 `points_world`。
3. 对 `sub memory` 路线来说，这之间存在真实的语义 gap。

因此下一阶段主线应当是：

1. `sub memory` 内部切到 `C1`
2. 每个 cluster 的点云、监督目标、回归输出都在各自 `ref_i` 下定义
3. 推理时每个 cluster 输出 `P_ref_i`
4. 再通过 `T_ref_i_c2w_world` 恢复到 world 做 DSAC / 统计

但要保留一个已经验证有效的原则：

- `scene_center` 的定义继续沿用“相机中心均值”
- 不改成点云质心

更严格的实现建议：

- 同时保存 `scene_center_world`
- 同时保存 `scene_center_ref`

这样在 `C1` 下才真正一致：

- head / fusion 用 `scene_center_ref`
- world 恢复和日志保留 `scene_center_world`

## 关于 ACE Cambridge 路线的借鉴

仓库根 README 已明确说明 ACE Poker / Cambridge 的 cluster 路线：

- 先做 spatial clustering
- 每个 cluster 单独训练 head
- 最后 ensemble merge

对我们有用的借鉴不是“必须多模型”，而是：

- 先切子数据域，再在子数据域内构建 memory

所以建议后续不要只在全场景 selected views 上做后处理式 cluster builder，而是显式引入：

- `cluster manifest`
- `sub-dataset view list`
- 每个 cluster 内独立 reference 选择
- 每个 cluster 内独立 `scene_center_world/ref`
- 每个 cluster 内独立 `C1` memory contract

## 当前任务判断

基于现状，当前任务优先级应当重排为：

1. 不再重复跑 `scene3` 的单一 fallback memory 训练。
2. 不再把“两个独立 checkpoint + ensemble eval”当作最终形态。
3. 当前最重要任务是把验证推进到：
   - `sub-dataset split`
   - `C1 reference-consistent sub memory`
   - `single shared model + multiple memory files + post-hoc pose selection`

换句话说，后续不该继续做“两个模型分别训练”的重复实验，而应该做：

## 下一阶段目标

### Goal A: Shared-model multi-memory prototype

目标：

- 一个共享的 backbone / compressor / fusion / regressor head
- 多个 memory 文件作为外部条件输入
- 同一套权重可以在不同 sub memory 上工作
- 推理时 query backbone 只算一次
- 对每个 memory 分支独立输出 pose
- 最后按几何质量选 winner

### Goal B: C1-first sub memory

目标：

- 每个 cluster memory 默认走 `C1`
- supervision target = `points_ref` 或 `points_ref_norm`
- eval 时恢复到 world

### Goal C: 保持 independent ensemble，不做 cross-cluster feature fusion

第一版明确不做：

- cross-cluster feature fusion
- pose averaging
- learned routing

第一版继续使用：

- independent prediction
- world-frame pose recovery
- DSAC inlier count / geometry quality selection

## 推荐的实际实现顺序

1. extraction schema 升级
   - 为 cluster memory 显式补充 `scene_center_world`
   - 为 cluster memory 显式补充 `scene_center_ref`
   - 确认 `C1` 下 cluster memory 的保存字段完整

2. sub-dataset manifest
   - 生成每个 cluster 的 view list / manifest
   - 让 cluster construction 更接近 ACE Cambridge 的“先分域再训练”

3. shared-model trainer prototype
   - 一个 checkpoint
   - 训练时 batch 绑定某一个 cluster memory
   - 同一套权重轮流消费多个 memory

4. shared-model query-time multi-memory eval
   - query backbone 一次
   - 多 memory 分支独立 forward
   - 恢复到 world
   - DSAC 选 winner

5. router 作为最后一步
   - 只有在 shared-model multi-memory 已成立之后，才考虑 learned router 或 pre-routing

## 明确哪些任务还没完成

以下任务仍然没有闭环：

1. `Q1b`：同一 selected set 下真正验证 `C0 vs C1`
2. `sub memory` 的 `C1` 正式 extraction + training + eval
3. `shared-model multi-memory` 训练路径
4. `clustered package` 作为训练输入的统一接口
5. learned router / pre-routing
6. 更稳定的 deterministic eval 或 DSAC eval mode 改造

## 当前最合理的下一步

如果继续从这里推进，优先做：

1. 先把 `Q1b` 结论写死：
   - `scene2a` 的 single-memory `C1` 已经 end-to-end 跑通
   - 但当前 same-recipe 下仍然落后于 repaired `C0`
   - 默认 single-memory policy 暂不切换到 `C1`
2. `scene3 cluster-local C1` 已经闭环：
   - `scene3` 的 `C1` cluster-fallback extraction 已完成
   - `cluster_01` / `cluster_02` 两个训练已完成
   - ensemble eval 已完成，结果接近旧 `C0` cluster ensemble，
     但仍略弱一些
3. 只有在这条 `scene3 cluster-local C1` 结果被正式记入结果文档之后，再考虑 shared-model
   multi-memory

而不是：

1. 再重复跑更多单 cluster 训练
2. 再刷同配置 single-memory baseline
3. 提前做 learned router

## 不要混入当前主线的未来研究线

下面这条线先不要和当前 `memory_extraction` 主线混在一起：

- 面向未来大规模多场景预训练的 `C1 + normalized target` 设计
- 包括：
  - weaker normalization / scaled normalized target
  - metric auxiliary loss

这部分已经单独记录在仓库根文档：

- `/home/xwh/project/ace_depth/ace_dinov2_lmc/C1_NORMALIZED_PRETRAINING_MEMO.md`

注意：

- 该 memo 目前只是设计备忘录
- 当前代码里还没有可直接运行的 `alpha` / weaker-normalization CLI 开关
- 在 `scene3 cluster-local C1` 没闭环之前，不要把这条未来研究线并入当前实验矩阵

## `scene3 cluster-local C1` 之后的最小机制验证矩阵

如果 `scene3 cluster-local C1` 已闭环，而我们要继续回答
“normalized-C1 为什么会掉高精度指标”，建议只做 `scene2a` 单场景小矩阵：

1. `C1 points_ref_norm(alpha=2)`
   - 保留 normalized target，只降低归一化强度
2. `C1 points_ref_norm + aux_ref_loss`
   - 主目标仍是 normalized `C1`
   - 增加一个小权重 reference-frame metric supervision
3. 第三个 run 只在前两个没有充分回答问题时再补：
   - `alpha=4`
   - 或调一个 aux 权重
   - 或组合版 `alpha=2 + aux_ref_loss`

判定标准固定：

- 主看 `acc5` 和 median translation
- 约束 `acc25` 不应明显恶化
- 若评测开销可接受，关键 checkpoint 默认重复评 `5` 次
- 继续按同一实验目录下全部 `*_eval_log.txt` 和 `eval_summary_*.txt`
  的 best-of 口径统计；备注里应写明 `best of 5 eval repeats`

不要把这个小矩阵扩成新的 cross-scene sweep，也不要在这一步就引入
refinement head / residual branch / shared-model 联合研究。

## 现在最推荐的下一步

不要再继续刷 `scene3`。

当前最推荐的下一步是：

1. deterministic eval 保留为“需要严格横向复核时”的工具，而不是硬性默认：
   - 必要时可加 `--eval_deterministic True`
   - `--dsacstar_seed 1305`
   - `--dsacstar_seed_per_frame True`
   - 不开 deterministic 时，关键 checkpoint 默认重复评 `5` 次，再按
     best-of 记结果
2. 在 `scene2a` 上做第一个最小机制验证：
   - `C1 points_ref_norm(alpha=2)`
3. 只在 `alpha=2` 不能解释问题时，再做第二个：
   - `C1 points_ref_norm + aux_ref_loss`

也就是说，下一步不再是“多场景扩展”，而是
“在 `scene2a` 单场景上用最小矩阵判断 fine-precision 掉点究竟来自
归一化强度，还是需要额外 metric supervision”。

## 当前已知关键文件

核心代码：

- `memory_extraction/run_memory_extraction.py`
- `memory_extraction/extract_memory.sh`
- `trainer_dinov2_lmc.py`
- `test_ace_dinov2_lmc.py`
- `test_ace_dinov2_lmc_ensemble.py`
- `utils_lmc.py`

关键文档：

- `memory_extraction/03_implementation/EXECUTION_PLAN_reference_consistent_memory.md`
- `memory_extraction/03_implementation/PROGRESS_reference_consistent_memory.md`
- `memory_extraction/03_implementation/HANDOFF_PROMPT.md`

请从这些文件恢复工作，而不是依赖旧的 `IMPLEMENTATION_NOTES*.md` 来判断当前阶段。
