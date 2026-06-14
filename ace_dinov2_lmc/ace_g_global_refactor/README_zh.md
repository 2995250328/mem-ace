# ACE-G 规划文档总导航（中文）

**绝对根目录：** `/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor`

## 1. 这份文档是干什么的

这份文档是当前 ACE-G 规划语料的**总导航入口**，对应目录：

```text
ace_dinov2_lmc/ace_g_global_refactor/
```

它的目标很简单：

1. 告诉你**遇到某个问题时先看哪个文件**；
2. 告诉你**每个规划文档放在哪里**；
3. 给每个重要文档一个**简短内容概述**；
4. 区分清楚：
   - 高层总计划
   - 实验参考/基线结论
   - 分步骤实施文档
   - 长期研究方向

这份文档本身只是导航，不是实验结果真值表，也不是某个单独方向的实现方案。

---

## 2. 快速入口

### 核心入口文件

- [PLAN.md](./PLAN.md)
  绝对路径：`/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor/PLAN.md`
- [PLAN_CN.md](./PLAN_CN.md)
  绝对路径：`/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor/PLAN_CN.md`
- [EXPERIMENT_REFERENCE.md](./EXPERIMENT_REFERENCE.md)
  绝对路径：`/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor/EXPERIMENT_REFERENCE.md`
- [LONG_TERM_RESEARCH_PLAN.md](./LONG_TERM_RESEARCH_PLAN.md)
  绝对路径：`/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor/LONG_TERM_RESEARCH_PLAN.md`
- [COMPARE_CPE_B1_scene2a_20260518.md](./COMPARE_CPE_B1_scene2a_20260518.md)
  绝对路径：`/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor/COMPARE_CPE_B1_scene2a_20260518.md`
- [SFM_TRACK_GUIDED_MULTI_VIEW_SUPERVISION_ZH.md](./SFM_TRACK_GUIDED_MULTI_VIEW_SUPERVISION_ZH.md)
  绝对路径：`/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor/SFM_TRACK_GUIDED_MULTI_VIEW_SUPERVISION_ZH.md`

### 当前最重要的路线文件

- [SFM_TRACK_GUIDED_MULTI_VIEW_SUPERVISION_ZH.md](./SFM_TRACK_GUIDED_MULTI_VIEW_SUPERVISION_ZH.md)
  绝对路径：`/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor/SFM_TRACK_GUIDED_MULTI_VIEW_SUPERVISION_ZH.md`
- [steps/15_pointrope_and_multilevel_fusion_roadmap.md](./steps/15_pointrope_and_multilevel_fusion_roadmap.md)
  绝对路径：`/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor/steps/15_pointrope_and_multilevel_fusion_roadmap.md`
- [steps/16_large_scene_multi_memory_reference_frame_plan.md](./steps/16_large_scene_multi_memory_reference_frame_plan.md)
  绝对路径：`/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor/steps/16_large_scene_multi_memory_reference_frame_plan.md`
- [steps/16_large_scene_multi_memory_reference_frame_plan_zh.md](./steps/16_large_scene_multi_memory_reference_frame_plan_zh.md)
  绝对路径：`/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor/steps/16_large_scene_multi_memory_reference_frame_plan_zh.md`
- [steps/17_view_token_memory_usage_plan.md](./steps/17_view_token_memory_usage_plan.md)
  绝对路径：`/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor/steps/17_view_token_memory_usage_plan.md`
- [steps/17_view_token_memory_usage_plan_zh.md](./steps/17_view_token_memory_usage_plan_zh.md)
  绝对路径：`/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor/steps/17_view_token_memory_usage_plan_zh.md`
- [steps/19_rio10_conservative_indoor6_migration_plan.md](./steps/19_rio10_conservative_indoor6_migration_plan.md)
  绝对路径：`/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor/steps/19_rio10_conservative_indoor6_migration_plan.md`
- [steps/20_rio10_generalization_diagnostic_plan.md](./steps/20_rio10_generalization_diagnostic_plan.md)
  绝对路径：`/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor/steps/20_rio10_generalization_diagnostic_plan.md`

---

## 3. 推荐阅读顺序

如果你中断了一段时间，现在重新回来继续 ACE-G，推荐按下面顺序看。

### A. 我想先看全局情况

先看：

1. [PLAN.md](./PLAN.md)
2. [EXPERIMENT_REFERENCE.md](./EXPERIMENT_REFERENCE.md)
3. [steps/15_pointrope_and_multilevel_fusion_roadmap.md](./steps/15_pointrope_and_multilevel_fusion_roadmap.md)

你会得到：

- 当前 ACE-G 的问题地图；
- 已接受的 baseline 和已否定方向；
- 当前近期开工优先级。

### B. 我想发起一个新实验，或者判断一个实验值不值得做

先看：

1. [EXPERIMENT_REFERENCE.md](./EXPERIMENT_REFERENCE.md)
2. 对应的 step 文档（在 `steps/` 下）

这样做的原因是：

- 前者帮你避免重复跑已经失败的方向；
- 后者告诉你这个机制原本是怎么设计的、有哪些约束。

### C. 我只想看某个具体方向

直接去 `steps/` 下面找对应文档：

```text
/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor/steps/
```

后面的“Step 索引”会给你逐个说明。

### D. 我想看更长远、更激进的方向，而不是当前马上要做的

先看：

1. [LONG_TERM_RESEARCH_PLAN.md](./LONG_TERM_RESEARCH_PLAN.md)
2. 如果需要，再配合 [PLAN.md](./PLAN.md)

这里记录的是那些**有价值，但不适合立刻塞进当前稳定化主线**的方向。

---

## 4. 顶层文件分别是干什么的

### [PLAN.md](./PLAN.md)
绝对路径：`/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor/PLAN.md`

定位：

- 高层总计划；
- 问题地图；
- 优先级框架；
- 新 session / 新 agent 的 onboarding 入口。

什么时候看它：

- 你想知道当前 ACE-G 主要在解决哪些问题；
- 你想知道哪些问题已经基本稳定，哪些还没解决；
- 你想从高层理解整个 global refactor。

注意：

- 它自己明确说明：**具体实施说明在 `steps/` 里按文件分开维护**。

### [PLAN_CN.md](./PLAN_CN.md)
绝对路径：`/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor/PLAN_CN.md`

定位：

- `PLAN.md` 的中文同步版。

什么时候看它：

- 你想用中文快速理解高层计划时。

### [EXPERIMENT_REFERENCE.md](./EXPERIMENT_REFERENCE.md)
绝对路径：`/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor/EXPERIMENT_REFERENCE.md`

定位：

- 当前实验参考真值表；
- baseline 决策文件；
- “哪些方向不要原样重跑”的规则表。

什么时候看它：

- 你准备发新实验前；
- 你想核对 accepted baseline；
- 你想查某个方向是否已经失败过；
- 你想查 FGPI-4090 一类参考指标。

这个文件应该是**任何新实验前的第一站**。

### [LONG_TERM_RESEARCH_PLAN.md](./LONG_TERM_RESEARCH_PLAN.md)
绝对路径：`/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor/LONG_TERM_RESEARCH_PLAN.md`

定位：

- 存放长期方向；
- 存放那些一次会改多个 contract、暂时不适合并入当前主线的想法。

什么时候看它：

- 你的想法涉及 multi-scene / multi-memory / joint training；
- 你的方向明显超出当前 baseline stabilization 范围；
- 你不确定某个想法该不该立刻并入 Step15/16 主线时。

### [COMPARE_CPE_B1_scene2a_20260518.md](./COMPARE_CPE_B1_scene2a_20260518.md)
绝对路径：`/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor/COMPARE_CPE_B1_scene2a_20260518.md`

定位：

- 某个具体实验家族的详细比较说明。

什么时候看它：

- 你想知道 CPE / B1 结论背后的细节；
- 你需要比 `EXPERIMENT_REFERENCE.md` 更细的上下文。

---

## 5. `steps/` 目录怎么理解

`steps/` 目录里的文档，原则上是：

> 一个文档对应一个具体方向、一个 contract 修复、或者一条明确的实施路线。

大体可以分三类：

### A 类：基础 contract / 语义 / hygiene

这些文档解决的是：

- 实验语义是否可靠；
- 训练/测试行为是否一致；
- 日志和配置是否可解释。

它们是所有后续架构改动的地基。

### B 类：具体架构方向

这些文档讨论具体方法，比如：

- geometry injection；
- multi-level compression/fusion；
- PointRoPE；
- 多 memory；
- view token 使用。

### C 类：综合 roadmap

这些文档不是只讲一个机制，而是把多个方向一起排序、归纳、给出下一步优先级。

---

## 6. Step 逐个索引

### Step 01 — [steps/01_deterministic_fps.md](./steps/01_deterministic_fps.md)
绝对路径：`/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor/steps/01_deterministic_fps.md`

主题：

- deterministic FPS。

作用：

- 去掉 latent 坐标构造中的随机性。

适合什么时候看：

- 你在检查 memory 构造是否可复现时。

---

### Step 02 — [steps/02_low_risk_contract.md](./steps/02_low_risk_contract.md)
绝对路径：`/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor/steps/02_low_risk_contract.md`

主题：

- 低风险 contract 清理。

作用：

- 在更大结构改动之前先把基础语义理顺。

适合什么时候看：

- 你在追踪 ACE-G 初期 contract/hygiene 改动时。

---

### Step 03 — [steps/03_s1_sampled_loss_step_mode.md](./steps/03_s1_sampled_loss_step_mode.md)
绝对路径：`/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor/steps/03_s1_sampled_loss_step_mode.md`

主题：

- S1 sampled loss 的 step-mode 语义。

作用：

- 帮助理解为什么 `s1_loss_step_mode=per_iter` 会变重要。

适合什么时候看：

- 你在复盘 S1 行为，或者比较旧版/新版 S1 设定时。

---

### Step 04 — [steps/04_lmc_mode_authority.md](./steps/04_lmc_mode_authority.md)
绝对路径：`/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor/steps/04_lmc_mode_authority.md`

主题：

- requested `lmc_mode` 与 effective `lmc_mode` 的 authority 问题。

作用：

- 防止命令写着 `global`，运行时却偷偷变成 effective-local。

适合什么时候看：

- 你要核对某次 run 是否真的 global；
- 你在审查 fallback 逻辑时。

---

### Step 05 — [steps/05_geomatch_near_term.md](./steps/05_geomatch_near_term.md)
绝对路径：`/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor/steps/05_geomatch_near_term.md`

主题：

- 早期 fusion-geometry / GeoMatch / geometry injection 方向。

作用：

- 解释最早那批 geometry-side fusion 改动是怎么立题的。

适合什么时候看：

- 你想追溯 A0/A1/A2、progressive geometry injection 的来源时。

---

### Step 06 — [steps/06_compressor_geometry_contract.md](./steps/06_compressor_geometry_contract.md)
绝对路径：`/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor/steps/06_compressor_geometry_contract.md`

主题：

- compressor geometry contract。

作用：

- 规定 key layer、坐标尺度、多层 memory 的理解方式。

适合什么时候看：

- 你在思考 key/value hierarchy；
- 你在设计 multi-level compression；
- 你在核对 scene-scale / geometry 假设。

---

### Step 07 — [steps/07_loss_contract.md](./steps/07_loss_contract.md)
绝对路径：`/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor/steps/07_loss_contract.md`

主题：

- loss path contract。

作用：

- 避免 S1/S2 loss 路径差异被误判成架构问题。

适合什么时候看：

- 你在核对 reprojection supervision 一致性时。

---

### Step 08 — [steps/08_non_arch_hygiene.md](./steps/08_non_arch_hygiene.md)
绝对路径：`/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor/steps/08_non_arch_hygiene.md`

主题：

- 非架构类 hygiene 工作。

作用：

- 在不改模型结构的前提下提升实验语义和可复现性。

适合什么时候看：

- 你在检查日志、配置、记录规范等问题时。

---

### Step 09 — [steps/09_module_mode_contract.md](./steps/09_module_mode_contract.md)
绝对路径：`/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor/steps/09_module_mode_contract.md`

主题：

- module mode consistency。

作用：

- 让训练/测试/冻结模块行为更容易解释。

适合什么时候看：

- 你在追踪模块 train/eval/frozen 行为时。

---

### Step 10 — [steps/10_runtime_observability_and_semantics.md](./steps/10_runtime_observability_and_semantics.md)
绝对路径：`/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor/steps/10_runtime_observability_and_semantics.md`

主题：

- runtime observability。

作用：

- 补充 token usage、attention entropy、norm 等诊断手段。

适合什么时候看：

- 你要知道当前系统记录了哪些运行时行为，或者还缺哪些诊断时。

---

### Step 11 — [steps/11_levelwise_latent_merge_plan.md](./steps/11_levelwise_latent_merge_plan.md)
绝对路径：`/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor/steps/11_levelwise_latent_merge_plan.md`

主题：

- levelwise latent merge / B3-lite 方向。

作用：

- 这是那个重要但失败的 multi-level compression 尝试，后续很多更保守设计都是从这里反思出来的。

适合什么时候看：

- 你想了解为什么 B3-lite 被否掉时。

---

### Step 12 — [steps/12_geobias_rbf_and_fourier_pe_v2_plan.md](./steps/12_geobias_rbf_and_fourier_pe_v2_plan.md)
绝对路径：`/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor/steps/12_geobias_rbf_and_fourier_pe_v2_plan.md`

主题：

- geometry bias RBF 和 Fourier PE v2。

作用：

- 记录 geometry-conditioning 分支里 bias redesign 和 Fourier PE 改进的思路。

适合什么时候看：

- 你在回看 GBR-main / FPE-main 这条线时。

---

### Step 13 — [steps/13_query_side_dpt_adapter_multilevel_plan.md](./steps/13_query_side_dpt_adapter_multilevel_plan.md)
绝对路径：`/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor/steps/13_query_side_dpt_adapter_multilevel_plan.md`

主题：

- query-side DPT-style / multi-level adapter 规划。

作用：

- 这是原始的 backbone-side multi-level dense prediction 方向。

适合什么时候看：

- 你想了解早期 multi-level 思路在 query/backbone 侧是怎么设计的。

当前状态：

- 仍然是有价值的背景文档，但不再是当前最优先的首个实现版本。

---

### Step 14 — [steps/14_point_rope_and_continuous_relative_bias_plan.md](./steps/14_point_rope_and_continuous_relative_bias_plan.md)
绝对路径：`/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor/steps/14_point_rope_and_continuous_relative_bias_plan.md`

主题：

- PointRoPE 和 continuous relative bias 规划。

作用：

- 是当前 PointRoPE 主线的直接前身。

适合什么时候看：

- 你在看 geometry encoding / bias 这条线的前置设定时。

---

### Step 15 — [steps/15_pointrope_and_multilevel_fusion_roadmap.md](./steps/15_pointrope_and_multilevel_fusion_roadmap.md)
绝对路径：`/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor/steps/15_pointrope_and_multilevel_fusion_roadmap.md`

主题：

- 当前**近期主路线图**。

作用：

- 这是现在最重要的 step 文档之一。

它当前整合了：

- accepted FGPI-style reference；
- 已完成 geometry 结果；
- 已拒绝/已延期方向；
- PointRoPE / fusion-side refinement / compression-side residual multi-level / bias 这些方向的优先级排序；
- 更新后的 Candidate C 拆分：
  - 优先 C1 = Cascading Internal Fusion Assembly；
  - 延后 C2 = backbone multi-layer query adapter。

适合什么时候看：

- 你想知道“现在下一步该做什么”；
- 你想看当前架构方向优先级；
- 你想看多个实验家族整合后的最新综合判断。

---

### Step 16 — [steps/16_large_scene_multi_memory_reference_frame_plan.md](./steps/16_large_scene_multi_memory_reference_frame_plan.md)
绝对路径：`/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor/steps/16_large_scene_multi_memory_reference_frame_plan.md`

主题：

- 大场景 multi-memory + reference-frame planning。

作用：

- 讨论当多个 MapAnything memory 处在不同 reference-conditioned 坐标/特征体系下时，该怎么组织系统。

核心结论：

- 第一阶段不要直接拼接多个 memories；
- 优先用 independent local memory bundles + routing / 几何打分选 pose。

适合什么时候看：

- 你在想大场景；
- 你在做 multi-memory routing；
- 你在考虑多个 local memory 如何组织。

中文参考版：

- [steps/16_large_scene_multi_memory_reference_frame_plan_zh.md](./steps/16_large_scene_multi_memory_reference_frame_plan_zh.md)
  绝对路径：`/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor/steps/16_large_scene_multi_memory_reference_frame_plan_zh.md`

---

### Step 17 — [steps/17_view_token_memory_usage_plan.md](./steps/17_view_token_memory_usage_plan.md)
绝对路径：`/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor/steps/17_view_token_memory_usage_plan.md`

主题：

- memory-side view token（`all_scale_tokens`）的保守使用方案。

作用：

- 记录如何在不污染当前 baseline 的情况下利用 memory 里存储的 per-view global token。

核心结论：

- 把 `all_scale_tokens` 视为 side-channel；
- 先 diagnostics；
- 再做 optional attention pooling / latent modulation；
- 后面再考虑 memory routing；
- 现在不要把 raw token 直接塞进 dense fusion；
- 现在不要启用 query CLS guidance。

适合什么时候看：

- 你在思考 camera/view token 怎么用；
- 你在判断这些 token 更适合 memory-side modulation 还是未来 routing。

中文参考版：

- [steps/17_view_token_memory_usage_plan_zh.md](./steps/17_view_token_memory_usage_plan_zh.md)
  绝对路径：`/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor/steps/17_view_token_memory_usage_plan_zh.md`

---

### Step 18 — [steps/18_rio10_wai_training_fix_log.md](./steps/18_rio10_wai_training_fix_log.md)
绝对路径：`/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor/steps/18_rio10_wai_training_fix_log.md`

主题：

- RIO10 WAI/ACE backend resize 和 sparse-depth attachment 修复记录。

作用：

- 记录为什么直接 WAI training 会触发 DINOv2 patch-size 约束；
- 记录修正后的 ACE-backend RIO10 training 路径；
- 记录 RIO10 sparse-depth matching 行为。

适合什么时候看：

- 你在调试 RIO10 dataset/backend 问题；
- 你要检查 ACE-backend training 的 sparse-depth attachment；
- 你在准备 Step19 的保守 RIO10 迁移实验。

---

### Step 19 — [steps/19_rio10_conservative_indoor6_migration_plan.md](./steps/19_rio10_conservative_indoor6_migration_plan.md)
绝对路径：`/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor/steps/19_rio10_conservative_indoor6_migration_plan.md`

主题：

- 从 Indoor6 最佳 ACE-G baseline 到 RIO10 的保守迁移方案。

作用：

- 在加入新架构之前，定义最低风险的 RIO10 stabilization recipe。

核心结论：

- 保留 true-global `per_iter` Indoor6 baseline 语义；
- RIO10 training 使用 ACE backend；
- memory extraction 使用 WAI/MapAnything ASB single-forward global memory；
- 使用 `patch_depth_sampling=nearest_valid`；
- sparse depth 只用于 valid-coordinate guided sampling；
- 关闭 auxiliary sparse-depth/reference supervision。

适合什么时候看：

- 你要发起下一轮 RIO10 stabilization run；
- 你要比较 guided sampling vs no guided sampling；
- 你要判断在 PointRoPE/CIFA/multi-memory 扩展前哪些变量必须固定。

中文参考版：

- [steps/19_rio10_conservative_indoor6_migration_plan_zh.md](./steps/19_rio10_conservative_indoor6_migration_plan_zh.md)
  绝对路径：`/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor/steps/19_rio10_conservative_indoor6_migration_plan_zh.md`

---

### Step 20 — [steps/20_rio10_generalization_diagnostic_plan.md](./steps/20_rio10_generalization_diagnostic_plan.md)
绝对路径：`/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor/steps/20_rio10_generalization_diagnostic_plan.md`

主题：

- 在继续架构修改之前，诊断 RIO10 泛化很差的问题。

作用：

- 区分 data/eval contract 问题、vanilla DINO ACE baseline 本身弱、LMC/memory/fusion 失败、sparse-guided sampling bias，以及训练参数影响。

核心结论：

- 当前 RIO10 run 只能作为 incomplete partial evidence；
- 先做 data/pose/calibration audit 和 train-set eval；
- 建立同 split 的 vanilla DINO ACE baseline；
- 然后再 ablate sparse sampling ratio、S2 fusion、training parameters 和 memory coverage；
- 在失败模式明确前，不要加入 PointRoPE/CIFA/multi-memory/view-token 改动。

适合什么时候看：

- RIO10 泛化很弱，需要定位原因；
- 判断问题来自 data/eval、baseline，还是 LMC-specific；
- 在更多架构工作前规划下一轮诊断实验矩阵。

中文参考版：

- [steps/20_rio10_generalization_diagnostic_plan_zh.md](./steps/20_rio10_generalization_diagnostic_plan_zh.md)
  绝对路径：`/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor/steps/20_rio10_generalization_diagnostic_plan_zh.md`

---

## 7. 按任务找文件

### “我想知道当前 accepted baseline 是什么？”

看：

- [EXPERIMENT_REFERENCE.md](./EXPERIMENT_REFERENCE.md)

### “我想知道当前总体架构路线图是什么？”

看：

- [PLAN.md](./PLAN.md)
- [steps/15_pointrope_and_multilevel_fusion_roadmap.md](./steps/15_pointrope_and_multilevel_fusion_roadmap.md)

### “我想知道哪些方向已经不要原样重跑了？”

看：

- [EXPERIMENT_REFERENCE.md](./EXPERIMENT_REFERENCE.md)

### “我想理解 multi-level compression / fusion 的历史演化？”

看：

- [steps/11_levelwise_latent_merge_plan.md](./steps/11_levelwise_latent_merge_plan.md)
- [steps/13_query_side_dpt_adapter_multilevel_plan.md](./steps/13_query_side_dpt_adapter_multilevel_plan.md)
- [steps/15_pointrope_and_multilevel_fusion_roadmap.md](./steps/15_pointrope_and_multilevel_fusion_roadmap.md)

### “我想知道当前优先的 multi-level 方向是什么？”

看：

- [steps/15_pointrope_and_multilevel_fusion_roadmap.md](./steps/15_pointrope_and_multilevel_fusion_roadmap.md)

重点：

- Candidate C1 = Cascading Internal Fusion Assembly。

### “我想看 large-scene / multi-memory 规划。”

看：

- [steps/16_large_scene_multi_memory_reference_frame_plan.md](./steps/16_large_scene_multi_memory_reference_frame_plan.md)

### “我想看 camera/view token 的规划。”

看：

- [steps/17_view_token_memory_usage_plan.md](./steps/17_view_token_memory_usage_plan.md)

### “我想把 Indoor6 最佳 baseline 保守迁移到 RIO10。”

看：

- [steps/19_rio10_conservative_indoor6_migration_plan.md](./steps/19_rio10_conservative_indoor6_migration_plan.md)
- [steps/18_rio10_wai_training_fix_log.md](./steps/18_rio10_wai_training_fix_log.md)

### “我想诊断为什么 RIO10 泛化很差。”

看：

- [steps/20_rio10_generalization_diagnostic_plan.md](./steps/20_rio10_generalization_diagnostic_plan.md)
- [steps/19_rio10_conservative_indoor6_migration_plan.md](./steps/19_rio10_conservative_indoor6_migration_plan.md)
- [steps/18_rio10_wai_training_fix_log.md](./steps/18_rio10_wai_training_fix_log.md)

### “我想看更长远、不一定马上做的方向。”

看：

- [LONG_TERM_RESEARCH_PLAN.md](./LONG_TERM_RESEARCH_PLAN.md)

---

## 8. 实际使用规则

在提出任何新的 ACE-G 实验前，建议先过这个最小清单：

1. 先看 [EXPERIMENT_REFERENCE.md](./EXPERIMENT_REFERENCE.md)；
2. 确认这个方向不是已经被否掉或重复的；
3. 再看对应的 step 文档；
4. 如果需要整体上下文，再看 [PLAN.md](./PLAN.md)；
5. 如果这个想法一次改多个 contract，就优先把它归入 [LONG_TERM_RESEARCH_PLAN.md](./LONG_TERM_RESEARCH_PLAN.md) 范畴，而不是硬塞进当前近期开工路线。

---

## 9. 最后浓缩版

如果你只记四个文件，就记这四个：

1. [PLAN.md](./PLAN.md)
   绝对路径：`/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor/PLAN.md`
2. [EXPERIMENT_REFERENCE.md](./EXPERIMENT_REFERENCE.md)
   绝对路径：`/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor/EXPERIMENT_REFERENCE.md`
3. [steps/15_pointrope_and_multilevel_fusion_roadmap.md](./steps/15_pointrope_and_multilevel_fusion_roadmap.md)
   绝对路径：`/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor/steps/15_pointrope_and_multilevel_fusion_roadmap.md`
4. [LONG_TERM_RESEARCH_PLAN.md](./LONG_TERM_RESEARCH_PLAN.md)
   绝对路径：`/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor/LONG_TERM_RESEARCH_PLAN.md`

其余 `steps/` 下的文件，就是对应某个具体方向、具体 contract、或者某段规划历史的局部文档。
