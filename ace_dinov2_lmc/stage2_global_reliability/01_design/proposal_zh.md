# 标题：面向 ACE-FCN-LMC Stage2 的可靠性可控 Global Conditioning

## 1. 问题定义
给定一个 Wayspots 场景，包括训练图像 \(I_i\)、相机位姿/内参、稀疏坐标监督，以及已提取的 ACE-FCN memory \(M\)，Stage1 学习一个 local ACE-FCN-LMC 坐标预测器 \(f_L(I, M) \rightarrow X\)。Stage2 进一步加入图像级 GLACE global feature \(g_i\)，得到 \(f_{LG}(I, M, g_i) \rightarrow X\)。目标是在提升重定位性能的同时，限制相对于 Stage1 的退化：
\[
\min_\theta \; L_{reproj}(f_{LG}) + \lambda_c L_{cons}(f_{LG}, f_L) + \lambda_g R(gate/global)
\]
可靠性机制应当允许 Bears 这类场景使用有效的 global cue，同时抑制 SquareBench 这类场景中有害的 global cue。

## 2. 文献背景与动机
DSAC*/ACE 等 scene coordinate regression 方法主要依赖密集局部图像证据和 RANSAC；当局部几何足够时，这类方法较稳健。特征条件化或场景 memory 方法会加入更大范围上下文，但 global descriptor 可能携带场景级先验，而且这种先验在不同场景中并不总是可靠。Mixture-of-experts、residual adapter、knowledge-distillation consistency loss 处理的是类似失败模式：新引入的上下文只应在可信时改善预测。当前 Stage2 实验直接暴露了这个缺口：很小的 scalar gate 上界能保护 SquareBench，但 Bears 需要更强 global 注入。因此，global conditioning 需要可靠性控制，而不是单一固定 scalar gate。

## 3. 设计空间与选定架构
方案 A 是按场景用验证集选择 `global_gate_max`。它简单、成本低，但不能按图像自适应，也容易对小验证集过拟合。方案 B 是 adaptive/residual global conditioning，并加入 Stage1 consistency。它稍复杂，但能直接针对负迁移：让 Stage2 成为一个保守地改进已知 local predictor 的 residual。第一版选定架构是：保留现有 Stage2 global concat 路径，为采样训练点或 feature map 加入 Stage1 teacher prediction，增加按 local confidence/error 加权的 consistency loss，并尽可能把 raw global concat 改成 zero-initialized residual/adaptive gate branch。第一轮实现应尽量最小化：先做 consistency loss 加 bounded gate/residual regularization，再考虑完整 dual-head 设计。

## 4. 评估与验证计划
主验证场景：`wayspots_bears` 与 `wayspots_squarebench`。指标：median rotation/translation error、Acc50/5deg、Acc25/5deg、Acc10/5deg、Acc5/5deg，沿用现有 post-train seeds 和 hypotheses。基线：Stage1 local、原始 Stage2 concat、bounded scalar gates max=0.01/max=0.1、zero global、random global。消融：consistency 权重、residual/global regularization、adaptive gate 类型、zero-init 与非 zero-init、checkpoint selection metric。成功标准：SquareBench 的 Stage2 Acc5 相比 Stage1 local 只允许小幅下降，同时 Bears 尽量接近 raw Stage2 Acc5，并且不让 SquareBench 崩溃。

## 5. 预期失败模式与工程风险
Consistency loss 可能过度限制 Stage2，使 Bears 的有效 global 改进被阻断。Adaptive gate 如果只靠 reprojection loss 训练，可能学到捷径。Per-image reliability 可能需要训练时没有的验证标签。Eval 必须和训练侧 gate/residual 逻辑完全一致，否则 checkpoint 结果会误导。Stage2 checkpoint 保存必须包含所有新增 gate/residual 状态和语义配置字段。

## 6. 复用计划
复用 `../options_dinov2_lmc.py` 添加新 CLI flags，并保留现有 gate flags。复用 `../trainer_dinov2_lmc.py` 中的 global feature mode、gate 初始化、Stage2 optimizer membership、local/global feature 拼接、consistency loss 插入点和 checkpoint 序列化。复用 `../test_ace_dinov2_lmc.py` 在 eval 端重建相同 gate/residual 行为。复用 `../scripts/run_squarebench_stage2_global_gate_matrix.sh` 跑 Bears/SquareBench 快速 variants。复用 `../scripts/summarize_wayspots_ace_fcn_lmc_suite.py` 汇总报告。`../memory_extraction/extract_memory_ace_fcn.py` 与 `../ace_fcn_lmc/ace_network_ace.py` 定义 Stage1 memory 与 local feature contract；除非验证需要，否则不应修改。