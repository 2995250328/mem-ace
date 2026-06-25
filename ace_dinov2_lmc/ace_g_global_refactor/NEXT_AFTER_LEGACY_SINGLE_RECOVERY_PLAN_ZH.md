# Legacy Single 恢复后的 PMRF 决策规划

更新时间：2026-06-22

状态：当前正在运行 `legacy_single_k64_repro_20260622_gpu01`；完成前不启动 PMRF 新训练。

本文是 PMRF-v3 主规划的前置决策状态机，负责决定 baseline 恢复后是先重训 v2、修复复现合同，还是进入新结构实现。

## 1. 当前任务与完成条件

当前启动器按顺序执行：

1. 当前代码的 legacy single K64，8个 Wayspots 场景；
2. commit `65789847cb0c8f3b0f5cc5450c6f19844b275b90` worktree 的同合同复现。

场景为 Bears、Squarebench、Cubes、Tendrils、Inscription、Lawn、Map、TheRock；排除 Winter Sign、Statue。

只有同时满足以下条件才判定复现矩阵完成：

- tmux orchestrator 输出 `all phases done`；
- GPU0/1 不再存在该矩阵的 train/test 进程；
- 两个 phase 各有8个场景的完整评估产物；
- post-train seed 1305、2026、4242 均存在；
- 没有 Traceback、RuntimeError、non-finite 或缺失 checkpoint。

## 2. 统一统计合同

- 历史目标使用纯 K64 的逐指标最优结果，不使用跨 K64/K256/K1024 混合 Stage1 表。
- 每个指标独立在 iter、cross、post-S2、post-train seeds 中取最优，不要求来自同一 checkpoint。
- 必须保存 `source_<metric>`，并将两个 phase 分开统计。
- Wayspots 决策优先 Acc50、Acc25；Acc10/median 用于解释，Acc5/2/1 监控副风险。

## 3. Baseline 恢复判据

一个 phase 被视为基本恢复纯 K64 历史水平，需要同时满足：

- 8场景 Acc50 均值与历史差不超过0.5个百分点；
- 8场景 Acc25 均值与历史差不超过0.5个百分点；
- 至少6/8场景的 Acc50、Acc25 各自与历史差不超过1个百分点；
- 至少6/8场景的 MedR、MedT 未恶化超过 `max(10%, 0.05°/0.3cm)`；
- 不存在单一场景的灾难性退化被均值掩盖。

当前已完成的 Bears、Squarebench 基本恢复历史水平，作为早期正信号；最终结论仍必须等待8场景和旧 commit phase 完成。

## 4. 恢复结果分支

### A. 当前代码和旧 commit 都恢复

采用当前代码的结果作为可信强 single baseline，进入 v2 双旧版重训。

### B. 旧 commit 恢复、当前代码未恢复

暂停 PMRF 训练。审计 current vs old 的 single 路径、训练默认值、eval 重建和随机性；修复后只重跑 Bears/Squarebench 确认，再进入 v2。

### C. 当前代码恢复、旧 commit 未恢复

当前代码可作为运行 baseline；先核查旧报告命令、依赖、数据和 artifact provenance。若当前8场景满足恢复标准，不因旧 worktree 单独失败阻塞 v2。

### D. 两者都未恢复

不修改 fusion。核查 memory/encoder/dataset checksum、历史命令、PyTorch/CUDA/DSAC*、训练随机性与报告统计来源；只做 Bears/Squarebench 定向验证，不立即重跑全矩阵。

## 5. v2 双旧版重训

v2 明确定义为两个旧结构，而不是一个模糊版本名：

- `pmrf_base`：`progressive_reread`、`post_norm=False`、alpha=0.10、gate_init=-2；
- `centered_s0`：`centered_reread`、`post_norm=False`、common_scale=0。

先在 Bears、Squarebench、Cubes、Tendrils 上重训。只使用 GPU0/1，保持恢复后的 single 的 memory、K64、it12、stage、buffer、分辨率、训练 seed、eval seed 和 hypotheses 不变。

训练前必须确认 evaluator 从 checkpoint 正确恢复 common_scale/cap，且 iteration、standalone、post-train eval 语义一致。

v2 晋级要求：

- Acc50、Acc25 四场景均值均不得比 single 低超过0.3个百分点；
- 至少一个主指标提高不低于0.2个百分点；
- 任一场景的主指标不得下降超过1个百分点；
- median 不得在多数场景系统性恶化；
- Acc5/2/1 完整报告，但不单独否决主指标明确提升的版本。

## 6. 后续结构决策

- centered_s0 最强：按主规划实现 `centered_reread_qknorm_layerscale`。
- pmrf_base 最强且 centered_s0 不通过：保留 progressive residual 语义，只加入 QK norm 与 LayerScale，不强行 centered/common 分解。
- 两个 v2 都通过：以 Acc50/Acc25 更优者为结构基底，另一个作为必要消融。
- 两个 v2 都不如强 single：暂停 PMRF-v3 长训练，先分析 attention entropy、A1/A2 divergence、update ratio 和 common ratio。

新结构先筛 Squarebench/Cubes，再晋级 Bears/Tendrils；四场景通过后才扩展到全部8个目标场景。

## 7. 固定工程约束

- 训练/评估从 `/home/xwh/project/ace_depth` 运行。
- torch/model smoke 使用 `conda run --no-capture-output -n mapanything python`。
- 长训练使用 tmux，启动后立即确认 GPU placement。
- 不回到 `post_norm=True`；不优先 geometry reread；不扫描 alpha/gate/tau。
- 不从单个 iteration 日志得出结论；最终判断必须使用统一 summary 和 source。
