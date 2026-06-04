# 标题：面向视觉重定位中 Latent Memory Compression 的审稿人友好实验矩阵

## 1. 问题定义

最终论文应支持一个精确主张，而不是过度泛化的主张。将一个重定位方法拆解为 backbone 或 local predictor \(B\)、scene memory/map representation \(M\)，以及 fusion/conditioning operator \(F\)。LMC 家族学习紧凑 latent memory \(Z = C(M)\)，并把它注入预测器：

\[
\hat{X}, \hat{T} = RANSAC\left(F(B(I_q), Z, g_q)\right)
\]

对于 scene-coordinate-regression 变体；或者：

\[
\hat{T} = P(B(I_q), Z, K_q)
\]

对于 APR/map-relative pose-regression 变体。实验目标是证明三个主张：

1. LMC 在困难 mapping/query shift 下能提升兼容的强 scene-coordinate backbone。
2. LMC 不绑定单一 backbone：它能在 Indoor6 上提升 DINO-LMC，也能在 Wayspots 上提升 GLACE/ACE-FCN-LMC。
3. LMC 是一个通用的 scene-memory conditioning 思路，至少需要一个轻量 APR/map-relative 接入实验展示正迁移。

论文应明确区分主要主张和边界结果。如果 DINOACE 在某个数据集本身很弱，论文应把 DINO+LMC 报告为 backbone-limited improvement，而不是宣称它在所有场景都是 universal SOTA。

## 2. 文献背景与动机

ACE 和 DSAC* 证明了 scene coordinate regression 可以准确且高效，但 per-scene regressor 容易过拟合 mapping images，在 query distribution shift 下失效。ACE-G 通过把 scene-specific map codes 与 scene-agnostic coordinate regressor 分离，并用 mapping/query split 进行预训练来解决这一点。它评估 Indoor-6、RIO10 和 Cambridge，并且其讨论表明 DINO-style features 在室内可能很强，但在部分 outdoor/OOD 条件下不够稳定。

GLACE 加入 global-local accelerated coordinate encoding，是 Wayspots 这类场景中很自然的强 SCR baseline。当前项目已经显示 GLACE+LMC 在 Wayspots 上有广泛提升，这是证明 LMC 可以提升一个竞争力很强的非 DINO local/global SCR backbone 的最强证据。

典型 APR 直接从图像特征回归相机位姿，不显式预测 dense scene coordinates，也不使用 PnP。APR+LMC 的通用性证明应优先采用这种直接 APR。marepo / map-relative pose regression 更接近 map-conditioned learned solver/RPR hybrid，可作为相关讨论，但不应作为主 APR 实验证据。

当前 gap 是：已有结果很强但分散。Indoor6 成功主要来自 DINO+LMC，Wayspots 成功主要来自 GLACE+LMC，还缺少一个清晰的论文级实验结构来解释 LMC 在何时、为何有效。

## 3. 设计空间与选定实验架构

### 3.1 数据集策略

采用分层数据集策略。

**Tier A：主要主张数据集。**
- Indoor6：主要的困难室内 mapping/query shift benchmark。保留 DINO+LMC 作为最强结果，同时补充 GLACE+LMC，回答审稿人关于“是否只在 DINO+Indoor6 上有效”的疑问。
- Wayspots：GLACE+LMC 的主要大规模 outdoor/indoor mixed scene benchmark。保留全场景覆盖，并补充 reliability/negative-transfer 消融。

**Tier B：规模/泛化数据集。**
- 主文优先 Cambridge，而不是 Naver 大型室内数据集，除非 Naver 有广泛认可的公开 benchmark protocol 和易比较 baseline。Cambridge 对你的方法未必最友好，但它标准、可比，且 ACE-G 已经使用。它适合作为诚实的 medium-scale/OOD boundary test。
- 如果 Naver 数据集公开、稳定、成本可控，可作为 supplementary 或 appendix。它能展示 scale，但如果没有 SOTA 数字，审稿说服力通常不如标准 benchmark。

**Tier C：APR 通用性证明。**
- 在 7-Scenes 或 Cambridge 上加一个小规模典型 APR-adapter 实验。目标不是击败所有 APR 方法，而是证明 compressed scene memory 可以 condition 直接 pose regressor，并且优于不带 LMC 的相同 APR backbone。

### 3.2 推荐主文表格

**Table 1：Indoor6 SOTA comparison。**
Rows：ACE、DINOACE、ACE-G、GLACE、DINO+LMC、GLACE+LMC、DINO+LMC+best memory extraction。Metrics：median cm/deg、Acc5、Acc10、mapping/training time、memory size。这是最强表。

**Table 2：Wayspots full-scene comparison。**
Rows：DSAC*、ACE、GLACE、可用时 DINOACE、GLACE+LMC、带 reliability control 的 GLACE+LMC。Scenes：全部 Wayspots scenes 和平均。Metrics：10cm/5deg 与 5cm/5deg，或 GLACE/Wayspots 相关 prior 使用的相同阈值。这是第二强表。

**Table 3：Cross-dataset summary。**
Datasets：Indoor6、Wayspots、Cambridge、如果容易则加入 RIO10。Columns：base backbone、base score、+LMC score、relative gain、SOTA gap。该表应明确展示：即使 base 不是 SOTA，LMC 也稳定提升自己的 base。

**Table 4：Resource and scalability。**
Rows：ACE/GLACE/DINOACE/LMC variants，以及 ACE-G reported/reproduced map-code setting。Columns：memory file size、latent tokens/map codes、token dimension、training time、inference FPS、GPU memory、map build/extraction time。审稿人会问 memory compression 是否值得，这张表直接回答；同时要明确当前 LMC 默认 K=64 的容量远小于 ACE-G 常见 1024/4096 map embeddings，因此必须报告 compactness vs accuracy 的 trade-off。

**Table 5：APR generality。**
Rows：PoseNet/MapNet-style APR baseline、APR + zero/random memory descriptor、APR + raw memory descriptor、APR + compressed LMC descriptor、可选 APR + LMC cross-attention。用少量场景。Metrics：median cm/deg、Acc5/Acc10、training/fine-tuning time。把它作为通用性表，而不是主 headline。

### 3.3 必要消融

**Ablation A：memory source and extraction。**
比较 no memory、random memory、DINO/ACE-FCN feature memory、BSE pooled memory、voxel/simple pooled memory、不同 view selection strategies，以及适用时 GT-vs-predicted memory。

**Ablation B：compression capacity。**
这是必须升级为论文级核心消融，而不是只扫小 K。当前默认 LMC 使用 K=64；若 feature dim 为 512/1024，则只有约 32K/65K 个 latent scalars。ACE-G 常见设置有 1024 或 4096 个 map embeddings，按 512 维计算约 0.5M/2.1M scalars，容量高出一个数量级以上。因此主消融应比较：

- compact LMC：K = 32、64、128、256、512；
- extended LMC：K = 1024，仅在 1-2 个代表场景上跑；
- ACE-G-scale reference：K = 4096，仅作为单场景或 supplementary reference，除非收益明显且算力可接受。

报告 accuracy vs memory size/training time/VRAM，并把结论写成“在远小于 ACE-G map-code 容量时是否仍能达到或超过强 baseline”，而不是只报告 K 越大越好。

**Ablation C：fusion location。**
Backbone-only、decoder/local-only、global concat、local+global、residual global、adaptive/reliability-gated global。这直接回应 SquareBench negative transfer。

**Ablation D：training stage contribution。**
Stage1 only、Stage2 only、S1+S2、no warmup、no query-style split、no final buffer refill、frozen vs trainable fusion/head。证明 two-stage training 是必要的。

**Ablation E：robustness。**
Mapping/query split difficulty、memory views 数量、view selection quality、appearance shift、可用时 pose noise。Indoor6 最适合做这类消融，因为主结果很强。

**Ablation F：failure analysis。**
报告 DINO+LMC 能提升 DINOACE 但仍低于 SOTA 的场景。加入 qualitative coordinate maps 或 error histograms。解释边界能增加论文可信度。

## 4. 评估与验证计划

### 4.1 立即优先级

1. **Indoor6 GLACE+LMC。** 这是必须做的。既然 DINO+LMC 在 Indoor6 很强、GLACE+LMC 在 Wayspots 很强，缺失的交叉验证就是 GLACE+LMC on Indoor6。若结果好，它强力支持 LMC 的 backbone-agnostic claim；若结果不好，也能定义兼容性边界。

2. **Scene-token capacity ablation。** 立刻补 Indoor6 scene3/scene4a 的 K sweep：64、128、256、512，必要时加 1024。这个实验直接回答“当前 LMC 是否只是因为容量太小而限制了效果”，也给 Resource/Scalability 表提供关键证据。若 K=256/512 明显优于 64，应把最优 K 用于 all-scene GLACE+LMC；若 K 继续增大收益很小，则突出 compact memory 的优势。

3. **完整 Wayspots GLACE+LMC with reliability controls。** Wayspots 已有提升，但论文需要全场景 average、threshold tables、per-scene deltas，以及 SquareBench/Bears negative-transfer control。使用 `stage2_global_reliability` 作为设计依赖。

4. **Cambridge as medium-scale/OOD benchmark。** 跑 DINOACE、DINO+LMC、可行时跑 GLACE、GLACE+LMC。预期不一定是 SOTA；目的在于展示 LMC 是否提升 base，并对齐 ACE-G 的 Cambridge protocol。

5. **APR adapter proof。** 只应在主要 SCR 表稳定后实现。选择最小接入，比较 APR baseline vs APR+LMC，在 7-Scenes 或 Wayspots 上做。这应放 supplementary，除非结果很强。

6. **Naver large indoor dataset。** 只有在数据公开、split 清楚、描述干净时使用。否则它是 supplement/demo，不是主文支柱。

### 4.2 每个数据集具体跑什么

**Indoor6。**
- DINOACE baseline。
- DINO+LMC best。
- GLACE baseline。
- GLACE+LMC。
- 可行时加入 ACE-G reported/reproduced。
- 在 scene3、scene4a 上做 K=64/128/256/512 容量消融，必要时单场景加 K=1024/4096；view-count 消融只做代表场景，不必全场景 exhaustive。
- 对 2 个困难场景给 qualitative coordinate reconstruction。

**Wayspots。**
- GLACE baseline。
- GLACE+LMC full all-scene run。
- GLACE+LMC reliability-gated variant。
- Bears 和 SquareBench 上做 zero/random memory controls。
- 全 scene 表；详细消融只做 Bears、SquareBench 和一个平均/中性场景。

**Cambridge。**
- DINOACE vs DINO+LMC。
- 如果实现成本可控，GLACE vs GLACE+LMC。
- 只用一个 memory capacity 设置。不要在 Cambridge 上投入过多 exhaustive ablation。
- 诚实报告为 OOD/scale boundary。

**Naver large indoor。**
- 只有 data loading 稳定时才跑 DINOACE vs DINO+LMC 和 GLACE vs GLACE+LMC。
- 用作 scale demonstration：memory size、extraction time、FPS、qualitative maps。
- 除非它有公开 baselines，否则不要做主要 SOTA 表。

**APR adapter。**
- 先做 PoseNet-style 典型 APR baseline；如果代码/算力允许，再做 MapNet/MS-Transformer-style APR。
- +raw scene memory conditioning。
- +compressed LMC memory conditioning。
- 可选 fine-tuning only adapter/gate。
- 尽可能选一个 indoor 和一个 Wayspots-like scene。

## 5. 预期失败模式与工程风险

主要风险是 over-claiming。如果 DINO+LMC 在 Indoor6 外较弱，论文必须把它表述为 backbone-limited，并用 GLACE+LMC 作为更强的一般 SCR 证据。另一个风险是 dataset overload：Cambridge、Naver、APR、所有消融全做可能拖慢论文，而不一定增强核心主张。计划应优先 Indoor6 GLACE+LMC 和完整 Wayspots reliability。

第二个风险是 global features 带来的 negative transfer。SquareBench 已经说明 raw 或过强 global injection 有害。因此在宣称 robust GLACE+LMC 前，需要 reliability controls、zero/random global controls 和 Stage1 consistency。

第三个风险是不公平 baseline。每个主表都应区分 reproduced numbers 与 reported numbers，列出 training time 和 memory size，并尽量使用 prior papers 的相同阈值。

## 6. 复用计划

使用 `../trainer_dinov2_lmc.py`、`../options_dinov2_lmc.py` 和 `../test_ace_dinov2_lmc.py` 作为 DINO-LMC 和 GLACE/ACE-FCN-LMC 的中心实现/评估路径。使用 `../stage2_global_reliability/01_design/proposal.md` 指导 Wayspots reliability variants。使用 `../scripts/run_squarebench_stage2_global_gate_matrix.sh` 和 `../scripts/summarize_wayspots_ace_fcn_lmc_suite.py` 作为 scene-level orchestration 与 reporting 模板。使用 `../../papers/Bruns 等 - 2025 - ACE-G Improving Generalization of Scene Coordinate Regression Through Query Pre-Training.pdf` 做 Indoor6/RIO10/Cambridge 定位。APR 泛化证明应优先采用典型直接 APR（PoseNet/MapNet/MS-Transformer/TransPoseNet 风格）；marepo 更接近 map-relative learned solver/RPR hybrid，只作为相关讨论，不作为主 APR 证据。

## 7. 最终推荐实验矩阵

### 投稿前必须完成

| Block | Dataset | Methods | Purpose |
|---|---|---|---|
| Main SCR-1 | Indoor6 | DINOACE, DINO+LMC, GLACE, GLACE+LMC, ACE-G/GLACE reports | 证明最强 SOTA claim 与 Indoor6 跨 backbone 有效性。 |
| Main SCR-2 | Wayspots | GLACE, GLACE+LMC, reliability-gated GLACE+LMC, zero/random controls | 证明广泛提升并处理 negative transfer。 |
| Ablation-1 | Indoor6 scene3/scene4a | scene-token K=64/128/256/512, optional 1024/4096 ACE-G-scale reference | 证明 LMC 容量/compactness trade-off，并选择 all-scene 主实验 K。 |
| Ablation-2 | Indoor6 subset | memory source, view count, S1/S2 | 解释 DINO+LMC 为什么有效。 |
| Ablation-3 | Bears/SquareBench | global gate, residual, Stage1 consistency, zero/random global | 解释 GLACE+LMC 为什么稳健。 |
| Efficiency | Indoor6 + Wayspots | memory size, latent-token/map-code count, time, FPS, VRAM | 证明 compact memory 的实用性，并与 ACE-G map-code 容量对齐。 |

### 算力允许时强烈推荐

| Block | Dataset | Methods | Purpose |
|---|---|---|---|
| Boundary | Cambridge | DINOACE vs DINO+LMC; GLACE vs GLACE+LMC if feasible | 标准 benchmark 和诚实 OOD boundary。 |
| Generality | 7-Scenes or Wayspots subset | APR baseline vs APR+LMC | 证明 LMC 不只适用于 SCR。 |

### 可选 supplementary

| Block | Dataset | Methods | Purpose |
|---|---|---|---|
| Scale demo | Naver large indoor | base vs +LMC, resource table, qualitative maps | 若公开 protocol 弱，则展示 large-scene feasibility。 |
| RIO10 | RIO10 | DINOACE/DINO+LMC if easy | 对齐 ACE-G；算力紧张时不是必需。 |

## 8. 对用户问题的直接回答

**Indoor6 是否要补 GLACE+LMC？** 要。这是最重要的缺失实验。它闭合逻辑三角：DINO+LMC 在 Indoor6 有效，GLACE+LMC 在 Wayspots 有效，而 GLACE+LMC on Indoor6 检验 LMC 的 Indoor6 成功是否不是 DINO-specific。

**Wayspots 是否还需要补实验？** 需要，但不要无止境 sweep。补全 full-scene GLACE+LMC tables，加入 zero/random controls，并在 Bears 和 SquareBench 上做 reliability/negative-transfer 消融。不要继续只扫 scalar gate。

**Cambridge 还是 Naver large indoor？** 主文选 Cambridge，因为它标准、可比。Naver 若没有公开 baselines 和 clean splits，作为 supplementary scale evidence。二选一时选 Cambridge。

**是否把 LMC 接到 APR？** 是，但作为主要 SCR story 稳定后的轻量通用性证明。最低有效结果是 APR baseline vs APR+raw memory vs APR+compressed LMC memory，在小标准子集上比较。不要让 APR 接入拖慢核心论文。

**最终 narrative 是什么？** LMC 是一个 compact scene-memory conditioning 机制。当 base relocalizer 兼容且数据集存在强 mapping/query shift 时，它给出 SOTA-level gains；即使 base backbone 本身不是 SOTA，它也能稳定提升 base model。论文如果诚实说明这个边界，会更有说服力。