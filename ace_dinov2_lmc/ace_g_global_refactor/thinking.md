1. Problem Restatement
你的当前研究问题可以精确表述为：
在 ACE-FCN-LMC 的 scene coordinate regression 框架中，如何在不破坏 strong single first-read feature contract 的前提下，引入一个真正有价值的 second memory read，使其成为可解释、可复现、可与 query-only adapter 区分开的结构创新，而不是一个偶然改变 feature distribution 的额外残差模块？
基于你现在的实验总结，最关键的事实是：早期 PMRF 的提升部分来自弱 single 对照；在更强 single 参考线下，PMRF-base 不是全局稳定优于 single，而 PMRF-v3：centered reread + second-read QK norm + patch/common LayerScale 是目前 reread 类结构里最值得保留的版本。它在 Bears 上相对 strong single 有较干净提升，但在 Squarebench 上不是全面赢，因此还不能直接定为最终方法。

2. Relevant Literature Landscape
2.1 传统 modular visual localization：retrieval + matching + PnP
NetVLAD 代表 image retrieval / place recognition 路线，把局部 CNN feature 聚合为 compact image descriptor，用于大规模场景检索；SuperGlue 代表 learned local matching 路线，用图神经网络和 attention 在两组 keypoints 间做上下文匹配和 outlier rejection。这类方法的核心假设是：query 可以通过显式图像检索和局部匹配找到足够可靠的 2D-2D / 2D-3D correspondences，再交给 PnP/RANSAC 求位姿。 它们的优势是几何解释性强、跨场景泛化好；弱点是依赖 database / sparse features / matching quality，在低纹理、重复纹理、视角变化大时仍可能失败，且不是 dense SCR 范式。(arXiv)
这类工作对你有两个启发：第一，first-read 的强稳定性很重要，不能轻易破坏主匹配/主路由；第二，任何 refinement 都应该能证明自己改善了 correspondence / routing，而不是只改变了 feature norm。

2.2 SCR / learning-based relocalization：DSAC*, ACE, ACE-G, SACReg
DSAC / DSAC* 代表 dense scene coordinate regression + differentiable / robust pose solver 路线，网络先预测每个像素的 scene coordinate，再通过 RANSAC/DSAC 或 PnP 得到 camera pose。它的强点是 dense correspondence 直接服务于位姿；弱点是 coordinate regressor 容易和场景、训练视角、feature distribution 强耦合。(arXiv)
ACE 把 relocalization network 拆成 scene-agnostic backbone 和 scene-specific MLP head，并通过 reprojection loss curriculum 实现快速 scene-specific training。这直接解释了你现在为什么不能随意改 first-read feature distribution：ACE/SCR 的 head 对输入 feature contract 很敏感，哪怕一个 LayerNorm 或 attention norm 改动，也可能让 head 看到不同分布。(arXiv)
ACE-G 和 SACReg 都在尝试解决“把场景全塞进网络权重”这个问题。ACE-G 把 coordinate regressor 和 scene-specific map code 分离，并通过大规模场景预训练提升 mapping-to-query generalization；SACReg 则输入 database image 和稀疏 2D-3D annotation，通过 query/database cross-attention 预测 dense scene coordinates。它们都说明：重定位的关键趋势不是单纯加深网络，而是让 query 与外部 scene/map representation 发生更有效的受控交互。 (arXiv)
你现在的 LMC/PMRF 与这条线的关系是：你不是用 full database image，也不是用 ACE-G 的大规模 map code transformer，而是在 compressed scene memory tokens 上做轻量 query-to-memory interaction。因此你的创新必须落在“compact memory 的受控 reread”上，而不是泛泛声称 cross-attention。

2.3 Iterative attention / residual scaling：Perceiver、SwinV2、LayerScale/CaiT
Perceiver 使用 iterative asymmetric attention，把高维输入反复蒸馏进 latent bottleneck；这说明“重复读取 memory/input”本身不是新算子。你的 PMRF 如果只是 H1 -> A2 -> H2，很容易被看成一个普通 iterative attention block。(arXiv)
Swin Transformer V2 使用 scaled cosine attention，即 Q/K normalization + learnable scale，用于改善大模型训练稳定性；CaiT / LayerScale 则使用 per-channel residual scaling 来稳定更深的 vision transformer。它们说明 QK norm 和 LayerScale 都是已有稳定化技术，不能单独作为创新点。(arXiv)
这对你当前方案的含义是：PMRF-v3 的新意不在于“用了 QK norm”或“用了 LayerScale”，而在于：
它把这些稳定化机制限制在 second-read adapter 中，不改 strong first-read；同时用 centered delta 保证 second read 不改变 routing 时 residual 为零。
这个组合才是你可以辩护的结构贡献。

3. Critical Comparison
3.1 为什么不要再直接改 single first read
single_qknorm_layerscale 的结果已经说明：直接把 QK norm / LayerScale 放进 first-read attention，会改变原 single routing，使 attention 更尖锐，但不等价于 pose 更准。实验上它在 Cubes 的 Acc10/Acc5/MedT 明显下降，在 Tendrils 也没有改善，因此不适合作为主线。
从方法层面看，这与 SCR 的特性一致：ACE/SCR 的 first-read feature 和 coordinate head 已经形成一个 task-specific contract。直接修改 first-read attention，相当于重新定义 head 的输入分布；这不像 classification transformer 里加 norm/scale 那么安全。

3.2 为什么 PMRF-base 不能直接作为最终结构
PMRF-base 的优点是概念简单：
H1 = SingleRead(Q, M)
Delta2 = SecondRead(H1, M)
H2 = H1 + scale * Delta2
但它的问题是：即使 A2 没有产生有意义的新 routing，out_proj(A2V) 仍可能携带 common component 或 projection bias，从而整体移动 feature。你们后续发现 reread residual 中存在很强 patch-common 分量，这解释了为什么普通 PMRF 有时提升 median 或宽阈值，但严格指标不稳定。
所以 PMRF-base 应该保留为 ablation，而不是主结构。

3.3 为什么 centered reread 是必要条件，但还不充分
Centered reread 的核心性质是：
Delta = out_proj(A2V) - out_proj(stopgrad(A1)V)

if A2 == A1:
    Delta = 0
这比普通 PMRF 更 identity-preserving，因为它只注入 second-read 相对 first-read 的差异。这个设计非常符合“不要破坏 first-read contract”的目标。
但你们的 centered/common-scale 实验证明：单纯 centered 还不够。centered_c0 有局部收益，但复现不稳；common_scale=0.5 基本失败；common_scale=0 能减少 common pollution，但不能稳定解决所有场景。
因此 centered 是必要条件，但需要更细的 residual 控制。

3.4 为什么 PMRF-v3 是当前最合理版本
PMRF-v3 同时满足三个要求：
1. 不动 strong single first read；
2. second-read 使用 QK norm 控制 attention sharpness；
3. centered delta + patch/common LayerScale 保证弱、可控、identity-preserving 更新。
它在 Bears 上相对 strong single 有较清晰的 Acc25 / Acc10 / median 改善，在 Squarebench 上虽不是全面赢，但没有像其他版本那样明显崩掉。你的实验总结已经把它定位为“目前所有 reread 类结构里最值得保留的版本”。
关键是：PMRF-v3 的创新叙事不应是“多读一次 memory”，而应是：
在已接受的 single memory fusion 之后，追加一个 centered, layer-scaled, cosine-attention second-read adapter，使 second read 只表达相对 first read 的 routing correction。

4. Identified Research Gap
已有工作已经较好解决了几件事：
1. retrieval + matching + PnP 的模块化视觉定位；
2. dense scene coordinate regression + robust pose solver；
3. iterative attention / QK norm / residual scaling 等通用 transformer 稳定化技术。
但还没有被充分解决的是：
在 scene-specific SCR 中，如何在不破坏已有 coordinate-head feature contract 的前提下，让 compact scene memory 被第二次、受控地重新读取，并证明这种 reread 的收益来自 memory routing correction，而不是 query-only adapter 或 feature distribution shift。
你的当前实验正好把这个 gap 暴露出来：
PMRF-base:
说明 second read 在部分场景有效，但不稳定。

centered/common-scale:
说明 common residual 和 projection bias 会污染 feature，但手动 common-scale 不够稳。

single_qknorm_layerscale:
说明直接改 first read 会破坏 strong baseline。

PMRF-v3:
第一次在 strong single 下出现较干净正信号，但还需要跨场景复验和 adapter-matched control。
因此下一步不能再发散到 geometry bias、teacher guard、STGS 或多层 reread。当前最合理的研究问题应该收敛为：
PMRF-v3 是否能稳定地作为 identity-preserving memory-routing correction module，并且是否显著优于同参数、同 LayerScale 的 query-only adapter？

5. Concrete Recommendation
5.1 最合理的下一步设计方案
我建议将下一步主设计固定为：
Centered Layer-Scaled Progressive Memory Re-reading
简称可以是：
CL-PMRF
或者保守一点：
PMRF-v3
结构固定如下：
Input:
Q: query local feature
Z, P: compressed memory feature and latent 3D point
K, V: memory key/value from accepted single contract

First read:
H1, A1 = SingleFusion(Q, K, V)
# exactly same as strong single

Second read:
Q2 = Wq2(LN(H1))
Q2 = normalize(Q2)
K2 = normalize(K)
A2 = softmax((Q2 K2^T) * tau_head)

Centered context:
C1_ref = stopgrad(A1) @ V
C2 = A2 @ V
Delta = out_proj(C2) - out_proj(C1_ref)

Patch/common decomposition:
Delta_common = mean_patch(Delta)
Delta_patch = Delta - Delta_common

LayerScale:
Delta_v3 =
    gamma_patch  * Delta_patch
  + gamma_common * Delta_common

Output:
H2 = H1 + Delta_v3
X2 = CoordinateHead(H2)
默认建议：
gamma_patch init = 0.01
gamma_common init = 0.0
tau_head learnable
no post-norm
no geometry bias
no teacher guard
no routing loss
核心原则：
first read is sacred;
second read is only a weak centered correction;
A2=A1 implies no update;
patch-wise residual is allowed first;
image-common residual must be learned cautiously from zero.
这一步不是简单堆模块。它针对的是 SCR 里非常具体的问题：head 已经适配 strong first-read feature，因此 refinement 必须在 identity-preserving residual space 内进行。

5.2 必须做的实验 1：完整代表场景复验
第一组实验只比较：
E0: strong single
E1: PMRF-base
E2: PMRF-v3 / CL-PMRF
场景建议：
Wayspots:
Bears
Cubes
Squarebench
Tendrils
The Rock 或另一个代表性 outdoor scene

Indoor:
Indoor6 scene2a
Indoor6 scene5 或 scene3
为什么要加 Indoor6：Indoor6 里你们曾经观察到 true-global memory fusion 有正信号；Wayspots 更能测 outdoor / ambiguous / hard scenes。两个域都跑，才能判断 CL-PMRF 是场景特例还是结构有效。
评价指标：
Acc50 / Acc25 / Acc10 / Acc5 / Acc2 / Acc1
median rotation / median translation
DSAC inlier count
pose solver failure ratio
per-frame delta
必要口径：
same seed
same hypotheses
same post-train summary
same Stage1 protocol
single and PMRF-v3 同轮复现
先只做 Stage1。PMRF 是 local fusion 结构，Stage2 GLACE global head 会混入 head adaptation，不适合作为第一判断。

5.3 必须做的实验 2：PMRF-v3 组件消融
为了证明不是偶然调参，做最小组件消融：
A0: strong single

A1: PMRF-base
Delta = out_proj(A2V)

A2: centered-only
Delta = out_proj(A2V) - out_proj(stopgrad(A1)V)

A3: centered + QK norm

A4: centered + LayerScale patch/common

A5: full PMRF-v3
centered + QK norm + LayerScale patch/common
如果资源有限，可以只跑：
A0, A1, A2, A5
但最终论文需要至少证明：
centered 有必要；
QK norm 放在 second read 有帮助；
LayerScale patch/common 控制是稳定性的关键。

5.4 必须做的实验 3：V3-matched query-only adapter control
这是决定 PMRF-v3 能否成为创新点的关键实验。
设计一个 control：
V3-Adapter-Control:
H2 = H1 + AdapterResidual(LN(H1))
但它必须匹配 PMRF-v3 的结构属性：
1. 不读 memory；
2. 参数量尽量匹配 Wq2 + out_proj；
3. 同样 patch/common decomposition；
4. 同样 gamma_patch init = 0.01；
5. 同样 gamma_common init = 0.0；
6. no post-norm。
判断标准：
PMRF-v3 > V3-Adapter-Control:
可以说收益来自 memory rereading / routing correction。

PMRF-v3 ≈ V3-Adapter-Control:
说明主要是 LayerScale adapter 或 residual capacity，不足以作为主创新。

PMRF-v3 < V3-Adapter-Control:
停止 PMRF 主线，转向 query-side refinement。
这一步非常重要，因为 Perceiver、SwinV2、LayerScale 等已有通用模块会让审稿人自然质疑：你的收益是不是只是“多了一个稳定残差 adapter”？这个 control 是最直接的反证。(arXiv)

5.5 必须做的诊断
PMRF-v3 的诊断不要只看 attention entropy。建议记录：
Attention routing:
entropy(A1), entropy(A2)
max(A1), max(A2)
JS(A1, A2)
top1 agreement(A1, A2)
top-k overlap(A1, A2)

3D memory behavior:
mu1 = Σ A1_j P_j
mu2 = Σ A2_j P_j
spread1 = Σ A1_j ||P_j - mu1||²
spread2 = Σ A2_j ||P_j - mu2||²
||mu2 - mu1||
||mu2 - predicted X2||

Residual behavior:
||Delta_patch||
||Delta_common||
||gamma_patch * Delta_patch||
||gamma_common * Delta_common||
gamma_patch distribution
gamma_common distribution
cos(H1, H2)
||H2-H1|| / ||H1||

Pose correlation:
per-frame metric delta
per-frame residual norm
per-frame JS(A1,A2)
per-frame 3D centroid shift
最有说服力的证据不是“A2 更尖”，而是：
1. PMRF-v3 改善的 frame 里，A2 相对 A1 有可解释的 memory centroid shift；
2. residual norm 不大，但与 pose improvement 正相关；
3. gamma_common 保持较小，说明模型主要使用 patch-wise correction；
4. PMRF-v3 明显优于 V3-matched adapter。

5.6 暂时不要做的方向
短期不要做：
1. geometry bias 重新进 attention logits；
2. teacher guard；
3. STGS 联合；
4. 多层 reread；
5. hard top-k routing；
6. 改 first-read single fusion。
理由：
geometry bias:
当前已有尝试伤严格阈值，不如 PMRF-v3 稳定。

teacher guard:
会引入 frozen teacher / student 训练合同，变量太多。

STGS:
监督创新和结构创新同时改，无法归因。

多层 reread:
在单层 reread 还未跨场景成立前没有必要。

first-read 改造:
single_qknorm_layerscale 已经说明风险大。

5.7 通过 / 停止标准
PMRF-v3 晋级为主结构的条件：
1. 至少在 2–3 个代表场景上优于 strong single；
2. Acc5 或 Acc2 至少一个稳定提升，median 不退化；
3. 不出现某一类场景显著崩溃，例如 Tendrils 大幅下降；
4. 优于 V3-matched query-only adapter；
5. gamma_patch 学到非零有效更新，gamma_common 不异常放大；
6. A1/A2 routing difference 与 pose improvement 有正相关证据。
停止条件：
1. 只在 Bears 单场景有效；
2. 与 V3-adapter-control 持平；
3. 主要提升来自 Acc25/Acc10，严格指标系统性下降；
4. gamma_common 变大并主导 residual；
5. A2 与 A1 几乎相同，或 routing difference 与 pose 无关。

5.8 最终论文叙事建议
不要把方法写成：
We introduce iterative memory rereading.
这个太容易被 Perceiver / cross-attention decoder / SACReg 相关工作覆盖。
建议写成：
We introduce a centered, layer-scaled memory re-reading adapter for scene coordinate regression. Unlike generic iterative attention, our adapter preserves the accepted first-read SCR feature contract and injects only the difference between a second memory read and the detached first-read context. This makes the update identity-preserving when memory routing does not change. Patch-wise and image-common residual components are controlled separately through per-channel LayerScale, enabling weak, localized memory correction without destabilizing the coordinate head.
中文概括：
我们不是简单重复 cross-attention，而是在强 single SCR fusion 后追加一个 centered、LayerScale 控制的弱重读 adapter。它只注入相对 first-read memory context 的差异，并将 patch-wise 与 image-common residual 分开缩放，从而在不破坏 coordinate-head feature contract 的前提下实现受控 memory rerouting。

最终建议一句话
下一步最合理的设计方案是固定 PMRF-v3 / CL-PMRF 为唯一主线，先做跨场景复验、组件消融和 V3-matched query-only adapter control；只有它在 strong single 下稳定成立，才继续考虑 geometry-informed gate 或 STGS。

## 6. 训练效率压缩实验规划：少 iter + 大 buffer

背景：
当前标准 Stage1 训练使用：
1. lmc_iterations = 12
2. training_buffer_size = 2.8M
3. buffer_size_final = 7.6M
4. epochs = 24
5. samples_per_image = 512
6. post-train hypotheses = 256

从已有 single / PMRF 训练曲线看，多数代表场景在中间 iter 已基本收敛，最后一轮大 buffer 经常带来明显提升。因此需要验证一个核心假设：
12 个 outer iteration 可能不是主要收益来源，真正有效的是最后的大覆盖 buffer 和足够的 head polish。

目标：
在不引入 fusion 新变量的情况下，先用 single 模式验证训练策略能否压缩。若少 iter 策略接近 12-iter strong single，则后续所有 fusion 结构探索优先用少 iter 作为 quick filter，只把有希望的版本上 12-iter 完整复核。

### 6.1 当前单卡效率矩阵

场景：
wayspots_bears

原因：
Bears 对 Acc25 / Acc10 / median 较敏感，且已有 strong single 与 PMRF-v3 结果，适合快速判断训练策略是否损伤主指标。

当前启动矩阵：

1. single_it01_buf7p6M
   - lmc_iterations = 1
   - training_buffer_size = 7.6M
   - buffer_size_final = 7.6M
   - s1_last_iter_use_final_buffer = True
   - epochs = 24

2. single_it01_buf10M
   - lmc_iterations = 1
   - training_buffer_size = 10M
   - buffer_size_final = 10M
   - s1_last_iter_use_final_buffer = True
   - epochs = 24

3. single_it02_buf5M_final7p6M
   - lmc_iterations = 2
   - training_buffer_size = 5M
   - buffer_size_final = 7.6M
   - s1_last_iter_use_final_buffer = True
   - epochs = 24

当前脚本：
`/home/xwh/project/ace_depth/ace_dinov2_lmc/scripts/launch_single_gpu_train_efficiency_matrix.sh`

当前运行：
tmux session = `train_eff_single_gpu_20260623_gpu2`

当前结果根目录：
`/data/xwh/ace_dinov2_lmc/04_evaluation/train_efficiency_single_gpu_20260623_gpu2`

注意：
本实验使用 GPU2 是因为启动时 GPU0/GPU1 仍在跑 single_qknorm 的 squarebench / tendrils。该矩阵本身是单卡串行设计，后续可改用 GPU0 或 GPU1。

### 6.2 判断标准

Wayspots 当前主要比较：
1. Acc50
2. Acc25

辅助指标：
1. Acc10
2. median_rotation_deg
3. median_translation_cm
4. Acc5 / Acc2 / Acc1 作为副风险

Bears strong single 参考：
1. Acc50 = 98.100
2. Acc25 = 96.380
3. Acc10 = 90.340
4. MedR = 0.973
5. MedT = 3.030 cm

快速通过线：
1. Acc50 >= 97.8
2. Acc25 >= 96.0
3. Acc10 >= 90.0
4. MedT <= 3.10 cm

若高效版本 Acc50 / Acc25 与 12-iter strong single 差距小于约 0.3–0.5 pct，且 Acc10 / median 不明显退化，则可认为训练压缩有效。

### 6.3 结果解释预案

如果 single_it01_buf7p6M 已接近 strong single：
说明一次大 buffer 基本足够，12 iter 主要是冗余。后续 quick filter 可以直接用 1iter 大 buffer。

如果 single_it01_buf10M 明显优于 7.6M：
说明 buffer 覆盖仍是瓶颈，可以考虑把 quick setting 设为 1iter 10M，但需要评估时间/收益比。

如果 single_it02_buf5M_final7p6M 明显优于两个 1iter：
说明 compressor/fusion 与 head 的交替修正仍然必要，但 2iter 已可能替代 12iter。后续 fusion quick setting 优先使用：
lmc_iterations = 2
training_buffer_size = 5M
buffer_size_final = 7.6M
s1_last_iter_use_final_buffer = True
epochs = 24

如果三者均明显低于 strong single：
说明 12iter 的多轮交替确实重要，少 iter 只能作为 smoke，不能作为结构筛选依据。

### 6.4 后续扩展

若 Bears 上 2iter 或 1iter 成立，需要在至少两个补充场景验证：
1. wayspots_squarebench
2. wayspots_cubes

若这三个代表场景都能接近 12iter strong single，则后续 PMRF-v3 / adapter-control / component ablation 的默认 quick protocol 改为少 iter 策略。

完整确认版本仍需保留 12iter：
1. 最终主结果
2. paper 表格
3. 与历史结果直接对齐的复现

### 6.5 当前工程结论

训练效率优化本身应该先独立于 fusion 结构完成。不要在训练策略还未稳定时同时比较 PMRF-v3，否则无法判断收益来自结构还是来自训练覆盖变化。

当前最务实流程：
1. single 上筛出最快可靠训练策略；
2. 用该策略复测 PMRF-v3 与 strong single；
3. 再做 PMRF-v3 组件消融和 V3-matched query-only adapter control；
4. 最后把晋级版本上 12iter 完整复核。

