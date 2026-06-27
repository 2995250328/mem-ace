你的这次只读核对基本把 SC-QGR 第一版的实现边界 定准了。我同意你的主结论：它不应该再被理解为 ace_fusion.py 内部的一个 PMRF/CCF refinement mode，而应该是 fusion 输出之后、head 回归之前的 query-side refinement 模块。这点非常关键，因为当前 ace_fusion.py 已经覆盖了 single、PMRF、centered、geometry reread、adapter、dual、coord prior 等 memory-injection / reread 变体；PMRF-v3 本身也已经包含 centered reread、second-read QK norm、patch/common LayerScale 等完整机制。继续把 SC-QGR 塞进这个分支，会把“再次读 memory”和“同图 patch 协同”两个问题混在一起。
下面我给出一个更收敛的版本：SC-QGR-v1 的合理性分析 + 第一版设计边界 + 实验验收标准。重点不是代码怎么写，而是哪些东西必须锁死，避免第一次实验结论失真。

1. 总体研判
你这次核对后，SC-QGR 的定位应当从“fusion variant”改成：
在 strong single memory fusion 之后，对同一 query image 内的 scene-conditioned patch features 做一次轻量图 refinement，然后仍然通过原来的 MLP head 回归 scene coordinates。
也就是说第一版主链路应该是：
raw / fused buffer rows
        ↓
single LMC memory fusion
        ↓
scene-conditioned patch features
        ↓
same-image query graph refinement
        ↓
refined features
        ↓
原 regressor.get_scene_coordinates(...)
        ↓
existing reprojection / coordinate loss
这个定位比前面讨论过的 PMRF gating、latent memory graph、coordinate delta 都更稳。原因是：它引入了一个当前所有 reread 分支都没有的信息源——同一查询图像内其他 patch 的 scene-conditioned evidence。PMRF/centered 的 second read 仍然是单个 patch 再读同一个 compressed memory；v3_adapter_control 是 query-only residual；v3_dual_refine 是 memory residual 和 adapter residual 的叠加。它们都没有真正解决“同一 query image 内 patch 之间是否应该协同”这个问题。你给出的代码核对也支持这个定位：fusion 输出接入点在 _fuse_lmc_features_for_head(...) 之后，head/loss 当前已有 full-map loss 和 sampled-feature loss 两类路径，因此 SC-QGR 最自然的位置就是 fusion output 与 get_scene_coordinates(...) 之间。
所以第一版不要再和 PMRF/CCF 混合。PMRF-v3、CCF-lite 继续作为 baseline；SC-QGR-v1 只回答一个干净问题：
在 strong single 已经完成 memory conditioning 后，同图 patch graph 是否能提供额外的有效上下文？

2. 为什么 sampler 是第一风险，而不是 graph module
你这次最重要的判断是：SC-QGR 第一版成败的代码关键不是 message passing 写得多复杂，而是 img_idx + target_px grouped sampler 是否正确。
这个判断完全正确。原因很简单：GNN 的 node 和 edge 只有在“同一图像内”才有语义。如果沿用当前全局随机 row sampling：
torch.randint(0, buffer_len, (draw_bs,))
那么一个 batch 内的 patch rows 可能来自不同图像、不同视角、不同相机位姿。此时任何 KNN、graph edge、attention edge 都是在伪图上做消息传递。即使 loss 变好，也无法说明同图上下文有效；即使 loss 变差，也不能否定 graph refinement。这个实验会彻底失真。
同理，_pack_feature_rows_for_head(... grid_h=16) 只是为了把随机 rows 伪装成 1x1 ACE head 可接受的张量形状。这个 16×W 伪网格不是图像真实空间邻域，绝对不能拿来构建 spatial edge。你把这一点明确列出来非常重要，因为这是最容易被工程实现误用的地方。
因此第一版 SC-QGR 的最小真实前提是：
每个 graph 内的节点必须来自同一 img_idx；
每条 spatial edge 必须基于真实 target_px；
不能基于随机 row order 或 packed pseudo-grid。
这也解释了为什么需要强制 indexed buffer。你核对到当前 raw buffer / fused buffer schema 已经有 features、target_px、pose、intrinsics、gt_scene_coords 等字段，而 indexed schema 额外有 img_idx；但普通 raw buffer 不一定保存 img_idx，只有 image-global / GLACE / multiframe 相关路径才会启用 image indices。对于 SC-QGR，img_idx 不是附加信息，而是核心结构变量。没有它，graph sampler 就没有合法定义。

3. SC-QGR-v1 的第一版设计边界
第一版应当极端克制，只实现一个干净版本：
base = strong single
refiner = same-image sparse graph
output = feature residual
head = 原 MLP head
training = freeze base, train graph only
不支持：
PMRF + SC-QGR
CCF + SC-QGR
direct coordinate delta
memory graph
multi-round GRU
DSAC-driven learning
这个边界看起来保守，但它有一个重要好处：实验结果可归因。如果 single + SC-QGR 超过 strong single、MLP control、shuffled-edge control，那么你能比较明确地说：同图 patch graph 在 scene-conditioned feature 上提供了新信息。如果一开始就叠 PMRF、CCF、graph、coordinate state，涨分也解释不清。

4. 模块输入输出应该怎么定义
SC-QGR-v1 不应该直接预测 scene coordinate delta。它应该预测 feature residual。具体地说：
给定 strong single 后的 features：
H0: [G, P, C]
其中 G 是每个 batch 中的图像数，P 是每张图采样的 patch 数，C 是 feature dim。
还需要：
target_px: [G, P, 2]
img_idx: [G]
intrinsics / intrinsics_inv: optional for logging or future
gt_pose: existing loss uses it
第一版可以额外使用第一次坐标预测作为 detached state：
X0 = head(H0).detach()
然后构造 node feature：
node_i = [
    LN(H0_i),
    PE_2D(target_px_i),
    PE_3D(stopgrad(X0_i normalized))
]
这里 X0 只作为状态提示，不参与反向更新。理由是：第一版要验证 graph refiner 是否有用，而不是让 graph 和 head 联合重写初始坐标分布。尤其在你的系统里 DSAC 不提供梯度，训练仍主要依赖现有 reprojection / coordinate loss。如果一开始直接预测 coordinate delta，沿相机射线方向的弱约束可能带来不可控漂移。
输出应为：
delta_H = graph_refiner(node, edges)
H1 = H0 + gate * layerscale * delta_H
X1 = head(H1)
注意这里 head 虽然 frozen，但不能在 no_grad() 下跑最终 X1 = head(H1)，否则 graph 收不到梯度。正确语义是：head 参数 frozen，但计算图允许梯度从 loss 通过 head 输入回传到 H1 和 graph refiner。X0 可以 no_grad 或 detach；X1 不可以 no_grad。

5. Graph 结构第一版要简单，不要一开始动态图
第一版建议固定 same-image spatial graph。最稳的是基于 target_px 的 KNN 或局部窗口邻域：
edges_i = KNN(target_px_i, target_px_j), j in same img_idx
如果你做的是 sampled-feature training，而不是 full image grid，KNN 的邻居不一定是密集图像邻域。所以 sampler 最好不要完全随机地从同一图像里抽 P 个 patch，而应该采样局部窗口或空间分层 patch。否则图仍然是同图，但局部邻域统计会和推理时 full grid 不一致。
第一版可以有两种采样策略，建议先用更稳的局部窗口：
局部窗口采样：每张图随机选一个或多个中心，在 target_px 空间中选最近的 P 个 patch rows，构成一个局部小图。这个策略最贴近 Conv/GRU 的局部上下文思想，但仍然兼容 ACE 的 buffer 训练。
空间分层采样：把图像划成粗格，每个格子采若干点，再基于 target_px 做 KNN。这个策略覆盖更全，但边可能跨较大距离，不如局部窗口稳定。
第一版不要用 predicted coordinate 构图。原因是如果 X0 错了，动态图会把错误预测组织成自洽簇，形成 self-confirmation。X0 可以作为 node state，但不要作为 hard edge topology。
边特征可以很简单：
edge_ij = [
    PE_2D(target_px_j - target_px_i),
    cosine(H0_i, H0_j)
]
第一版甚至可以先只用相对 2D PE，不加 feature cosine，减少变量。feature cosine 作为 v1.1 消融。

6. Message passing 第一版不要复杂
第一版 graph refiner 只需要一层或两层，目标不是追求表达力，而是验证同图关系是否真的有收益。建议结构语义是：
message_ij = MLP([node_j, edge_ij])
alpha_ij = softmax_j(MLP([node_i, node_j, edge_ij]))
m_i = sum_j alpha_ij * message_ij
delta_H_i = MLP([node_i, m_i])
gate_i = sigmoid(MLP([node_i, m_i]) + gate_bias)
H1_i = H0_i + gate_i * gamma * delta_H_i
初始化必须保守：
gamma init: 很小，例如 0.01 或更低
gate bias: 负值，让初始 gate 偏小
delta output projection: 可零初始化
residual L1: 小权重
这样 SC-QGR 初始近似 strong single，不会一开始破坏已有稳定路径。这个思路和 PMRF-v3 里 patch/common LayerScale 的安全哲学一致；只是 SC-QGR 的 residual 来源不再是 second memory read，而是 same-image query graph。PMRF-v3 当前已经使用 centered residual、QK norm、patch/common LayerScale，并且 v3_adapter_control 也作为 query-only matched control 存在，因此 SC-QGR 必须更严格地证明它不是“多一个 residual MLP”。

7. 训练策略必须采用 Stage B 冻结基座
你提出 “Stage B 冻结 backbone、compressor、fusion、head，只训练 graph refiner” 是正确的。第一版不要联合训练，因为联合训练会让所有归因混掉。
推荐流程是：
Stage A:
  使用已有 strong single checkpoint

Stage B:
  freeze backbone
  freeze GeoLMC / memory compressor
  freeze LMC fusion
  freeze coordinate head
  train query_graph_refiner only
训练 loss 使用现有 loss，不引入 DSAC。因为你已经明确 DSAC 在当前工作中只是无梯度位姿解算器，它不能作为可学习闭环。SC-QGR 的训练目标应保持：
refined_features -> regressor.get_scene_coordinates(...) -> existing reprojection / coordinate loss
同时加一个小的 residual regularization：
L_total = L_existing(X1) + λ * ||gate * gamma * delta_H||_1
是否加 L_existing(X0) 作为辅助不建议第一版加入，因为 base frozen，X0 本身不会更新。可以记录 X0 loss，作为“refinement 改善/恶化多少”的诊断指标，但不需要把它放进训练目标。
第一版训练时还要记录：
loss_before_graph
loss_after_graph
fraction_improved_patches
delta_H_norm
gate_mean / gate_histogram
edge_attention_entropy
否则即使 pose 指标有变化，也很难判断 graph 是在修正局部错误，还是只是做了 feature smoothing。

8. eval 侧必须提前想清楚
训练时你用 grouped sampled patches，但测试 pose 时通常需要对整张 query feature map 输出 scene coordinates。SC-QGR 必须支持 eval full-map 或至少支持按图分块处理。
第一版建议 eval 直接在完整 query grid 上构建稀疏局部 graph：
nodes = all query patches
edges = 4/8-neighbor or KNN in target_px grid
如果显存压力大，再做 tile/block 方式，但要避免 tile 边界带来明显不一致。训练 sampler 如果使用局部窗口，eval full-grid 的分布差异会小一些；如果训练时是同图随机点，eval full-grid 的局部性会更强，可能出现 train-test gap。
因此我建议第一版 sampler 直接模拟 eval：局部窗口 + 2D KNN/邻接。不要全图随机同图采样。

9. 必须做的 controls
你列的 controls 很完整，我建议第一版最小矩阵如下：
1. strong single
2. PMRF-v3
3. CCF-lite
4. SC-QGR-v1
5. MLP-control
6. shuffled-edges
7. cross-image-edges
其中最关键的是 5、6、7。
MLP-control：同样输入 H0 + PE_2D + PE_3D(X0)，但不看邻居，只输出 feature residual。这个对照回答：涨分是否只是因为多了一个 per-node residual network。
shuffled-edges：同一图像内节点不变，但边随机打乱。这个对照回答：真实 target_px 空间关系是否重要。
cross-image-edges：节点数量、边数量、参数量保持一致，但故意把邻居换成其他图像的 patch。这个对照回答：graph 是否真的依赖同图结构。如果 cross-image 也涨，说明模块只是 regularizer，不是 query context。
如果 SC-QGR 不能同时超过这三个 control，就不能声称“同图 patch 协同有效”。最多只能说“额外 residual capacity 有帮助”。

10. 成功判据不要只看 Acc25
第一版成功不能只看宽阈值，因为已有 PMRF/common-scale 现象已经说明某些 residual 会提升 Acc25/Acc10，但伤害 Acc2/Acc1 或 median translation。你提供的事实文档里，centered 和 PMRF 在 Bears/Squarebench 上呈现明显 trade-off；Tendrils 上所有 reread/common variants 都很弱，这说明只看粗指标很容易误判。
建议第一轮成功标准是：
SC-QGR-v1 > strong single
SC-QGR-v1 > MLP-control
SC-QGR-v1 > shuffled-edges
SC-QGR-v1 > cross-image-edges
并且至少满足：
Acc5 / Acc2 不下降
MedT 不恶化
Acc25 / Acc10 不靠大幅牺牲 strict metrics 换来
场景上建议优先：
Bears
Cubes
Squarebench
Tendrils
解释如下。Bears 和 Squarebench 是 PMRF/centered 已经出现不同收益模式的场景；Cubes 是 strong single 很强、容易检验 graph 是否破坏 baseline 的场景；Tendrils 不应设为必须提升，而应作为 failure diagnostic。如果 SC-QGR 也救不了 Tendrils，但 real graph 与 shuffled/cross-image 差异明显，仍然说明方法有价值；如果所有东西都一样，Tendrils 失败就更可能是 memory/feature 表征上游问题。

11. 我对你当前计划的微调建议
你的实现顺序基本正确，我只建议补三个约束。
第一，lmc_query_graph_refine_mode != none 时不仅要强制 indexed buffer，还应该在日志和 checkpoint 中显式写出：
query_graph_requires_img_idx=True
query_graph_sampler=grouped_by_img_idx
query_graph_edge_source=target_px
否则后面复现实验时很容易混到非 grouped sampler。
第二，freeze_base=True 第一版应当是硬默认，最好训练日志里打印每类参数的 requires_grad 统计。这个模块一旦不小心让 head 或 fusion 参与更新，实验就不再是“query graph refinement”验证，而变成了新一轮联合微调。
第三，第一版不要让 CCF-lite 的 gate stats 或 PMRF 的 attention stats 进入 SC-QGR node feature。它们可以记录，但不要作为主方法输入。否则如果 SC-QGR 有收益，你无法判断收益来自同图 patch graph，还是来自 attention confidence feature。后续如果图本身成立，再加入这些信号作为 v2。

12. 最终结论
你这版核对之后，SC-QGR-v1 的边界已经非常清楚：
它不是 ace_fusion.py 的第十三个 refinement mode，而是一个 trainer/test pipeline 中的 query-side refinement stage。它接在 _fuse_lmc_features_for_head(...) 之后，接在 regressor.get_scene_coordinates(...) 之前；它必须使用 img_idx + target_px 构建真实同图小图；它第一版只服务 strong single，不和 PMRF/CCF 混合；它只输出 feature residual，不直接预测 coordinate delta；它第一轮冻结所有基座，只训练 graph refiner。
我认为你下一步最应该锁定的不是 GNN 结构，而是 grouped sampler 的正确性。如果 sampler 做对了，即使第一版 graph 很简单，实验结论也是可信的；如果 sampler 错了，再复杂的 graph 都只是在伪网格上做无意义消息传递。
