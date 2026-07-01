你的收敛方向是正确的，而且现在应该把目标进一步明确成一句话：
这次 patch 的目标不是让 query graph 涨分，而是证明“query-graph 通路在严格 identity / capped residual 条件下不会破坏 strong single”。
当前 mlp_control 的退化太严重，已经不能作为未来 SC-QGR 的有效 baseline。它现在更像一个 unbounded frozen-head adapter，说明 frozen head 前 feature residual 极其危险。所以下一步加 identity_control + safe_mlp_control 是必要的，而且应该优先于任何真正 message passing。

1. 你当前判断基本正确
你列出的现象可以归纳成一个很清楚的诊断：
grouped sampler 大概率初步正确
但 residual path 不安全
G=4 P=128 uniq_img=4 insuff=0 和 edge_source=target_px 说明同图分组至少日志层面没问题。Bears/Cubes 掉到 Acc50 2–4%，同时 gate 接近 0.999、effective update norm 到 10+，说明问题不是“方法小幅无效”，而是 feature manifold 被强行改写，frozen head 无法解释新的 feature 分布。
所以你的下一步不应该是继续跑更多场景，也不应该上 sc_qgr_v1。正确动作就是：
先验证通路是否 identity-safe；
再验证 capped residual 是否不会灾难性破坏；
最后才进入真正 graph。

2. identity_control 是必须的，不是可选项
identity_control 的作用不是方法实验，而是排除通路 bug。它应该完整经过：
grouped sampler
reshape [G, P, C]
query_graph_refiner forward
flatten
head/loss
checkpoint save/load
eval refine hook
但输出严格满足：
H_out = H_in
如果 identity_control 结果不接近 single，那问题一定不在 residual 方法，而在以下某个环节：
grouped sampler 字段重排
features / target_px / pose / intrinsics 对齐
flatten 顺序
loss 输入 shape
eval refine hook
checkpoint config 恢复
这个模式的验收标准应该非常硬：同一 checkpoint、同一 eval seed、同一 hypotheses 下，结果应接近 single；如果有明显差距，暂停所有 safe_mlp / SC-QGR。

3. safe_mlp_control 的关键不是 gate，而是 effective update 的硬上限
你写的 safe_mlp 形式是对的：
delta = MLP(LN(H))
gate_eff = gate_max * sigmoid(gate_logit)
gamma_eff = gamma_max * sigmoid(raw_gamma)
update = gate_eff * gamma_eff * delta
update = norm_cap(update)
H_out = H + update
但我建议对 update_norm_cap 做一个小修正：不要只用绝对 cap，最好支持相对 cap。因为 feature scale 可能随 scene、fusion mode、checkpoint 变化。更稳的是：
cap = cap_ratio * rms(H)
或者至少同时记录：
update_norm / feature_norm
否则 0.01 在某些 feature scale 下几乎为零，在另一些 scale 下仍可能过大。第一版可以保留绝对 cap，但一定要记录相对比例。真正判断是否安全，看的是：
||H_out - H|| / ||H||
而不是 raw update norm。
我建议默认设置更保守一点：
gate_max = 0.05
layerscale_max = 0.05
update_norm_cap = 0.01 或 cap_ratio = 0.005~0.01
effective_update_l1_weight > 0
anchor_weight > 0
query_graph_lr 明显低于 base lr
如果 gate_max * layerscale_max 已经很小，再加 norm cap 会非常保守，这是好事。第一轮目标是“不破坏”，不是涨分。

4. 正则必须作用在最终 effective update 上
这一点你已经写对了，但要强调：不要正则 raw delta。
真正进入 head 的是：
H_out - H
所以正则应该是：
L_eff = ||H_out - H||_1
或者：
L_anchor = ||LN(H_out) - LN(H)||_2
我建议两个都可以支持，但默认只开一个即可，避免变量太多。第一版默认更推荐：
effective_update_l1_weight
anchor_weight 可以先置 0，只做记录或第二轮再开。否则 safe_mlp 变成 cap + L1 + anchor 三重限制，可能完全学不动。不过它作为保险参数保留是合理的。

5. loss_before / loss_after 诊断非常关键
你在 trainer 里加同 batch 的：
loss_before = loss(head(H0))
loss_after  = loss(head(H1))
这是必须的。它能回答一个关键问题：
refiner 是真的改善了当前 batch，还是只是训练过程把 feature 推离 head manifold？
建议记录四类指标：
query_graph_loss_before
query_graph_loss_after
query_graph_loss_delta = after - before
query_graph_improved_fraction
不过要注意两个实现细节。
第一，loss_before 只是诊断，最好不参与反传，避免增加额外梯度路径。可以 detach 或 no_grad 计算；loss_after 才参与训练。
第二，loss_before 和 loss_after 必须使用完全相同的 valid mask、pose、intrinsics、target_px、loss config。否则差值没有意义。
如果 short train 中出现：
loss_after 明显低于 loss_before
但 eval 崩
说明 safe_mlp 仍然在 sampled distribution 上过拟合。如果训练和 eval 都不崩，再进入下一步。

6. freeze_base 要同时控制 requires_grad 和 train/eval mode
你写的 freeze_base 方向是对的，但需要更严格：
frozen base: requires_grad=False + eval()
query_graph_refiner: requires_grad=True + train()
optimizer: only query_graph_refiner params
这三件事缺一不可。尤其是 frozen base 的 eval() 很重要。否则 frozen fusion/backbone 中的 dropout 或 normalization 行为仍然可能随机变化，safe_mlp 会学到一个不稳定目标。
日志建议打印：
query_graph_freeze_base=True
frozen_modules_eval_mode=True
optimizer_param_groups=[query_graph_refiner]
trainable_param_count
frozen_param_count
如果 optimizer 里还残留 base 参数，即使 requires_grad=False 通常不会更新，也会让实验审计很混乱。

7. mlp_control 应该保留，但标注为 unbounded baseline
我同意保留当前 mlp_control，但暂停作为有效 baseline。它现在的意义是负面对照：
unbounded feature residual 会灾难性破坏 frozen head
未来如果写实验，它可以作为 appendix 里的 failure case，但不应该拿来和 SC-QGR 比。真正的容量 baseline 应该是：
safe_mlp_control
否则 SC-QGR 只要“不崩”就能赢过当前 mlp_control，这没有研究价值。

8. 最小验证矩阵是对的，但通过标准还可以更硬
你列的四步非常合理：
1. none
2. identity_control
3. safe_mlp_control iter0 eval
4. safe_mlp_control short train
我建议把通过标准具体化为：
identity_control:
  Acc / MedR / MedT 接近 single
  query_graph_update_norm = 0
  eval summary 标记 identity path enabled

safe_mlp iter0:
  Acc / MedR / MedT 接近 single
  update_norm_ratio 接近 0 或严格小于 cap
  gate_eff <= gate_max
  gamma_eff <= layerscale_max

safe_mlp short train:
  Acc50/Acc25 不灾难性退化
  MedT 不出现数量级恶化
  gate_eff 不饱和到 gate_max
  update_norm_ratio 稳定小于 cap
  loss_after 不靠巨大 update 低于 loss_before
这里“接近 single”不必要求完全相同，因为 eval 可能存在微小随机性，但不能出现从 95% Acc25 掉到个位数这种情况。若 identity 或 iter0 出现明显掉点，直接查通路，不继续训练。

9. 暂停事项完全正确
这几件事现在都应该暂停：
暂停当前 unbounded mlp_control 作为主对照
不上 sc_qgr_v1
不跑 Squarebench/Tendrils
不碰 ace_fusion.py 的 PMRF/CCF/reread
不做 direct coordinate delta
不做 memory graph
尤其不要碰 ace_fusion.py。当前问题已经在 query-side refiner 通路，不应该把 PMRF/CCF 主线再混进来。

10. 我建议最终把这次 patch 定义为 “identity-safe query refiner infrastructure”
这次修改的研究意义不是模型本身，而是建立一个可靠前置条件：
后续所有 SC-QGR 实验都必须建立在 identity-safe 和 capped-residual-safe 的通路上。
所以这次 patch 的目标可以明确写成：
Add identity-safe query graph refinement infrastructure.

Modes:
  none
  identity_control
  mlp_control
  safe_mlp_control

Purpose:
  verify grouped sampler, feature-refiner insertion point,
  frozen-base training, checkpoint/eval restore, and capped residual safety
  before implementing actual graph message passing.
这就是当前最稳、最可解释的下一步。

最终判断
你的计划是正确的，而且比继续调 GNN 结构重要得多。当前结果已经说明：只要 head 前 feature residual 不受硬约束，就会把 frozen ACE head 的输入分布打崩。 所以下一步必须先建立：
identity_control 证明路径无 bug；
safe_mlp_control 证明受限 residual 不破坏 single；
然后才有资格做 sc_qgr_v1。
真正 SC-QGR 的有效对照也必须是 safe_mlp_control，不是当前失控的 mlp_control。这点要从现在就锁死。
