这个补充批评也是对的，而且它把方案从“方向正确”推进到了“可实现、可写文档”的层面。最关键的四点我同意：

* 必须显式写出 **Reference Frame Assumption**
* `Q_probe-consistency` 必须收敛成可实现的 probe 指标
* “存在 single reference domain” 必须变成 **top-M reference candidates + bounded probe** 的可计算过程
* cluster 分支必须明确 **query 端接口**，第一版不做 learned routing、不做 cross-cluster fusion 

下面我直接把方案收敛成一个更硬的 **v2 设计**。这个版本不再只是概念框架，而是带有明确假设、数据结构、判定规则和分支接口的技术方案。

---

# V2 方案名称

## Reference-Consistent Adaptive Memory Construction v2

它的核心不是“再优化选帧”，而是：

* 先定义 **memory 的 reference-local contract**
* 再判断 **single reference-local domain 是否可行**
* 可行则在该 domain 内做 topology-aware + reference-risk-aware + auto-stop 的 single-forward memory construction
* 不可行则进入 **clustered reference-local memories + ensemble fallback**

---

## 1. 先固定研究边界

第一版明确排除以下变量，避免系统继续膨胀：

* 不研究 BSE / voxel pooling / point density optimization
* 不更换 backbone
* 不引入 learned query routing
* 不引入 learned scene policy
* 不做 cross-cluster feature fusion
* 不把 cluster 分支默认化，只作为 fallback

这个收敛方向和 reviewer 的建议一致 

---

## 2. Reference Frame Assumption

这是 v2 的第一个硬定义。

### Assumption A

**默认假设：MapAnything 的 shared memory features 由选定的 reference view 启动，并且其几何语义近似锚定在该 reference camera frame 上。**

也就是说，第一版中我们把 MapAnything 的 feature frame 近似为：

[
\mathcal{F}*{feat} \approx \mathcal{F}*{refcam}
]

其中：

* (\mathcal{F}_{refcam}) = 选定 reference 相机的相机坐标系
* 不先假设它是 dataset world frame
* 也不先假设它是模型内部某个不可解释 latent frame

### 对应坐标变换

若 reference 相机在 world frame 下的 camera-to-world 为：

[
T_{c2w}^{ref} = [R_{ref} \mid C_{ref}]
]

则：

[
P_{ref} = R_{ref}^{T}(P_{world} - C_{ref})
]

[
P_{world} = R_{ref}P_{ref} + C_{ref}
]

这是第一版 memory contract 的数学基础。

---

## 3. 但这个假设必须被验证

reviewer 指出的反例很重要：
如果 MapAnything 的内部表征不是简单的 reference camera frame，而是某种 pose-conditioned globally aligned latent space，那么强制 `points_world -> points_ref` 未必更正确 

所以 v2 必须加一个 **reference-swap sanity check**。

### Sanity Check 设计

对同一组 selected views，固定同一 scene，切换不同 `reference_index`：

1. 以 `ref_a` 构建 memory
2. 以 `ref_b` 构建 memory
3. 比较以下量是否能通过刚体变换 (T_{ref_a \to ref_b}) 被解释：

* predicted poses
* points_ref
* pose residual distribution
* memory consistency score

### 判定

若 reference 切换后，结果大体满足刚体变换可解释，则 Assumption A 可继续使用。
若不可解释，则说明：

* feature frame 不是简单 reference camera frame
* 那么 contract 需要降级成：

  * **reference-conditioned domain id**
  * 而不是显式 3D reference coordinate frame

也就是说，v2 允许这个假设被证伪，但第一版先按它实现。

---

## 4. 固定 Memory Schema

这是第二个硬定义。
无论 single 还是 cluster，每个 memory 都必须使用同一 schema。

### Single-memory schema

```python
memory = {
    "mode": "single_forward",

    "points_world": ...,
    "points_ref": ...,
    "points_ref_norm": ...,
    "features_ref": ...,

    "reference": {
        "reference_index": ...,
        "T_ref_c2w_world": ...,
        "T_world_to_ref": ...,
    },

    "normalization_ref": {
        "mu_ref": ...,
        "sigma_ref": ...,
    },

    "selection": {
        "selected_views": ...,
        "stop_reason": ...,
        "coverage_stats": ...,
        "topology_stats": ...,
        "reference_risk_stats": ...,
        "probe_stats": ...,
    }
}
```

### Clustered schema

```python
memory_package = {
    "mode": "clustered",
    "clusters": [
        cluster_0_memory,
        cluster_1_memory,
        ...
    ],
    "policy": {
        "decision": "cluster_branch",
        "reason_codes": ...,
    }
}
```

其中每个 `cluster_i_memory` 与 single-memory schema 完全同构。

### 明确语义

* `features_ref` 绑定 `points_ref / points_ref_norm`
* `points_world` 只用于最终 world-frame relocalization / PnP
* network learning frame = `ref` 或 `ref_norm`
* PnP frame = `world`

这一步是整个系统的底座，不再是“预留扩展位”。

---

## 5. Single-Domain Feasibility Gate

这是第三个硬定义：
“是否存在一个可行的 single reference domain” 不能靠想象，必须变成可计算的 bounded search。

---

### 5.1 Top-M Reference Candidates

不穷举所有 reference，而是先选 top-M 候选 reference。
第一版建议来源：

1. **距离全场景 camera-center mean 最近**
2. **global covisibility 平均值最高**
3. **graph centrality 高**
4. **轻量 probe 下 reference-risk 最低**

实际实现时可以先取这几种规则的并集，再去重，得到 (M=4 \sim 8) 个候选。

---

### 5.2 对每个 reference candidate 跑 bounded selector probe

对每个 (r_i)：

1. 以 (r_i) 为 root/reference
2. 运行一次 bounded TA-ASB probe
3. 只允许最多选到 `max_views_probe`
4. 输出一个 candidate selected set (S_i)

这里的 probe 不是最终 memory construction，而是 feasibility test。

---

### 5.3 四个 adequacy 判据

对每个 (S_i)，计算四个布尔量：

#### A. coverage_ok

例如：

* `coverage_mean <= tau_cov_mean`
* `coverage_max <= tau_cov_max`

#### B. topology_ok

例如：

* `num_components == 1`
* `isolated_count == 0`
* `degree_min >= 2`

#### C. ref_risk_ok

例如：

* `ref_dist_q90 <= tau_ref_q90`
* `ref_dist_max <= tau_ref_max`
* `far_view_budget_used <= tau_far_budget`

#### D. probe_ok

这里 reviewer 的建议我完全接受，第一版只保留两个 probe 指标，避免抽象化 

##### Probe 指标 1

**MapAnything predicted camera pose after aligning reference to GT 的 translation residual**

##### Probe 指标 2

**selected views 中 non-reference pose residual 的 q90 / max**

于是：

```python
probe_ok = (
    trans_q90 < tau_probe_q90 and
    trans_max < tau_probe_max
)
```

这让 `Q_probe-consistency` 从抽象概念变成可实现判据。

---

### 5.4 Single-domain existence rule

最终规则：

```python
For top-M reference candidates:
    run bounded selector probe
    compute coverage_ok, topology_ok, ref_risk_ok, probe_ok

If any candidate satisfies all four:
    choose the best single-forward domain
Else:
    go to cluster branch
```

### best single-forward domain 的选择

在所有满足四个条件的候选中，再按一个 ranking 选最优，例如：

[
Score_{single}(S_i)=
-w_1 \cdot coverage_mean
-w_2 \cdot ref_dist_q90
-w_3 \cdot trans_q90
+w_4 \cdot topology_margin
]

这样 large-scene gate 不再是“很多阈值拼盘”，而是：

> 先问 single-domain feasibility 是否成立；
> 若成立，再从可行 single domains 中选最优；
> 若不成立，才 cluster。

---

## 6. Single 分支：选帧 + 自适应停止

这是第四个硬定义。
single 分支的核心不再只是 topology-aware ASB，而是：

## Reference-Risk-Aware TA-ASB + Auto-Stop

---

### 6.1 候选池构造

沿用你当前 `anchor_support_select_views()` 的主框架：

* root/reference
* adaptive scene params
* supportable candidate filtering
* oversampled candidate pool 

但逻辑上把 candidate pool 理解成：

> 当前 reference 下的 **reference-safe candidate domain**

而不是简单的“supportable views 集合”。

---

### 6.2 增量选帧目标

每步选点分数改成：

[
Score(v \mid S)=
w_c \Delta Cov(v)
+
w_s \Delta Supp(v,S)
+
w_t \Delta Topo(v,S)
--------------------

w_r Risk_{ref}(v,S)
+
w_p ProbeBonus(v,S)
]

其中：

* `ΔCov`：coverage gain
* `ΔSupp`：shared-neighbor / local support 增益
* `ΔTopo`：component merge、degree 提升、bridge risk 降低
* `Risk_ref`：ref q90/max、far-view budget、baseline-to-ref 恶化
* `ProbeBonus`：若轻量 probe 指示该点改善 pose consistency，可给 bonus

第一版里 `ProbeBonus` 可以先不做在线增量，只在 bounded probe 阶段评估整组。

---

### 6.3 自适应停止

只给一个 `max_views`。
停止条件必须同时满足：

#### Stop A: coverage adequacy

`coverage_ok`

#### Stop B: topology adequacy

`topology_ok`

#### Stop C: ref-risk adequacy

`ref_risk_ok`

#### Stop D: marginal gain saturation

最近若干步的：

* `Δcoverage_mean`
* `Δcoverage_max`
* `Δtopology_margin`
* `Δref_risk`

都很小。

### Stop rule

```python
if (
    len(S) >= min_views and
    coverage_ok and
    topology_ok and
    ref_risk_ok and
    saturated
):
    stop
```

也就是说：

* 不是 coverage 好了就停
* 不是 topology 好了就停
* 而是 **single-reference-safe domain 已经够好了** 才停

---

## 7. Cluster Branch

这一部分继续保留 ACE-style clustering，但接口更硬。

---

### 7.1 Cluster Planner

使用 ACE 原版的层次式二分 kMeans 思路：

* 全部 views 为一个 cluster
* 每次拆当前最大 cluster
* 使用 camera centers 做 k=2 split
* 直到满足 termination rule 

### 终止规则

不是固定 `num_clusters=4`，而是：

> 只要某个 cluster 仍不满足 single-domain feasibility，就继续 split。

所以 cluster planner 的目标不是“分成 K 份”，而是：

> 把全场景分解为若干个 **reference-local feasible domains**

---

### 7.2 每个 cluster 内部仍然 single-forward

每个 cluster 内部都运行完整 single pipeline：

* 选 reference
* bounded probe
* TA-ASB
* auto-stop
* 构建 one cluster-local memory

也就是说，cluster 不是另一套 memory construction，只是：

* **先分 domain**
* 每个 domain 内继续 single-forward

---

## 8. Cluster Ensemble Interface

这是 reviewer 特别要求说清楚的点，我这里明确固定第一版接口 

### 第一版明确采用：

* **No learned routing**
* **No cross-cluster feature fusion**
* **Each cluster predicts independently**
* **Each cluster prediction is transformed from `ref_i` to `world`**
* **Final pose selected by DSAC score / inlier count**

---

### 8.1 Query 端怎么处理

这里必须明确，第一版采用 reviewer 建议的 **A 路线**：

> **Query backbone feature 不随 cluster reference 改变。**

也就是说：

* query 自身 backbone feature 只算一次
* cluster reference frame 只作用于：

  * memory geometry contract
  * memory feature contract
  * 输出坐标的 ref->world recovery

而不是对 query 也重复做多次 reference-conditioned MapAnything forward。

### 为什么这样定

因为如果 query 端也要对每个 cluster 再做一次 MapAnything-style conditioning，就会把 cluster ensemble 成本抬得很高，而且会把研究问题从 memory policy 扩展到 query-side conditioning，变量失控。

所以第一版明确采用：

* query feature = cluster-invariant backbone feature
* cluster-specific memory fusion 输出 (P_{ref_i})
* 再通过 (T_{ref_i\to world}) 恢复为 (P_{world})
* 最后每个 cluster 各自跑 PnP / DSAC
* 取 score 最优者

---

## 9. 研究问题的最终表述

到 v2 为止，我建议把整个问题正式写成：

## Reference-Consistent Adaptive Memory Construction for MapAnything-Conditioned Visual Relocalization

而不再叫：

* adaptive ASB
* topology-aware view selection

因为真正的系统问题已经很明确了：

1. **reference-local memory contract 是什么**
2. **single reference-local domain 是否存在**
3. **存在时如何做 topology-aware + reference-risk-aware + auto-stop 的 single-forward construction**
4. **不存在时如何分解成多个 reference-local domains 并通过 ensemble fallback 恢复到 world-frame relocalization**

这才是完整问题。

---

## 10. 建议的实验问题重写

最后我把实验问题也按 v2 重新固定，避免继续发散。

### Q1. Assumption A 是否成立？

同一组选帧，改变 reference，比较：

* points_ref 的刚体可解释性
* predicted poses 的 ref-swap consistency
* pose residual 的变换一致性

这一步用于验证 `feature frame ≈ reference camera frame` 是否成立。

---

### Q2. Single-domain feasibility gate 是否有效？

比较：

* fixed mean-center reference
* top-M references + bounded probe
* always single

指标：

* single-domain success rate
* gate false positive / false negative
* PoseEval / DSAC 指标

---

### Q3. Auto-stop 是否避免过选？

比较：

* fixed 20 / 30 / 40
* auto-stop
* auto-stop without ref-risk
* auto-stop without topology

指标：

* final view count
* stop reason
* coverage
* ref-risk
* probe residual
* final pose accuracy

---

### Q4. Cluster branch 是否只是 fallback，而不是默认更优？

比较：

* always single
* always clustered
* auto-policy

指标：

* policy decision reason
* cluster count
* per-cluster ref-risk
* final ensemble pose score
* 误判案例分析

---

## 11. 最终收敛版结论

我现在会把方案最终收敛成四条核心决策：

### 决策 1

**先固定 reference-local memory contract，再谈选帧。**

### 决策 2

**single 分支的核心不是 topology 本身，而是 single-reference-safe domain construction。**

### 决策 3

**large-scene gate 的本质是 single-domain feasibility test，不是 scene size threshold。**

### 决策 4

**cluster 分支第一版只做 independent ensemble fallback，不引入 learned routing 和 cross-cluster fusion。**

这四条一旦固定，整个方案就不会在实现阶段继续发散了。

下一步最合适的是把这个 v2 方案写成一份真正的技术设计文档版本，直接分成：

* assumptions
* schema
* gate
* single branch
* cluster branch
* experiments
* engineering plan

这样你后面无论是继续讨论方法、写实验计划，还是准备落代码，都会更稳。
