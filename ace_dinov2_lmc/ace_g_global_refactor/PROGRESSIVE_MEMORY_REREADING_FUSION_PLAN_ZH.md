# 递进式 Memory 重读融合实施规划

更新时间：2026-06-13

英文名称：**Progressive Memory Re-reading Fusion（PMRF）**

阅读提示：本文将方法公式、可读伪公式和实现草图区分开。第 3 节用于理解方法；第 5 节才是代码落地草图。

本文重新评估论文第一个结构创新点。结论是：不再把主要问题定义为“memory 注入可能伤害 baseline，因此需要零门控保护”，而是定义为：

> 当前单次 query-to-memory cross-attention 是否充分利用了 compressed memory，以及能否让已经获得场景上下文的 query 再次读取 memory，从而得到更有效的 token routing 和 scene-coordinate feature。

PMRF 替代此前规划中的 **Zero-Gated Residual Memory Refinement / CIFA**。SfM Track 引导的多视图监督（STGS）仍保留为独立的监督创新。

## 1. 问题定义修正

### 1.1 当前证据不支持“memory 普遍有害”

Indoor6 `scene2a` 上，正确训练合同下的 true-global memory fusion 是有效的：

```text
FGPI-4090 true global + per_iter
pct5 = 83.66
pct2 = 34.24
med_t = 2.6233 cm
med_r = 0.2831 deg
```

它优于对应 local-fallback / fixed-zero controls。因此不能把 memory conditioning 的主要矛盾写成“memory 有时有害，所以必须保守注入”。

GLACE-LMC 在 Wayspots SquareBench 的退化也不能直接归因于 memory feature。GLACE decoder 输入是 `global descriptor + local feature`，修改 global feature 会同时影响 multi-center routing 和 coordinate residual regression。这个现象首先是 GLACE global/local feature contract 问题。

### 1.2 当前真正的缺口是“融合是否足够有效”

现有 `LMCFusionBlock` 只执行一次 memory 读取：

```text
raw query feature
  -> query-to-memory attention
  -> residual + FFN
  -> coordinate head
```

第一次 attention 的 query 只来自 backbone/local feature，还没有吸收当前场景 memory 提供的上下文。在重复纹理或局部外观相似区域，一次 token routing 可能不够。

已有 diagnostics 也给出弱信号：`value_only_raw` 下 64 个 latent tokens 的全局 effective token count 约为 57-58。该统计不能单独证明 attention 错误，但说明当前读取很分散，没有证据表明一次读取已经充分完成 memory selection。

## 2. 历史方向审计

### 2.1 应保留的稳定合同

- Layer12 是稳定 key anchor。
- selected Layer12 key + all-layer concatenated value 是当前接受合同。
- `value_only_raw` 是 fusion geometry reference。
- `s1_loss_step_mode=per_iter` 对 true-global 路径重要。
- memory fusion 本身已有收益，下一步应提升利用效率，而不是默认压低贡献。

### 2.2 不应原样继续的方向

**Memory-side GeoKey：** `geokey_norm` 得到 `pct5=77.04`、`pct2=28.02`、`med_t=3.0951cm`。仅把 memory 坐标加入 key 没有改善 matching，因为仍缺少可靠 query-side 场景上下文。

**Scalar key-layer mix：** 最终权重约为 `0.00034, 0.00034, 0.99862, 0.00034, 0.00034`，几乎完全退回 Layer12，不应继续简单全局层权重混合。

**Compression-side levelwise merge：** B3-lite 明显低于 FGPI，说明在 latent bottleneck 处提前合并 multi-level memory 过于扰动稳定合同。

**Query-side DPT/FPN：** 理论上合理，但需要 DINO 中间层提取、buffer/recompute policy、S1/S2/test 重建和更大接口变化，更像 backbone enhancement。

**Detached routing / coordinate GeoMatch：** 独立 routing head 缺少直接 token label；coarse coordinate GeoMatch 又依赖不可靠的 pre-fusion coordinate。二者都比利用第一次融合结果生成第二次 query 更复杂。

### 2.3 当前 CIFA 的可取部分与问题

`cascade_internal` 已有有价值的雏形：先用原始 query 读取 memory 得到 `H1`，再用 `H1` 继续读取 memory 得到 `H2`。写成公式就是：

- H₁ = F₁(Q, Z)
- H₂ = F₂(H₁, Z)

问题在于旧设计后面又加入了 assembly MLP 和零初始化 gate：

- R = G([H₂, ...])
- H_out = H₁ + γR
- γ₀ = 0

这会导致：

1. 叙事转向 safe fallback，而不是有效融合。
2. gamma=0 时，第一步训练中后续 cascade 和 assembly 收不到主损失梯度。
3. CIFA-2 约增加 16M 参数，难以区分 routing 收益与容量收益。
4. 第二个完整 block 重复计算 memory PE、K/V 和 4x FFN，很多计算与重新 routing 无关。

因此应保留 context-conditioned second read，删除 zero-gated assembly 和重复完整 block。

## 3. 首选设计：PMRF

### 3.1 核心假设

第一次 memory read 为 query 注入粗粒度场景上下文。更新后的 `H1` 比原始 `Q` 更适合重新判断当前 patch 应关注哪些 memory tokens。

换句话说，PMRF 不是为了“安全地少用 memory”，而是为了检验：

```text
raw local query
  -> first memory read: get scene-aware query
  -> second memory read: re-route with scene-aware query
  -> coordinate head
```

### 3.2 符号约定

| 符号 | 含义 |
|---|---|
| `Q` | query feature，来自当前图像 / patch 的局部特征 |
| `Z` | compressed memory token features |
| `P` | memory token 对应的 3D latent point 坐标 |
| `C` | scene center 或归一化参考中心 |
| `PE(P-C)` | 对 memory 3D 坐标的 positional encoding |
| `K, V` | memory key 和 value，只编码一次并在两次读取中共享 |
| `A1, A2` | 第一次和第二次 query-to-memory attention |
| `H1` | 第一次读取后的 scene-aware query feature |
| `H2` | 第二次读取后的最终 fusion feature |

### 3.3 计算形式

下面的公式写成“可直接读”的形式，避免把实现代码和方法定义混在一起。

**第一步：只编码一次 memory。**

公式：

- K = WₖZ
- V = Wᵥ(Z + φ(P - C))

含义：

- key 仍来自当前接受的 memory feature 合同；
- value 保留现有 value-side geometry injection；
- `K,V` 在两次读取中共享，不重复计算 PE/K/V。

**第二步：第一次读取，保持 baseline 语义。**

公式：

- A₁ = softmax((Wq₁ Q Kᵀ) / √d)
- H₁ = BaselineFusion(Q, A₁V)

含义：

- `A1` 是当前 baseline 的 query-to-memory attention；
- `H1` 是已经吸收一次 memory context 的 query feature；
- 这一步必须和 `single` baseline 的行为对齐。

**第三步：用 scene-aware query 重读同一份 memory。**

公式：

- A₂ = softmax((Wq₂ LN(H₁) Kᵀ) / √d)
- ΔH₂ = Wo₂(A₂V)
- H₂ = LN(H₁ + ΔH₂)

含义：

- `A2` 的 query 不再是原始 local feature，而是 `H1`；
- 第二次读取仍看同一组 `K,V`，因此验证变量只有“query 是否更懂场景”；
- `DeltaH2` 是第二次 memory read 带来的增量；
- `H2` 直接送入已有 coordinate head。

### 3.4 第一版边界

- 不使用 zero-gated final assembly。
- 不 concat `H1/H2` 后再过大 MLP。
- 不重新计算 memory PE、K 和 V。
- 不增加第二个 4x FFN。
- 不使用 hard top-k、coarse-coordinate bias 或 routing loss。
- 不修改 memory file、compressor output、buffer、coordinate head 和 DSAC/PnP。

第二次读取从第一步训练开始就是活跃路径，直接接收主损失梯度。

### 3.5 为什么共享 K/V

第一版只验证一个变量：query 获得场景上下文后，重新计算 attention 是否有帮助。共享 K/V 能保持 accepted memory representation 不变，减少参数，并允许直接比较 `A1` 与 `A2`。

当 `C=1024` 时，第二次读取只增加 `q2 projection + out2 projection`，约 2.1M 参数，远低于当前 CIFA-2 的约 16M。

## 4. PMRF 与当前 CIFA 的差异

| 当前 CIFA | PMRF |
|---|---|
| 完整第二 FusionBlock | 轻量 memory re-read block |
| 第二套 PE/K/V | 共享第一次 K/V |
| 第二套 4x FFN | 第一版不增加 FFN |
| concat assembly MLP | 删除 |
| zero-init gamma | 删除 |
| `H1 + gamma*R` | `LN(H1+DeltaH2)` |
| safe fallback | context-aware routing |

建议代码模式：

```text
lmc_fusion_refinement_mode = progressive_reread
```

保留 `single` 和 `cascade_internal` 以兼容旧 checkpoint 和历史 ablation。

## 5. 逐文件修改规划

### 5.1 `/home/xwh/project/ace_depth/ace_fusion.py`

把 memory-side 计算拆成内部 helper。下面是实现草图，不是方法公式：

```python
K, V, memory_stats = encode_memory(memory_z, memory_p, scene_center)
attention_out, attention, read_stats = read_memory(query, K, V)
```

默认 `single` 路径仍使用同一 helper，必须保持数值行为不变。

新增 `LMCProgressiveRereadBlock`。它只做第二次读取和残差归一化：

```python
query2 = q_proj(layer_norm(H1))
attention2 = softmax(query2 @ K.transpose(-2, -1) / sqrt(d))
delta_h2 = out_proj(attention2 @ V)
H2 = layer_norm(H1 + delta_h2)
```

该 block 不含 coordinate encoder、K/V projection、4x FFN 和 final gate。

`LMCFeatureFusion` 增加 `progressive_reread` 分支：

```python
K, V, memory_stats = self.encode_memory(memory_z, memory_p, scene_center)
H1, A1, read1_stats = self.forward_single_read(query, K, V)
H2, A2, read2_stats = self.progressive_reread(H1, K, V)
return H2, merge_stats(memory_stats, read1_stats, read2_stats)
```

### 5.2 `options_dinov2_lmc.py`

增加：

```text
--lmc_fusion_refinement_mode progressive_reread
```

第一版固定一次 reread，不暴露 gate、top-k、temperature 或 independent-K/V 参数。

### 5.3 `trainer_dinov2_lmc.py`

- 验证并构造 `progressive_reread`；
- 将 mode 写入 `lmc_config`；
- S1 及 fusion-in-S2 optimizer 加入 reread 参数；
- 记录两次读取的 attention、feature 和成本诊断。

### 5.4 Evaluator

修改：

```text
test_ace_dinov2_lmc.py
test_ace_dinov2_lmc_ensemble.py
```

按 checkpoint 重建 PMRF；旧 checkpoint 缺字段时默认 `single`。

### 5.5 合同测试

新增 `test_lmc_progressive_reread_contract.py`，覆盖：

1. `single` 默认路径数值不漂移；
2. PMRF shape 正确；
3. reread 参数第一步就获得有限、非零梯度；
4. 两次读取确实共享 K/V；
5. 输出依赖 memory；
6. state-dict round-trip；
7. train/test reconstruction 一致；
8. diagnostics 有限。

## 6. 必须记录的诊断

Attention：

```text
read1/read2 attention entropy
read1/read2 per-query effective tokens
read1/read2 avg max attention
read1/read2 global usage effective count
read1-read2 JS divergence
read1-read2 top1 agreement
```

Feature 与成本：

```text
H1 norm
DeltaH2 norm
DeltaH2 / H1 ratio
H1-H2 cosine
H2 norm
additional parameters
fusion latency
checkpoint size
peak memory
```

`A2` 熵更低不自动代表更好，可能只是错误 attention 变尖；必须与 pose metrics 联合判断。

## 7. 最小实验矩阵

主场景使用 Indoor6 `scene2a`，固定 FGPI 合同：

```bash
--lmc_profile legacy
--s1_loss_step_mode per_iter
--lmc_key_slice_idx 2
--lmc_feature_hierarchy_mode selected_key_concat_value
--lmc_fusion_geometry_mode value_only_raw
```

| ID | 设置 | 目的 |
|---|---|---|
| F0 | accepted FGPI single fusion | control |
| F1-PMRF | single + one shared-K/V reread | 主候选 |
| F1-FFN | H1 后增加参数量接近的 query-only adapter，不再读 memory | 容量对照 |

只有 F1-PMRF 明显优于 F1-FFN，才能说明收益来自 memory re-reading，而不是额外参数。

第一轮不运行多次 reread、independent K/V、top-k、attention prior、PMRF+PointRoPE 或 PMRF+STGS。

scene2a 有正信号后，再运行 Indoor6 scene5 或 scene3。之后才在 Wayspots SquareBench 验证。GLACE 实验必须使用 local fusion target 或明确保持 global prefix 不变。

## 8. 与 STGS 的组合

PMRF 和 STGS 独立通过后，再运行：

| ID | PMRF | STGS |
|---|---|---|
| C0 | off | off |
| C1 | on | off |
| C2 | off | on |
| C3 | on | on |

互补性仍需由 2x2 实验支持。

## 9. 成功与停止标准

晋级条件：

- 5-seed post-train 中 Acc5 或 Acc2 稳定改善；
- median translation/rotation 不实质退化；
- F1-PMRF 优于 parameter-matched F1-FFN；
- `A2` 与 `A1` 存在有意义的 routing 差异；
- `DeltaH2` 非零但不覆盖 `H1`；
- 第二场景不明显退化；
- 参数和延迟增量显著低于 CIFA-2。

停止条件：

- F1-PMRF 与 F1-FFN 相同或更差；
- `A2` 与 `A1` 几乎完全相同；
- `A2` 只变尖但 pose metrics 下降；
- update norm 极小且无收益；
- 延迟不可接受；
- 收益只出现在单 seed 或单场景。

负结果后不要立即增加更多 reread layers、hard top-k、geometry bias 或 routing loss。

## 10. 论文定位

推荐英文：

> We introduce Progressive Memory Re-reading Fusion, a lightweight two-stage memory interaction module for scene coordinate regression. The first read injects coarse scene context into query features, while the second read re-queries the same compact memory using the context-enhanced representation. By sharing memory keys and values across stages, the module improves query-conditioned memory routing with limited parameter and inference overhead.

推荐中文：

> 我们提出递进式 Memory 重读融合。第一次读取将粗粒度场景上下文注入 query feature，第二次读取使用已经获得场景上下文的表示重新查询同一组 compact memory。两个阶段共享 memory key/value，从而以较小的参数和推理开销改善 query-conditioned memory routing。

避免表述：

- memory 注入通常会伤害 baseline；
- PMRF 的目标是抑制 memory；
- GLACE SquareBench 的退化证明 LMC memory 有害；
- 第二次读取一定产生更尖锐 attention；
- 多次读取等价于多层 backbone feature fusion。

## 10.1 论文新颖性风险

小范围论文查重表明，iterative cross-attention 不是新的通用算子：Perceiver 使用 iterative attention；ICAFusion 使用参数共享的 iterative cross-attention；SACReg 已在 SCR 中使用 query/database cross-attention。

因此 PMRF 不能声称：

- 首次提出 iterative cross-attention；
- 首次在视觉任务中重复读取 memory；
- 仅靠堆叠第二层 attention 就构成充分创新。

可辩护的贡献必须限定为以下组合：

1. 面向 compressed 3D scene memory 的 context-enhanced query re-read；
2. 两阶段共享由 memory feature 和 latent 3D point 编码得到的 K/V；
3. 不改变 SCR head 和 DSAC/PnP 的插件式实现；
4. parameter-matched query-only control 证明收益来自再次读取 memory；
5. attention 路由变化与 pose improvement 的联合实证。

如果 PMRF 只获得很小收益，或 F1-FFN 能复现同样收益，它不适合作为主创新，只能作为 fusion-depth ablation。

## 11. 最终判断

PMRF 是当前最可行的第一个结构创新点：

1. 它直接回答“如何让融合更有效”。
2. 它保留 CIFA 中有价值的 context-conditioned reread，但删除零门控大 assembly 和重复完整 block。
3. 它不修改 memory、compressor、buffer、head 和 DSAC/PnP 合同。
4. 它只增加约 2.1M 参数，远低于 CIFA-2 的约 16M。
5. parameter-matched FFN control 可以验证收益是否真的来自再次读取 memory。
6. 它与 STGS 分别位于结构和监督层面，仍能形成清晰的双创新故事。

后续双插件方法改为：

```text
结构侧：Progressive Memory Re-reading Fusion（PMRF）
监督侧：SfM-Track Guided Multi-View Supervision（STGS）
```
