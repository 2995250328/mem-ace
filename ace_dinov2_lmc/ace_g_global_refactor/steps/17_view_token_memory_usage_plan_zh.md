# Step 17：Memory 中 View Token 的保守使用方案

## 1. 目的

本文档记录一个保守方案，用于利用 ACE-DINOv2-LMC memory 文件中已经保存的中间 view-level token。

在当前代码库中，相关字段不是显式的 `camera_token`，而是 `all_scale_tokens`。它是目前最接近 per-view global token 的字段。这些 token 以 memory view 为粒度保存，来源于 memory extraction 过程中的 CLS 风格全局 token。

关键问题不是这些 token 是否存在，而是如何在不破坏 ACE-G 已接受契约、不 destabilize compressor/fusion、也不把弱全局语义误当成 dense metric geometry 的前提下使用它们。

因此本文将区分：

1. 现在已经有什么；
2. 现在已经怎么用了；
3. 下一步什么做法是安全的；
4. 什么做法目前不该做。

## 2. 当前代码状态

当前 token-like memory 字段是：

```text
all_scale_tokens: [M, D]
```

其中：

- `M` = memory views 数量；
- `D` = token 维度。

当前系统行为是保守的：

- memory extraction 会保存 `all_scale_tokens`；
- memory loading 会保留它；
- trainer/test 会把它放进 `memory_dict`；
- GeoLMC 可以通过轻量的 scale-token injection 路径消费它；
- fusion 不会直接消费 raw view tokens；
- 当前训练/test 的 fusion API 不会传 query-side CLS tokens。

现有使用方式非常弱，而且是 summary-based：

```text
all_scale_tokens [M, D]
-> mean over views
-> small MLP / scale-token injection
-> added to latent features
```

这说明代码库已经支持一种最小化的 memory-side view token 用法，但还不支持完整的 token-aware fusion 设计。

## 3. 这些 token 应该如何理解

当前应把这些存储的 token 视为 **view-level global descriptors**，而不是 dense coordinate evidence，更不是可靠的 metric pose tokens。

它们可能包含：

- 全局外观线索；
- scene-level 语义上下文；
- 粗粒度视角/布局信息；
- 弱视点或尺度信息；
- 来自 MapAnything forward 的 reference-conditioned context。

不应假设它们稳定包含：

- 精确 camera pose；
- 直接的 metric 3D 约束；
- 稳定的跨参考帧不变性；
- 可替代 geometric memory points 的能力。

因此它们最适合扮演 **side-channel signal**：

- memory-side conditioning；
- token pooling；
- latent modulation；
- memory routing；
- 可选的 calibration 或 gating。

## 4. 总体建议

第一步不要把 raw `all_scale_tokens` 直接插入 dense fusion 或 query features。

应该采用分阶段路线：

1. 先做 diagnostics；
2. 再做低风险 memory-side summary use；
3. 然后考虑 optional latent modulation；
4. 之后再考虑 multi-memory routing；
5. query-side CLS/view-token conditioning 延后。

baseline ACE-G contract 保持不变：

- 保留 accepted C0/FGPI-style path；
- 保留当前 training/test/checkpoint compatibility；
- 所有 token 相关新增逻辑必须默认关闭；
- 当 memory 文件缺少 `all_scale_tokens` 时必须完整 fallback。

## 5. Stage 0：先做 diagnostics，再改训练

在引入新的 learned use 之前，先检查这些 stored view tokens 是否真的携带稳定信息。

推荐 diagnostics：

```text
shape / dtype / device
per-dimension mean and std
token norm distribution
pairwise cosine similarity distribution
PCA / t-SNE / UMAP projections
token similarity vs camera-center distance
token similarity vs viewing-direction difference
token similarity vs covisibility / overlap if available
token clustering vs spatial scene region
```

目标是回答：

```text
Do nearby or overlapping views have token similarity structure that the model could exploit?
```

如果答案很弱或不稳定，那么这些 token 应继续只作为辅助 metadata。

## 6. Stage 1：更安全地替代当前 mean pooling

最低风险的下一步是改进当前 summary 操作：

```text
summary = mean(all_scale_tokens)
```

替换或增强为可选 view-token pooling：

```text
summary = ViewTokenPool(all_scale_tokens)
```

推荐第一版：

```text
learnable query
-> cross-attention over all_scale_tokens
-> pooled summary
-> residual blend with mean summary
```

概念形式：

```text
summary_mean = mean(all_scale_tokens)
summary_attn = CrossAttention(q_learned, all_scale_tokens)
summary = summary_mean + gamma * proj(summary_attn)
```

其中 `gamma=0` 或近零初始化。

这个方向有吸引力，因为它：

- 完全留在 memory side；
- 保持当前 feature buffer 和 query path；
- 只修改当前 token-summary 逻辑；
- 易于关闭和比较。

## 7. Stage 2：在 GeoLMC 内部做 latent modulation

一个更强但仍保守的选项，是用 pooled view-token summary 去调制 latent features。

概念上：

```text
summary = ViewTokenPool(all_scale_tokens)
scale, shift = MLP(summary)
latent = latent * (1 + gamma * scale) + gamma * shift
```

或者：

```text
latent = latent + gamma * latent_bias(summary)
```

相比 raw token 注入 fusion，这条路径更优，因为它：

- 仍留在 compressor side；
- 不需要修改 public fusion forward signature；
- 避免把 view tokens 和 dense query patch tokens 直接混在一起；
- 与当前 `all_scale_tokens` 的 memory-side side signal 定位一致。

设计约束：

- 初始化为 identity 或 zero residual；
- 不改变 feature dimension；
- 需要在 `lmc_config` 中显式保存 config；
- 当 `all_scale_tokens` 缺失时要干净 fallback。

## 8. Stage 3：multi-memory routing descriptor

如果系统后续走向 multiple local memories 或大场景 memory routing，`all_scale_tokens` 可以变成 memory-level descriptor。

对每个 memory bundle，计算：

```text
memory_descriptor = pool(all_scale_tokens)
```

可用于：

- memory clustering；
- memory similarity analysis；
- top-k memory preselection；
- multi-memory routing scores；
- scene-chunk retrieval。

重要限制：

当前 training/test path 不会传 query-side CLS tokens。因此第一版 routing 不应声称已经实现真正的 query-adaptive token matching。

更安全的早期用法是：

- memory-vs-memory organization；
- local-map grouping；
- 带离线 oracle 支持的 routing analysis；
- 等 test path 中有稳定 query global descriptor 后再做更强版本。

## 9. 可选后续扩展：camera-pose-conditioned view tokens

更激进的方向是把每个 stored view token 与显式 camera metadata 结合。

对每个 memory view：

```text
view_token_i
+ pose_encoding(camera_center_i, rotation_i, viewing_direction_i)
+ reference-frame metadata
```

可以形成更丰富的 per-view descriptor，用于：

- geometry-aware token pooling；
- routing；
- memory-view scoring；
- attention bias。

但这应当作为更后面的选项，因为它带来几类风险：

- frame convention mistakes；
- accidental coupling to reference-conditioned coordinates；
- overclaiming metric meaning from weak global descriptors；
- 更大的 metadata contract。

如果要尝试，也应先只作为 attention bias 或 side descriptor，而不是替代 geometric memory points。

## 10. 当前不该做的事情

以下做法在第一轮应避免：

1. **不要把 raw `all_scale_tokens` 直接送进 dense fusion。**
   它们是 view-level global tokens，不是 patch-wise query features。

2. **不要默认改变 accepted C0/FGPI baseline。**
   所有新 token 逻辑必须 opt-in。

3. **不要假设它们就是精确 camera pose tokens。**
   它们至多是弱 camera/view-conditioned global descriptors。

4. **不要现在就加 query CLS guidance。**
   当前 trainer/test/ensemble 路径不传 query CLS tokens 到 fusion。

5. **不要把它们当作 direct metric geometry supervision。**
   它们应该做调制或总结，而不是替代 point-based geometry。

6. **不要立刻把它和 PointRoPE、multi-level fusion 或 multi-memory routing 其他改动绑在一起。**
   先 isolate 机制本身。

## 11. 实验计划

### Stage A：offline analysis only

目标：

- 判断 `all_scale_tokens` 是否携带稳定且与场景相关的信息。

交付物：

- token statistics report；
- token-to-pose correlation report；
- PCA/t-SNE plots；
- 是否继续推进的建议。

### Stage B：dry-run forward compatibility

目标：

- 验证 optional pooling/modulation modules 在不改变训练语义前提下能正常前向。

检查项：

- shape；
- dtype；
- NaN/Inf；
- device placement；
- latency；
- checkpoint save/load compatibility。

### Stage C：pooled-summary ablation

比较：

```text
none
mean
attention_pool
```

指标：

- training stability；
- S1 alignment loss；
- S2 reprojection loss；
- median translation/rotation；
- `pct5`；
- runtime cost。

### Stage D：latent modulation ablation

比较：

```text
baseline
latent_bias
latent_film
```

同样保持与其他 geometry/fusion 改动隔离。

### Stage E：later routing use

只有在上面稳定后：

- 用 pooled token descriptors 做 memory grouping/routing；
- 与 memory routing baselines 比较；
- 把 query-adaptive token routing 视为未来扩展。

## 12. 文档与配置建议

这个方向不应并入当前 Step15 主线。Step15 已经聚焦在 PointRoPE、fusion-side internal refinement 和 compression-side residual multi-level changes。

它也不应替代 Step16 的大场景多 memory 主计划。

因此，正确位置就是像本文这样单独成 note。

如果后续进入实现，建议显式配置字段，例如：

```text
use_view_tokens
view_token_mode = none | mean | attn_pool | latent_bias | latent_film
view_token_gamma_init
view_token_pose_bias = false | true
```

这些字段都应保存在 `lmc_config` 中，并默认回到当前 no-change 行为。

## 13. 最终建议

把 `all_scale_tokens` 视为一个有潜力、但当前被低度利用的 memory-side side channel。

第一步的正确做法不是把它送进 dense fusion，而是：

1. 先测它到底携带什么信息；
2. 用更安全的 optional learned pooling 替代当前 mean pooling；
3. 可选地调制 GeoLMC latent features；
4. 未来把同样的 summary 复用到 multi-memory routing；
5. 等 training/test API 干净支持后，再讨论 query-side CLS/view-token guidance。

这样既保住 ACE-G baseline，又打开了一条低风险利用 stored global view tokens 的路径。
