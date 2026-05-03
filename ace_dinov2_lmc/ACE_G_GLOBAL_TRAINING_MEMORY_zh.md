# ACE-G Global 训练链路长期记忆

本文档记录 `ace_dinov2_lmc` 中 **`--lmc_flow ace_g` 且 `--lmc_mode global`** 的真实训练链路与后续讨论约束。后续分析 LMC、fusion、S1/S2、compressor 优化时，应以本文为默认前提。

范围：

- 只讨论 `ACE-G + global GeoLMC` 的真实训练路径。
- 不展开 local / hierarchical / learned LMC。
- 不展开 memory 构建流程。
- 核心代码涉及：
  - `ace_dinov2_lmc/trainer_dinov2_lmc.py`
  - `ace_compressor.py`
  - `ace_fusion.py`
  - `ace_network_dinov2.py`
  - `ace_loss.py`

---

## 1. 总体结构

ACE-G Global 不是单阶段 end-to-end trainer，而是一个外层迭代框架。每轮包含两阶段：

1. **S1**：训练 `compressor + fusion + head`
2. **S2-G**：训练 `head`，并可选继续训练 `fusion`

DINOv2 encoder / backbone 在整个 LMC 训练中保持冻结，只负责提供 raw image feature。

真正参与学习的新模块是：

- `GeoLMC(global)`：把 memory 点云特征压成少量 latent tokens。
- `LMCFeatureFusion(global)`：把当前图像特征和 latent tokens 做 cross-attention 融合。
- ACE Head：从 fused feature 回归 scene coordinates。

整条链的本质是：

```text
raw image feature -> fusion(memory latent) -> ACE head -> reprojection loss
```

其中 `memory latent` 不是常量，而是由 `GeoLMC` 从 memory 动态编码得到。

---

## 2. 每轮 ACE-G iteration 的顺序

ACE-G 主循环每轮执行：

1. 跑 S1。
2. 可选做一次 cross-iteration eval。
3. 按策略 reset head。
4. 用当前 compressor 把 memory 压一次，缓存到 `self._s2_compressor_out`。
5. 构建只含 raw backbone feature 的 S2 buffer。
6. 配置 S2 optimizer 和 scheduler。
7. 跑 S2-G。
8. 保存 checkpoint 并评估。

关键差异：

- ACE-G 的 S2 buffer **不存 fused feature**。
- ACE-G 的 S2 buffer 只存 **raw backbone feature**。
- fusion 在 S2 每个 batch 内即时计算。

这是 ACE-G 与普通 iterative 流的核心区别。

---

## 3. GeoLMC(global) 的真实内部功能

### 3.1 输入张量约定

`GeoLMC(global)` 输入约定：

```text
pooled_points:    (B, N, 3)
pooled_features:  (B, N, C * L)
scene_center:     (B, 3)
all_scale_tokens: optional (B, V, D)
```

其中：

- `N` 是 memory 点数。
- `L` 是多层 DINO 特征层数。
- `C` 是每层特征维度。
- `K` 是 `num_latent_tokens`。

### 3.2 Key / Value 输入不对称

`GeoLMC` 内部 key 和 value 的输入不同：

- `k_proj` 不吃完整多层特征，而是只吃 `_get_layer_slice(pooled_features, 1)` 的一个 slice。
- `v_proj` 吃完整的 `C * L` 拼接特征，再投到 `compress_dim`。

含义：

- key 更像一个单层检索视角。
- value 才承载完整 memory 内容。

因此 compressor 里的 key/value 语义并不对称。后续优化 LMC 时要注意这一点。

### 3.3 latent 坐标来自随机起点 FPS

如果 `N > K`，则对 `pooled_points` 做 FPS 选出 latent 坐标。

注意：FPS 初始点来自 `torch.randint`，所以 latent 坐标集合不是严格确定常量，而是带随机起点的 FPS 结果。

这对复现实验和 token 稳定性很重要。

### 3.4 compressor query 是纯几何 PE

`GeoLMC(global)` 的初始 query 来自：

```text
latent_coords - scene_center -> FourierPositionEncoding -> q
```

也就是说：

- query 完全由几何位置生成。
- query 不直接由 memory feature 初始化。

这说明当前 global GeoLMC 的 latent tokens 本质上是 **geometry-anchored latent queries**。

### 3.5 global attention = content attention + learned geometry bias

`GeoAttentionBlock(global)` 内部大体执行：

1. 对 query 做 LayerNorm。
2. 线性投影得到 q/k/v。
3. 标准 attention logits：

```text
q @ k.T / sqrt(d)
```

4. 计算几何项：

```text
log(dist_sq + 1e-6) -> geo_bias_mlp -> per-head geometry bias
```

5. 将 geometry bias 加到 attention logits。
6. softmax 后聚合 value。
7. residual + MLP residual。

因此 global 模式不是“完全不看几何的全局 attention”，而是：

```text
global content attention + learned geometric bias
```

### 3.6 scale token 是全局 shift，不是逐 token 对齐注入

如果存在 `all_scale_tokens`：

1. 先沿 view 维求平均。
2. 经过 `scale_mlp`。
3. 得到 `(B, C)` 全局偏置。
4. 加到每个 latent token 上。

因此 scale token 是整组 latent 共享的 global shift，不是和每个 latent token 局部对齐的 scale/geometry 信息。

---

## 4. Fourier Position Encoding 的关键事实

`FourierPositionEncoding` 工作方式：

1. 输入 coords 一般形状为 `(B, M, 3)`。
2. 如果 `normalize_input=True`，按 batch 内坐标 std 归一化。
3. 用固定随机矩阵 `B_gauss` 投影。
4. 拼接 `sin(2πx)` 和 `cos(2πx)`。
5. 经过两层 MLP 输出目标维度。

注意：

- `B_gauss` 是 buffer，不训练，但模块初始化时随机采样。
- compressor 和 fusion 各自有独立 Fourier PE，不共享参数。

在 ACE-G global 中，PE 出现两次：

1. compressor query：

```text
latent_coords - scene_center
```

2. fusion value：

```text
memory_p - scene_center
```

这意味着当前系统中 geometry PE 既参与 memory compression，也参与 query-memory fusion，但两者参数不同、语义不同。

---

## 5. Fusion 的真实内部功能

`LMCFeatureFusion(global)` 输入：

```text
query_feats:  (B, Nq, 1024)  # image backbone feature
memory_z:     (B, K, Cmem)   # compressor output
memory_p:     (B, K, 3)      # latent coordinates
scene_center: (B, 3)
```

内部不是 symmetric attention，而是：

```text
q = q_proj(query_feats)
k = k_proj(memory_z)
pe = PE(memory_p - scene_center)
v_input = memory_z + pe_proj(pe)
v = v_proj(v_input)
attention = softmax(q @ k.T)
out = attention @ v
```

核心事实：

- attention logits 只看 `q(query_feats)` 和 `k(memory_z)`。
- memory 坐标 PE 只明确进入 value 支路。
- memory 坐标不会直接影响 query-key 匹配分数。
- 坐标只影响被读出来的内容表示。

这也是后续优化 LMC/fusion 时最重要的结构弱点之一。

Fusion 最后还有两层 residual：

1. `LayerNorm(residual + attention_out)`
2. `LayerNorm(x + FFN(x))`

另外，fusion 默认带 `Dropout(0.1)`：

- S1 训练时 fusion 有 dropout。
- ACE-G R2 的 S2 训练时 fusion 也有 dropout。
- R1 或 buffer fill 时 fusion 处于 eval，无 dropout。

---

## 6. S1 真正训练什么

S1 是 compressor 的唯一主训练阶段，不只是“预热 head”。

常用 preset 下通常是：

```text
s1_use_buffer=True
s1_loss_mode=sample_per_image
```

此时每个 update 的流程：

1. 从 raw buffer 随机抽 feature rows。
2. 裁成 `16 * floor(batch/16)`。
3. reshape 成 fake `1 x C x 16 x W`。
4. 直接调用 `self.compressor(self.memory_dict)`，这是带梯度路径。
5. 调 `_fuse_features(raw_features, comp_out)`。
6. fused feature 过 ACE head。
7. 计算 reprojection loss 和 invalid loss。
8. 反传更新 `compressor + fusion + head`。

注意：

- S1 不调用 `_compress_memory()`。
- S1 直接调用 `self.compressor(self.memory_dict)`。
- 因此 S1 是 compressor 的真实训练路径。

S1 optimizer 参数组：

- compressor：`base_lr_s1`
- fusion：`base_lr_s1`
- head：`0.1 * base_lr_s1`

也就是说，S1 中 head 被刻意降低学习率，主要目的是训练 compressor/fusion，而不是让 head 主导拟合。

---

## 7. S1 损失本质仍是 ACE 式 reprojection supervision

S1 sampled-feature 路径逻辑：

1. fused rows 被 pack 成 fake `1 x C x 16 x W`。
2. head 输出 `pred_scene_coords`。
3. 如果有 normalized contract，按当前训练坐标约定处理坐标。
4. 用 `gt_inv_pose` 把 scene coordinate 变到 camera space。
5. 用内参 `K` 投影到像素平面。
6. 计算 reprojection error。

监督分两类：

- valid 点：`ReproLoss.compute(...)`
- invalid 点：把预测 camera coord 往 `depth_target * invK * pixel` 这个 proxy target 拉

`ReproLoss` 默认是 `dyntanh`，soft clamp 会随 iteration 变化，不是固定阈值。

因此 S1 本质还是 ACE 式 reprojection supervision，只是输入特征换成了 fused feature。

---

## 8. S2-G 真正训练什么

S2 开始前先执行：

```text
self._s2_compressor_out = self._compress_memory()
```

而 `_compress_memory()` 是：

```text
self.compressor.eval()
with torch.no_grad():
    out = self.compressor(self.memory_dict)
self.compressor.train()
```

因此：

- S2 中 compressor 绝对冻结。
- 不存在 compressor 的隐式梯度泄露。

每个 S2 batch：

1. 从 raw feature buffer 取 rows。
2. pack 成 fake `1 x C x 16 x W`。
3. 用缓存的 `_s2_compressor_out` 做 fusion。
4. 过 head。
5. 计算与 S1 同构的 reprojection loss + invalid loss。
6. 更新参数。

R1/R2 区别：

- `ace_g_fusion_in_s2=False`：
  - fusion 包在 `torch.no_grad()` 中。
  - 只训练 head。
- `ace_g_fusion_in_s2=True`：
  - fusion 正常前向。
  - 训练 `head + fusion`。

当前常用 preset 通常是 R2。

---

## 9. S2 optimizer 与 schedule

R2 时，S2 optimizer 只含：

- `regressor.heads.parameters()`
- `fusion.parameters()`

不含 compressor。

学习率：

```text
head_lr = s2_learning_rate_max * head_lr_multiplier_s2 * boost
fusion_lr = head_lr * ace_g_fusion_lr_ratio
```

调度器是 `OneCycleLR`，并专门处理了小 `total_steps` 时 `pct_start` 退化的问题。

`step_eff` 有三种模式：

1. `global_monotonic`
2. `per_iter`
3. `legacy_rewind`

对于 ACE-G R2，auto 默认选 `per_iter`。原因是：

- fusion 在 S2 还会继续变化；
- 如果沿用一个已经接近最小 soft-clamp 的晚期全局时间轴，容易发散；
- 每轮独立调度更稳定。

---

## 10. 后续分析必须牢记的核心事实

后续讨论 LMC/ACE-G 时，必须默认以下事实：

1. compressor 只在 S1 训练，S2 永远不训练 compressor。
2. fusion 在 S1 一定训练，在 S2 只有 R2 才训练。
3. head 在 S1 和 S2 都训练，但 S2 前可能 reset。
4. global compressor 的 query 是纯几何 PE，不是 feature 初始化。
5. global compressor attention 的几何信息进入 logits。
6. fusion 的几何信息目前只进入 value，不进入 logits。
7. S1/S2 的监督不是 feature-level loss，而始终是 scene-coordinate reprojection loss。
8. 整个系统最终仍受老 ACE head 约束。
9. sampled rows 必须 pack 成 fake `1 x C x 16 x W` 才能过当前 head。
10. S1 是 compressor 的主训练阶段；S2-G 是 head/fusion 适配阶段。

---

## 11. 对后续 LMC 优化的直接含义

基于以上事实，后续优化 LMC 时应避免以下误解：

- 不能说 global compressor 完全没有几何；它已经有 learned distance bias。
- 不能把 S2 误认为还能训练 compressor；S2 中 compressor 是 no-grad cached output。
- 不能只看 fusion 结构而忽略 S1/S2 训练分工。
- 不能把 LMC 说成直接 coordinate predictor；它仍是 memory-conditioned feature adapter。
- 不能忽略 old ACE head 对输入 feature layout 的强约束。

当前最明确的 LMC 结构弱点包括：

1. fusion geometry 只进 value，不进 query-key matching logits。
2. compressor latent coords 来自随机起点 FPS，token 稳定性与可重复性需要关注。
3. compressor key 只用单层特征 slice，value 用多层拼接，key/value 语义不对称。
4. scale token 是全局 shift，不是 token-specific 或 view-specific 几何信息。
5. compressor/fusion 缺少 token-level auxiliary supervision，主要依赖最终 reprojection 间接监督。
6. memory 中保留的 ray / Plücker / visibility 等信息尚未真正进入主 LMC attention 机制。

这些点应作为后续 LMC 优化方案的基础前提。
