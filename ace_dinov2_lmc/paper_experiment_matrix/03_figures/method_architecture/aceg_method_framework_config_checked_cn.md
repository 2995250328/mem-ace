# Mem-ACE 方法总体说明与取号一致性基线

日期：2026-07-10
状态：依据当前实现与 `launch_final_full_matrix_20260705.sh` 的最终训练协议整理。本文档是后续方法章节、主图、摘要和 OpenReview 取号字段的共同事实源；它不是实验结果表，也不替代最终的消融与统计。

## 0. 一句话主线

**Mem-ACE 将由三维点和对齐视觉特征构成的大型场景记忆压缩为少量带三维锚点的 scene tokens，并以场景中心对齐的几何编码将这些 tokens 注入现有 SCR 的稠密特征，从而在不替换坐标回归头的前提下提供可缓存、可复用的场景表征。**

这里的核心不是另建一个 pose solver，也不是让 latent anchor 直接预测坐标。Mem-ACE 是作用于 scene coordinate regression (SCR) 的 scene-memory interface：输入为场景记忆和 query image feature，输出仍然是标准 SCR head 的稠密场景坐标。

## 1. 论文范围、名称与术语

### 1.1 应写入论文主方法的稳定范围

1. 从带位姿的 mapping images、相机内外参和 SfM/COLMAP reconstruction 构建 pooled scene memory。
2. 用 **Geo-token Compressor**（代码名 `GeoLMC`）把 memory 压缩为固定数量的 anchored scene tokens。
3. 用一次 **Memory Fusion** 将 token content 与 scene-centered anchor geometry 注入 query dense features。
4. 用原有或兼容的 SCR coordinate head 回归稠密 scene coordinates。
5. 用 source-frame reprojection 训练；在有可靠 track sidecar 的最终适配路径中，额外加入 cross-view track reprojection。
6. 采用两个语义训练阶段：**Geometric Alignment** 和 **Cached Memory Adaptation**。

### 1.2 术语对照

| 论文名称 | 代码/旧称 | 含义与写作边界 |
|---|---|---|
| Mem-ACE | ACE-DINOv2-LMC / ACE-G LMC flow | 方法名；不要把 DINO、ACE-FCN 或 GLACE 写入方法定义。 |
| Geo-token Compressor | `GeoLMC` | 从点-特征 memory 中产生 compact anchored tokens 的压缩器。 |
| Anchored scene token | `(latent_z, latent_p)` | `z_k` 是 token content，`p_k` 是从 memory points 采样的 latent 3D anchor。 |
| Memory Fusion | `LMCFusionBlock` / single fusion | query features 对 tokens 进行一次 cross-attention read。不要称为 bridge。 |
| Geometric Alignment | S1 | 联合优化 compressor、fusion 与 coordinate head 的阶段。 |
| Cached Memory Adaptation | S2-G | 缓存 token、冻结 compressor、适配 fusion/head 的阶段。不要在正文写 `S2-G`。 |
| Track-guided supervision | STGS sidecar | COLMAP track 提供的采样和跨视角几何约束；正文先用完整语义名，代码缩写只在实现细节中出现。 |

### 1.3 不属于主方法定义的内容

- DINOv2、MapAnything、ACE-FCN 是可替换的 feature/memory source，不是 Mem-ACE 的必要定义。
- GLACE concat head 是兼容性和泛化性验证路径，不是 Mem-ACE 的内部模块。
- scale token、selected key layer、key/value feature-source asymmetry、cascade/reread/dual fusion、coordinate-prior 分支、local/hierarchical/learned 的历史分支都不进入主方法图和主摘要。
- 开发期曾存在 global-to-local 自动回退 guard，但 paper-facing protocol 已固定为 global compression。该机制不会出现在论文标题、摘要、引言、方法、实验、主图或附录中，也不构成贡献或消融项。
- DSAC*/PnP/RANSAC 是标准的下游 pose recovery；网络方法的直接输出是 dense scene coordinates。

## 2. 问题表述与统一坐标参考

给定 query image $I$，SCR backbone 产生稠密特征 $Q=B(I)$。标准 coordinate head 从每个图像位置 $u$ 预测 scene coordinate $\hat{X}(u)$，再由几何求解器从 2D--3D correspondences 恢复相机位姿。

Mem-ACE 为一个场景建立 memory bank

\[
\mathcal{M}=\{(p_i,f_i)\}_{i=1}^{N}, \qquad p_i\in\mathbb{R}^{3},\; f_i\in\mathbb{R}^{D},
\]

其中 $p_i$ 是与视觉 feature $f_i$ 对齐的三维点。memory 同时保存 scene center $c$。在标准 pooled-memory contract 中，$c$ 是由相机中心得到的场景参考；在 reference-normalized contract 中，它是同一预测坐标空间中的参考点。无论具体 contract 如何，以下两处使用同一个 $c$：

1. anchor geometry 使用 $p_k-c$ 进行编码；
2. coordinate head reset 后其输出均值与 $c$ 对齐。

因此应写“Mem-ACE 将 token geometry 和 coordinate-regression parameterization 对齐到同一 scene reference”，而不是笼统地说“网络直接回归相对坐标”。实现上 head 的输出均值被设为 `scene_center`；最终输出仍是该坐标 contract 中的完整 scene coordinate。

## 3. 离线场景记忆与 track sidecar

### 3.1 Pooled scene memory

离线前处理的输入是 mapping RGB images、camera poses/intrinsics，以及 COLMAP sparse point cloud 和 observations。它把可投影的三维支持与 backbone-specific visual features 对齐并池化：

```text
posed mapping images + SfM reconstruction
  -> sparse depth / patch-level 3D support
  -> feature extraction and 3D-feature association
  -> pooled scene memory M = {pooled_points, pooled_features, scene_center}
```

必须保留的 memory 字段为：

```text
pooled_points:   N x 3   scene-space 3D points
pooled_features: N x D   visual features aligned with the points
scene_center:    3       coordinate reference shared with the SCR head
```

`N` 随场景规模变化，且通常远大于最终 token 数。memory construction 是离线过程；后续 Cached Memory Adaptation 不重新构建 COLMAP 或 memory bank。

### 3.2 Track sidecar

同一 SfM reconstruction 的 tracks 生成一个训练 sidecar，而不是新的网络模块。每一条可用关系至少包含 source observation $u_s$、target observation $u_t$、对应相机、共享 track/point identifier，以及用于筛选的 reprojection/alignment metadata：

```text
COLMAP tracks
  -> source-target observation pairs
  -> track-guided sampling and cross-view reprojection supervision
```

主文可以说“geometrically verified SfM tracks”；不需要在方法图里暴露 sidecar 的文件字段或采样实现。

## 4. Geo-token Compressor：紧凑的 anchored scene representation

### 4.1 压缩接口

Geo-token Compressor 将 $N$ 个 point-feature pairs 映射为 $K$ 个 anchored tokens：

\[
\mathcal{C}_\theta(\mathcal{M},c)
\;\longrightarrow\;
\mathcal{T}=\{(z_k,p_k)\}_{k=1}^{K},
\qquad z_k\in\mathbb{R}^{d},\;p_k\in\mathbb{R}^{3}.
\]

当前最终协议使用 $K=64$ 与 4 个 compression attention layers。anchors 由 memory points 的 farthest-point sampling 选出，初始点策略为 `farthest_from_center`。每个 anchor 先以相对位置 $p_k-c$ 初始化几何 query，再通过几何感知 attention 从 $\{(p_i,f_i)\}$ 聚合内容，得到 $z_k$。

这给出两项稳定事实：

- `latent_p` / $p_k$ 是从 scene memory 采样并保留的三维锚点，不是 coordinate head 预测的 latent coordinate；
- `latent_z` / $z_k$ 是锚点处聚合得到的 content token，而不是原始点云的简单 subsampling feature。

### 4.2 压缩尺寸的正确表述

论文应先报告固定 token count：**Mem-ACE 在最终协议中以 64 个 anchored tokens 表示一个场景。** token channel dimension $d$ 由 memory feature contract 决定，而非在所有分支中固定为 1024：

- ACE-FCN memory 的实现要求每个 pooled feature 为 512 维，因此其 token content 是 $64\times512$；anchor 坐标另为 $64\times3$。
- DINO/MapAnything memory 的 $d$ 由实际 memory 的 layer contract 解析。只有经对应 run metadata 确认为 1024 维时，才可将该分支写为 $64\times1024$。

因此，正文和图中通用标签写成 `$64\times d$ anchored scene tokens`。效率表再逐分支报告实际数字、显存、训练时间和 inference latency。不要把 `$64\times1024$` 写成所有 Mem-ACE runs 的普遍事实，也不要在没有逐条核验 ACE-G 论文和所复现配置前写出 `$4096\times768$` 这类对比数字。

可安全主张的计算优势是：query-time memory read 的 token 轴从场景点数 $N$ 缩减为固定 $K=64$，因此 downstream fusion 的 memory-read cost 随 $K$ 而非 $N$ 扩展。具体速度或显存优势必须由最终 efficiency table 支持。

### 4.3 `latent_p` 的解释边界

`p_k` 的存在支持“token 的场景分布不是退化为围绕 scene center 的单一全局向量”的分析假设，但它本身不能证明最终预测点云一定多峰、覆盖更广或误差更低。这些结论必须由最终 coordinate prediction 的点云分布、coverage/density-bin 误差和跨视角一致性统计来验证。

因此正文可写：

> Each scene token retains a sampled 3D anchor, allowing the compact representation to preserve spatial organization while remaining fixed in size.

但不能写：

> The anchors are supervised coordinate predictions / The anchors alone cause the improvement.

## 5. Memory Fusion：content 与 centered geometry 的单次读取

对于 query feature $q_u$，最终默认的 fusion 使用 `value_only_raw`。令 $\phi(\cdot)$ 为 Fourier positional encoder，$P_\mathrm{geo}$ 为其线性投影，则 token 的 attention inputs 为：

\[
g_k=P_\mathrm{geo}(\phi(p_k-c)),\qquad
k_k=W_K z_k,\qquad
v_k=W_V(z_k+g_k).
\]

query 从 $\{k_k,v_k\}_{k=1}^{K}$ 做一次 cross-attention read，随后经过 residual/FFN 得到 memory-conditioned feature $\tilde q_u$：

\[
\tilde q_u = F_\varphi(q_u,\mathcal{T},c),
\qquad
\hat X(u)=H_\psi(\tilde q_u).
\]

这一定义有三个需要保持准确的点：

1. `value_only_raw` 中，raw centered anchor 的 Fourier encoding **只注入 value**；attention key 仅由 $z_k$ 构成。它不是把几何同时加到 key 的 `geokey_norm` 变体。
2. `z_only` 是去除 anchor geometry 的对照：key 和 value 都仅由 $z_k$ 构成。长训消融中 `value_only_raw` 略优于 `z_only`，因此前者是默认选择；该结果只支持把它作为默认设计，不能单独归因于 $p_k$。
3. $p_k$ 只经 $g_k$ 参与 Memory Fusion，不能从 $p_k$ 画一条箭头到 coordinate head，也不存在独立的 $p_k$ coordinate loss。

主图的最小正确数据流是：

```text
Query dense features Q
  + 64 anchored scene tokens {(z_k, p_k)}
  + centered anchor encoding PE(p_k - c)
  -> single Memory Fusion
  -> memory-conditioned dense features
  -> SCR coordinate head
  -> dense scene coordinates X_hat(u)
```

## 6. 训练：Geometric Alignment 与 Cached Memory Adaptation

代码的 `lmc_iterations=10` 不是两个只执行一次的大阶段，而是在每个 iteration 中依次执行下列两个语义步骤。论文可以用这两个名称组织叙述，避免暴露 `S1`、`S2-G` 等代码标签。

### 6.1 Geometric Alignment

该步骤联合更新 Geo-token Compressor、Memory Fusion 和 SCR head，使压缩 tokens 能服务于坐标回归：

```text
M -> C_theta -> {(z_k, p_k)}
Q -> F_phi(Q, {(z_k, p_k)}, c) -> H_psi -> X_hat
update C_theta, F_phi, H_psi
```

主要几何监督是 source-frame reprojection。对于有 track sidecar 的 batch，采样过程可以提高可用跨视角关系的占比；但在最终 full-matrix protocol 中，显式 cross-view reprojection loss 的 active stage 是 Cached Memory Adaptation，不能写成所有 Geometric Alignment steps 都带该 loss。

### 6.2 Cached Memory Adaptation

在这一语义步骤中，先运行一次 compressor 并缓存 scene tokens：

```text
(Z, P) = C_theta(M)          # cache once for the current adaptation step
freeze C_theta
Q -> F_phi(Q, (Z, P), c) -> H_psi -> X_hat
adapt F_phi and H_psi
```

当前 trainable-fusion protocol 启用 `ace_g_fusion_in_s2=True`：compressor 冻结，fusion 与 head 继续适配；fusion 的学习率相对 head 使用较小比率（0.01）。这使 cache 是稳定的 scene representation，而不是让 compressor 在同一适配步骤内继续漂移。

GLACE compatibility run 属于下游实例：它加载/冻结相应的 local stack，并在 GLACE 原始 concat formulation 中验证 Mem-ACE representation 的可接入性。它不改变上面的 Mem-ACE token interface，也不应在方法图中展开为所谓的 bridge。

## 7. 几何监督

### 7.1 Source-frame reprojection

对 source image 的采样位置 $u_s$，从网络预测 $\hat X_s=\hat X(u_s)$，并投影回 source camera：

\[
\mathcal{L}_{\mathrm{src}}
= \rho\bigl(\pi_s(\hat X_s)-u_s\bigr),
\]

其中 $\pi_s$ 使用已知 source intrinsics/extrinsics，$\rho$ 表示实现中采用的鲁棒 reprojection contract。它是所有相关路径的基础监督。

### 7.2 Cross-view track reprojection

对同一 COLMAP track 的 source-target observations $(u_s,u_t)$，把 source prediction 投影到 target camera：

\[
\mathcal{L}_{\mathrm{track}}
= \rho\bigl(\pi_t(\hat X(u_s))-u_t\bigr).
\]

最终 ACE-FCN/GLACE full-matrix protocol 在 Cached Memory Adaptation 中启用该项：$\lambda_{\mathrm{track}}=0.05$，前 20\% 适配进度 warm up，最后 30\% 衰减，并使用 10\% track-guided samples。于是可写为：

\[
\mathcal{L}=\mathcal{L}_{\mathrm{src}}+
\lambda_{\mathrm{track}}\mathcal{L}_{\mathrm{track}}.
\]

必须保留的限定是：DINO+MapAnything 的 final full-matrix branch 仅运行 Stage 1；其 launch config 虽携带 track-loss 参数，但 `apply_to=stage2_g`，因此该显式 track loss 不会在该 Stage-1-only branch 触发。论文表格和摘要不能把跨视角 loss 说成所有分支、所有阶段均已启用。

## 8. 固定 Global Protocol 与论文边界

所有 paper-facing experiments 使用固定的 global Geo-token Compressor：每个 anchor 通过几何感知 attention 从完整 pooled scene memory 聚合信息。最终 launch script 显式关闭自动可见性回退，并且 global `scene5` 复跑的结果已满足纳入该统一协议的要求。

开发期的 global-to-local 自动回退只保留在代码历史中，不属于 Mem-ACE 方法、训练策略、分析对象或复现实验变量。它不会出现在论文正文、附录、图注、主图、消融表或 OpenReview 摘要中。后续写作直接使用以下固定表述：

> We use a global Geo-token Compressor that aggregates the full pooled point-feature memory into a fixed set of anchored scene tokens.

## 9. 最终实验实现的共享关系

所有主实验使用相同的高层接口：

```text
backbone-specific feature memory
  -> Geo-token Compressor
  -> 64 anchored scene tokens
  -> single Memory Fusion
  -> compatible SCR coordinate head
```

当前 final matrix 的实现覆盖：

| 实验路径 | 作用 | 不应误写成 |
|---|---|---|
| Indoor6, DINOv2 + MapAnything memory, Stage 1 | 验证 memory interface 在 foundation-feature memory 上的可行性 | Mem-ACE 的唯一 backbone 或完整 two-stage track-loss 结果。 |
| Indoor6 / Wayspots / Cambridge, ACE-FCN memory, Stage 1 + GLACE concat Stage 2 | 主比较和 GLACE-compatible downstream validation | GLACE 内部结构就是 Mem-ACE，或 GLACE 是方法核心模块。 |
| SfM track sidecar | 可用时提供可靠采样和跨视角 target | 一个新增的 inference network component。 |

因此正文在第一次介绍 Mem-ACE 时只说 `backbone-specific features` 或 `pooled point-feature memory`；分支名称放到实验设置和实现细节中。

## 10. 方法图与正文写作约束

### 10.1 主图应显示

```text
Offline: posed images + SfM cloud/tracks -> pooled memory + track sidecar

Online/training: pooled memory -> Geo-token Compressor -> {(z_k, p_k)}
                  query image -> dense features -> Memory Fusion -> SCR head -> X_hat

Supervision: source reprojection; source-to-target track reprojection in adaptation
```

将 $p_k-c$ 画为进入 `PE` 再进入 Memory Fusion 的分支即可。主图终点是 `Dense Scene Coordinates`，相机视锥和点云只用于表达重投影几何。

### 10.2 主图和正文不能显示/宣称

- `$p_k \rightarrow$ coordinate head`、latent-coordinate prediction 或 anchor supervision；
- global-to-local fallback、visibility score、routing decision 或相关的开发期实验；正文只画并描述固定 global compressor；
- `S1`、`S2-G` 或 “bridge”；
- 任一 backbone 的 layer index、scale token 或 key/value source asymmetry；
- GLACE global descriptor/concat head 的内部细节；
- 未经结果支持的 “more multimodal predictions”“better sparse-region accuracy”“stronger multi-view consistency”。这些应作为待完成的分析图和统计结论，而非方法事实。

## 11. 可直接转写到论文的方法概述

### English

> Mem-ACE augments scene coordinate regression with a compact, reusable scene-memory interface. From posed mapping images and an SfM reconstruction, it builds a pooled memory bank of 3D points and aligned visual features. A Geo-token Compressor selects a fixed set of spatial anchors and aggregates the point-feature memory into anchored scene tokens. Each token pairs learned content with a 3D anchor. For a query image, a single Memory Fusion block conditions dense features on token content and on anchor geometry encoded relative to the scene center; a standard SCR head then predicts dense scene coordinates in the same coordinate reference. Training first aligns the compressor, fusion module, and coordinate head, and then caches the compressed scene tokens while adapting the downstream predictor. When verified SfM tracks are available, they additionally provide cross-view reprojection supervision during cached adaptation.

### 中文

> Mem-ACE 为 scene coordinate regression 增加了一种紧凑且可复用的场景记忆接口。方法从带位姿的 mapping images 和 SfM 重建中构建由三维点及其对齐视觉特征组成的 pooled memory bank。Geo-token Compressor 选择固定数量的空间 anchors，并聚合 point-feature memory 得到 anchored scene tokens；每个 token 同时包含学习到的内容和一个三维 anchor。对于 query image，单次 Memory Fusion 将 token content 以及相对于 scene center 编码的 anchor geometry 注入稠密特征，随后由标准 SCR head 在同一坐标参考下预测稠密 scene coordinates。训练先对 compressor、fusion 和 coordinate head 做几何对齐，随后缓存压缩后的 scene tokens 并适配下游预测器。当存在经过验证的 SfM tracks 时，它们还会在缓存适配阶段提供跨视角重投影监督。

## 12. 取号内容的对照规则

推荐取号标题：

```text
Mem-ACE: Compact Scene Memory for Accelerated Coordinate Encoding
```

该标题保留 ACE 的全称，突出压缩 scene memory，但不承诺任何尚未由最终结果验证的绝对性能结论。

取号 TL;DR 应只承诺方法事实：

```text
Mem-ACE equips Accelerated Coordinate Encoding with a compact set of geometry-anchored scene tokens that can be reused across query images for scene coordinate regression.
```

取号摘要和最终摘要可以使用第 11 节的概述，但必须遵守如下边界：

- 可以写：pooled point-feature memory、fixed-size anchored tokens、scene-centered geometry、single fusion、standard SCR head、cached adaptation、可用时的 SfM-track supervision。
- 不应写：DINOv2/MapAnything/ACE-FCN/GLACE 的具体内部连接、`S1/S2-G`、bridge、统一的 `$64\times1024$`、latent anchors 是预测坐标、任何开发期 visibility fallback/routing、全部分支都用 cross-view loss、普遍或 SOTA 改进。
- 不必在取号版本中列数据集和数字；最终 results-ready abstract 再加入一条由最终聚合表支持的结果句。

详细的 copy-ready Title/TL;DR/Abstract 与表单核对项维护在：
[`paper_draft/aaai2027_openreview_registration.md`](../../../paper_draft/aaai2027_openreview_registration.md)。

## 13. 代码事实索引

后续改论文结论前，优先复核以下实现位置：

- `GeoLMC` 的 FPS anchors、centered seed encoding 与几何 attention：[`ace_compressor.py`](../../../../ace_compressor.py:1230)。
- `value_only_raw` 的 key/value 定义：[`ace_fusion.py`](../../../../ace_fusion.py:148)。
- coordinate head 的 scene-center mean alignment：[`trainer_dinov2_lmc.py`](../../../trainer_dinov2_lmc.py:6757)。
- 每个 iteration 的 Geometric Alignment 与 cached adaptation 执行顺序：[`trainer_dinov2_lmc.py`](../../../trainer_dinov2_lmc.py:9897)。
- final matrix 的 fixed-global token count、loss schedule、dataset paths 和 branch coverage：[`scripts/launch_final_full_matrix_20260705.sh`](../../../scripts/launch_final_full_matrix_20260705.sh:1)。

本说明不添加文献引用。论文中的 ACE、ACE-G、GLACE、NeuMap 等引用仍必须以已核验的官方论文元数据和 BibTeX 为准，不能从本文档反向生成或猜测引用。
