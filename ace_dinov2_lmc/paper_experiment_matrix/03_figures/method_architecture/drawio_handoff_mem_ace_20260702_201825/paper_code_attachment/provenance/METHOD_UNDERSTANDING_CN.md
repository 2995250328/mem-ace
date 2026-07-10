# Mem-ACE 方法图理解稿

日期：2026-07-02

本文档用于约束 draw.io 方法图。主图不罗列历史实验分支，也不展开 backbone-specific 实现差异，而是提炼 Mem-ACE 的共同方法事实。

## 核心故事

Mem-ACE 是一个用于增强 scene coordinate regression (SCR) 的场景记忆插件。它从带位姿的训练图像和 SfM/STGS 几何中构建 pooled scene memory，再将大规模 memory 压缩为少量带 3D anchor 的 scene tokens。查询图像的 dense features 通过 memory fusion 读取这些 tokens，随后由标准 SCR coordinate head 回归 dense scene coordinates。

## 共同流程

```text
Posed training images + SfM/STGS geometry
  -> Offline Memory Construction
  -> Pooled Scene Memory {p_i, f_i}, scene center c
  -> Geo-token Compressor
  -> Anchored Scene Tokens {(z_k, p_k)}

Query image
  -> SCR backbone / encoder
  -> Dense Query Features Q
  -> Memory Fusion with {(z_k, PE(p_k-c))}
  -> SCR Coordinate Head
  -> Dense Scene Coordinates X_hat(u)
```

## 训练口径

图中使用语义阶段名，而不是代码阶段名：

```text
Geometric Alignment:
  jointly align compressor, fusion, and coordinate head using geometric supervision

Cached Memory Adaptation:
  cache anchored tokens per scene, keep the compressor fixed, adapt fusion/head
```

稳定监督包括：

```text
single-view reprojection
cross-view track reprojection from SfM/STGS tracks
```

多帧/跨视角监督可以画入正文主图，但建议放在下方 training band，用虚线连接到 scene coordinates 或 coordinate head。

## Geo-token Compressor

输入：

```text
pooled_points:   B x N x 3
pooled_features: B x N x D
scene_center:    B x 3
```

输出：

```text
latent_z: B x K x d
latent_p: B x K x 3
```

`latent_p` 在论文图中可称为 `latent positions`、`3D anchors` 或 `anchor positions`。它们来自 memory points 的 anchor sampling，并在进入 position encoding 前减去 scene center：

```text
PE(latent_p - scene_center)
```

注意：`latent_p` 不是最终预测的 scene coordinate，也没有单独的 coordinate loss。最终坐标来自 SCR coordinate head。最新长训消融支持默认使用 `value_only_raw`：fusion 同时读取 `latent_z` 与 `PE(latent_p - scene_center)`；`z_only` 只作为去除几何通道的对照。

## Memory Fusion

推荐图示：

```text
Dense Query Features Q
  + latent features Z
  + centered geometry PE(P-c)
  -> Memory Fusion
  -> Memory-conditioned query features
```

主图只画 single fusion，不展开 cascade、reread、coord-prior 或其他实验分支。

## 与下游 SCR 系统的关系

部分实验会把 Mem-ACE 接入已有 SCR 系统来验证泛化性。正文方法主图不展开这些下游全局 refinement 的内部结构或分支特定 head 细节。

## 主图不要画

- 代码阶段名；
- 下游全局 refinement 或 branch-specific head 内部；
- DINO 层索引和 key/value 来源差异；
- scale token；
- local/hierarchical/learned compressor；
- cascade/reread/coord-prior fusion；
- 数据集分支和训练工程参数。
