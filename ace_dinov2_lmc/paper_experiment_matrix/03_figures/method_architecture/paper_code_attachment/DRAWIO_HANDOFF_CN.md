# Mem-ACE Draw.io 绘图交接入口

绘图 Codex 请按以下顺序阅读：

1. `../../figure_spec.md`：主图语义与硬约束。
2. `../../method_architecture.mmd`：节点和箭头的结构草图。
3. `configs/common_method.json`：共同方法事实。
4. `provenance/SOURCE_MAP.md`：需要核实时回到生产代码。

## 主图必须表达

```text
Posed training images + SfM/STGS geometry
  -> Offline Memory Construction
  -> Pooled Scene Memory {p_i, f_i}, scene center c
  -> Geo-token Compressor
  -> Anchored Scene Tokens (Z, P)

Query image -> dense query features
  + Anchored Scene Tokens (Z, P)
  + PE(P-c)
  -> Memory Fusion
  -> SCR Coordinate Head
  -> Dense Scene Coordinates
```

下方训练监督 band：

```text
Geometric Alignment:
  single-view reprojection
  cross-view track reprojection from SfM/STGS tracks

Cached Memory Adaptation:
  cache anchored scene tokens per scene
  freeze the compressor
  adapt fusion and coordinate head
```

可以在主干末尾放一个很小的 `PnP/RANSAC Pose Solver` 下游几何块，但视觉重点必须停留在 `Dense Scene Coordinates`。

## 模块小图必须表达

```text
Memory Points X + Memory Features F
  -> Geo-token Compressor
  -> latent features Z
  -> latent positions P

P + scene center c -> PE(P-c)
Query Features Q + Z + PE(P-c)
  -> Memory Fusion
  -> Fused Query Features
  -> Coordinate Head
```

关键事实：`P` 是 latent 3D anchor，会进入 fusion 的几何编码；`P` 不是最终 scene coordinate prediction。最新长训消融显示 `value_only_raw` 略强于 `z_only`，因此主图必须把 `Z + PE(P-c)` 画成默认 fusion 路径，而不是可选分支。

## 不得画入主图

- 代码阶段名或 launch-script 术语；
- 下游全局 refinement 内部结构或分支特定 head 细节；
- DINO 层索引、K/V 特征来源差异、scale token；
- local/hierarchical/learned compressor 分支；
- cascade/reread/coord-prior fusion 分支；
- 数据集分支或 benchmark 结果面板；
- buffer 大小、学习率、采样比例、GPU、run directory。

## 推荐 draw.io 版式

- 双面板横向图。
- 左侧大面板：Mem-ACE framework。
- 右侧小面板：Scene-anchored memory module。
- 主流程用实线箭头；训练监督和离线依赖用虚线箭头。
- query path 用蓝色，memory/token path 用绿色，coordinate/projection path 用橙色，training band 用灰色。
