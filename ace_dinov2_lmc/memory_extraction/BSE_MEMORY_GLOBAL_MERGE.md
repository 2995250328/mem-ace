# BSE Memory 效果分析与全局体素合并方案

本文档合并了两部分内容：**BSE Memory 与 Pooled pipeline 的差异与根因**，以及 **Pass 2 后全局 voxel merge 的具体改法**（方案 A）。便于评审与落地实现。

---

## 1. 核心问题：逐 view 独立 pooling vs 全局合并 pooling

### Pooled pipeline（map-anything，约 76.26%）

- 所有 view 的点 → `torch.cat` 合并 → SOR 全局去噪 → `voxel_pooling_optimized` **一次性全局 pool**。
- 同一个 0.05 m 体素内，来自不同 view 的点被平均成一个点，特征也被平均。

### BSE pipeline（当前实现，约 71.60% / 66.54%）

- **每个 view 独立**：unproject → BSE pool（voxel hash + Otsu split）→ 保存 chunk。
- **Pass 2**：`torch.cat` 所有 chunk → Welford 归一化 → 组装结果。
- **没有跨 view 合并**。同一 3D 位置被多个 view 看到时，会产生多个独立的 pooled 点。

---

## 2. 带来的问题

### 2.1 点云冗余膨胀

- 每个 view 最多约 37×37 = 1369 个 grid 点，BSE pool 后大约每 view 数百点；40 个 view 可达约 8k–32k 点。
- Pooled 版本全局 pool 后通常只有约 2k–5k 点。
- GeoLMC compressor 用 FPS 采 K=64 个 latent：从 **3 万级冗余点** 里采 64 个，与从 **3k 级去重点** 里采 64 个，空间覆盖率差异很大。

### 2.2 同一位置特征不一致

- 同一 3D 位置在不同 view 下 DINOv2 特征有差异（视角、光照等）。
- Pooled 版本在体素内做了平均，表示更稳定；BSE 保留多条「近似同位置」样本，对 cross-attention 更易表现为噪声。

### 2.3 Otsu 二分裂加剧碎片化

- `sub_ids = cluster_ids * 2 + is_outlier` 把每个体素再分成 inlier/outlier 子簇。
- 在已「逐 view 独立」的前提下再分裂，碎片化更严重；relocalization 里边界处的 outlier 往往仍是有效几何信息。

---

## 3. 归一化：非主因但有影响

- **BSE normalized（约 71.60%）**：坐标 `(p - μ) / σ`，`scene_center ≈ 0`；trainer 侧 depth threshold 缩放数学上可行，但 reprojection loss 在归一化坐标系下的梯度尺度与世界坐标系不同。
- **BSE denorm（约 66.54%）**：还原世界坐标，但 μ 来自点云质心而非相机中心均值，与 pooled 的 `scene_center` 定义存在细微差异；且 **denorm 仍不解决逐 view 独立 pool 的冗余**。

---

## 4. 修改方案概览

| 方案 | 思路 | 改动面 |
|------|------|--------|
| **A（推荐先做）** | Pass 2 在 `torch.cat` 之后、`result` 组装之前，增加一步 **全局 voxel merge**（与 map-anything 体素均值池化同类逻辑） | 主要改 `run_memory_extraction.py`，约数十行；可选 `extract_memory.sh` 环境变量 |
| **B** | 在 BSE 框架内将 `pooler.pool()` 换为简单全局 voxel mean（去掉 Otsu split），贴近 map-anything | `bse_pooling.py` + `run_memory_extraction.py` CLI + shell |
| **C** | 两阶段：保留逐 view BSE 降采样 → 全局 merge → Otsu 可选 | 方案 A + `--use_otsu` 等开关组合 |

**建议优先做方案 A**：改动最小、只增不改核心 BSE 逻辑，便于验证「全局 merge 是否为关键因子」；若追上 pooled，根因基本坐实；若未追上，再排查 Otsu 等因素。

---

## 5. 方案 A 实施细节（`--global_merge`）

### 5.1 目标

在 `two_pass_processing()` 的 **Pass 2 拼接完成之后、写入 `result` 字典之前**，对 `final_points` / `final_features` / `final_colors` / **射线与 cluster 相关张量** 做一次与体素对齐的合并，消除跨 view 重叠。通过 **`--global_merge`** 控制，**默认开启**。

> 说明：当前仓库里 Pass 2 在 `run_memory_extraction.py` 中除 `points/features/colors/cluster_sizes` 外，还拼接了 `ray_dirs`、`ray_dirs_mean` 等；实现时需与现有字段一致，在 merge 后同步更新 `ray_dirs_mean`（或与 dominant/first/plücker 的约定一致），并在 merge 后清空失去语义的多 view 派生射线列表（若存在）。

### 5.2 新增函数 `global_voxel_merge`（示意）

放在 `two_pass_processing()` 定义之前即可。逻辑：体素量化 → 哈希 → `torch.unique` 得 inverse → 对点、特征、颜色、射线方向做 **scatter mean**，对 `cluster_sizes` 做 **scatter sum**，射线方向再 L2 归一化。

```python
def global_voxel_merge(points, features, colors, ray_dirs, cluster_sizes, voxel_size=0.05):
    """全局体素合并：将跨 view 落在同一体素内的样本合并为单个点。"""
    quantized = torch.floor(points / voxel_size).long()
    hash_vals = (
        quantized[:, 0] * 73856093
        + quantized[:, 1] * 19349663
        + quantized[:, 2] * 83492791
    )
    _, inverse = torch.unique(hash_vals, return_inverse=True)
    n_voxels = int(inverse.max().item()) + 1

    def _scatter_mean(src, idx, n):
        out = torch.zeros(n, src.shape[1], device=src.device, dtype=src.dtype)
        out.index_add_(0, idx, src)
        cnt = torch.zeros(n, 1, device=src.device, dtype=src.dtype)
        cnt.index_add_(0, idx, torch.ones(len(idx), 1, device=src.device, dtype=src.dtype))
        return out / cnt.clamp(min=1)

    def _scatter_sum_1d(src, idx, n):
        out = torch.zeros(n, device=src.device, dtype=src.dtype)
        out.index_add_(0, idx, src)
        return out

    m_points = _scatter_mean(points, inverse, n_voxels)
    m_features = _scatter_mean(features.float(), inverse, n_voxels).to(dtype=features.dtype)
    m_colors = _scatter_mean(colors, inverse, n_voxels)
    m_ray_dirs = _scatter_mean(ray_dirs, inverse, n_voxels)
    m_ray_dirs = torch.nn.functional.normalize(m_ray_dirs, dim=1, eps=1e-6)
    m_sizes = _scatter_sum_1d(cluster_sizes.float(), inverse, n_voxels).long()
    return m_points, m_features, m_colors, m_ray_dirs, m_sizes
```

### 5.3 在 Pass 2 末尾插入调用（示意）

在 `final_cluster_sizes = torch.cat(...)` 之后、构建 `result = { ... }` 之前：

```python
if config.global_merge:
    n_before = final_points.shape[0]
    final_points, final_features, final_colors, final_ray_dirs, final_cluster_sizes = (
        global_voxel_merge(
            final_points,
            final_features,
            final_colors,
            final_ray_dirs,  # 或与 ray_dirs_mean 对齐，按当前 pipeline 实际张量选择
            final_cluster_sizes,
            voxel_size=config.voxel_size,
        )
    )
    final_ray_dirs_mean = final_ray_dirs  # 若 mean 与 primary ray 共用语义
    all_ray_dirs_dominant = []
    all_ray_dirs_first = []
    all_plucker_rays = []
    n_after = final_points.shape[0]
    print(
        f"[Global Merge] {n_before} -> {n_after} points "
        f"(removed {n_before - n_after} cross-view duplicates)"
    )
```

实现时需对照当前 `two_pass_processing()` 内 **`final_ray_dirs` / `final_ray_dirs_mean` / `result` 字段** 逐项一致，避免只 merge 了部分张量导致 shape 或语义不一致。

### 5.4 配置与 CLI

- 在 **`ExtractionConfig`**（`use_otsu` 附近）增加：`global_merge: bool = True`。
- 在 **`parse_args()`** 增加例如：

```python
parser.add_argument(
    "--global_merge",
    type=lambda x: str(x).lower() in ("true", "1", "yes"),
    default=True,
    help="Pass 2 后全局体素合并，消除跨 view 重叠点（默认 True）。",
)
```

- 构造 `ExtractionConfig(...)` 时传入 `global_merge=args.global_merge`。

### 5.5 `extract_memory.sh`（可选但推荐）

- 环境变量：`GLOBAL_MERGE="${GLOBAL_MERGE:-true}"`。
- 构建 `PYTHON_ARGS` 时追加 `--global_merge true/false`。
- 在输出目录的 `config_tag` 与 `extraction_config.json` 中记录该选项，便于复现实验。

### 5.6 明确不修改的文件（若仅做方案 A）

- `bse_pooling.py`、`utils_lmc.py`、`trainer_dinov2_lmc.py`、`options_dinov2_lmc.py` 可保持不动（以方案 A 为界）。

---

## 6. 验证步骤

1. **重新提取 memory**（示例，按你的数据与 loader 调整）：

   ```bash
   SCENE_TRAIN=scene2a_train DATASET_TYPE=indoor6 N_VIEWS=40 \
     bash ace_depth/ace_dinov2_lmc/memory_extraction/extract_memory.sh
   ```

2. **检查点数**：合并后期望从 1 万+ 降至约 2k–5k 量级（与场景与 voxel 有关）。

   ```bash
   python -c "import torch; m=torch.load('path/to/memory_bse.pt'); print(m['points'].shape[0])"
   ```

3. **训练对比**：关注 acc5 / pct5 等是否与 pooled memory（约 76.26%）拉近。

---

## 7. 参考锚点（仓库内）

- `two_pass_processing()`：`run_memory_extraction.py` 中 Pass 2 拼接约在 `final_points = torch.cat(all_points, ...)` 一段（行号随版本变动，以文件为准）。
- `ExtractionConfig`：`run_memory_extraction.py` 内 `class ExtractionConfig` 定义处。

---

*文档来源：会话内技术笔记整理；实现时请以当前 `run_memory_extraction.py` 实际张量与字典字段为准做对齐。*
