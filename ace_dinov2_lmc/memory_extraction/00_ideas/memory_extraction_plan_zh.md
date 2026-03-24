# 计划：BSE Memory Extraction 迁移与实现

## 背景

设计文档：`memory.txt`（同目录）
待迁移源码：`map-anything/mapanything/tasks/run_memory_extraction.py`
目标位置：`ace_dinov2_lmc/memory_extraction/`（新子目录，不修改现有文件）

目标：将内存提取从基础体素池化（Paradigm A）升级为
带全局度量归一化的双边超体素提取（Bilateral Supervoxel Extraction, BSE，Paradigm C）。

---

## 目标文件结构

```
ace_dinov2_lmc/
├── 00_ideas/
│   ├── memory.txt                        ← 设计文档（不变）
│   └── memory_extraction_plan.md         ← 本文件英文版
├── trainer_dinov2_lmc.py                 ← 现有文件（不变）
├── options_dinov2_lmc.py                 ← 现有文件（不变）
└── memory_extraction/                    ← 新增
    ├── __init__.py
    ├── bse_pooling.py                    ← BSE 算法（独立）
    ├── run_memory_extraction.py          ← 迁移并增强
    └── extract_memory.sh                 ← 适配后的 bash 脚本
```

---

## 关键文件（实现前先阅读）

| 文件 | 作用 |
|------|------|
| `map-anything/mapanything/tasks/run_memory_extraction.py` | 待迁移源文件（只读） |
| `map-anything/bash_scripts/ace/fps_memory.sh` | 需要适配的 bash 脚本 |
| `ace_dinov2_lmc/trainer_dinov2_lmc.py` | 可复用的 DINOv2 加载模式 |
| `ace_depth/dataset_dinov2.py` | 可复用的数据集加载器 |

---

## 阶段 1 — 迁移 + 最小增量改造

**`memory_extraction/run_memory_extraction.py`**

替换 mapanything 依赖：
- `mapanything.models.init_model` → 直接加载 DINOv2（复用 `trainer_dinov2_lmc.py` 模式）
- `mapanything.datasets.SevenScenesWAI / Indoor6WAI` → 使用 `dataset.py` / `dataset_dinov2.py`
- `@hydra.main` → 普通 `argparse`

保留所有辅助函数：`save_pcd_with_open3d`、`clean_point_cloud_sor`、
`voxel_pooling_optimized`、`_save_depth_on_rgb_vis`

新增内容：
- 计算 `O_cam = t`、`D_raw = normalize(P_raw - O_cam)`
- 体素池化后计算 `mu_scene`、`sigma_scene`，并归一化 `P_bse`
- 保存扩展 `.pt` 格式（见下方输出格式）
- 增加 `--output_dir` 参数（默认：`ace_dinov2_lmc/04_evaluation/`）

**`memory_extraction/extract_memory.sh`**
- 基于 `fps_memory.sh` 适配，去除 Hydra，改为 argparse CLI

---

## 阶段 2 — 向量化几何哈希

**`memory_extraction/bse_pooling.py`**

```python
def coarse_voxel_hash(points: Tensor, voxel_size: float) -> Tensor:
    """通过 torch.unique 在量化坐标上返回 cluster_ids。"""
    quantized = torch.floor(points / voxel_size).long()
    _, cluster_ids = torch.unique(quantized, dim=0, return_inverse=True)
    return cluster_ids  # [N]
```

将 `voxel_pooling_optimized` 中分块 `index_add_` 循环替换为基于 `cluster_ids`
的单次向量化实现。

注意：使用 `index_add_` + 计数除法（不要用 `torch_scatter`）——该方式已在
现有 `voxel_pooling_optimized` 中验证，可避免新增依赖。

---

## 阶段 3 — 双边特征聚类（BSE 核心）

**`memory_extraction/bse_pooling.py`** — 新增 `bse_pooling()`

```python
def bse_pooling(points, ray_dirs, features, colors,
                voxel_size=0.05, tau=0.90, chunk_size=500_000) -> dict:
    """
    1. coarse_voxel_hash → cluster_ids
    2. 通过 index_add_ / count 计算簇均值特征
    3. 计算每个点与其簇均值特征的余弦相似度（fp32）
    4. is_outlier = sim < tau → sub_cluster_id = cluster_id*2 + is_outlier
    5. 在 sub_cluster_ids 上做细粒度 scatter_mean
    6. 池化后对 ray_dirs 做 L2 归一化
    """
```

**在 `run_memory_extraction.py` 中集成：**
- `--use_bse` 开关（默认：False，保证向后兼容）
- `--bse_tau`（默认：0.90）
- `--bse_voxel_size`（默认：0.05）
- 自适应回退：若输出 > 100k 点，则将 voxel_size 加倍后重试
- 余弦相似度前强制转 fp32，分母加 ε=1e-6

---

## 输出格式（`.pt` schema）

```python
{
    "points":   P_norm,      # [N, 3] float32，归一化（零均值、单位标准差）
    "ray_dirs": D_bse,       # [N, 3] float32，单位向量
    "features": F_bse,       # [N, C] float16，DINOv2 特征
    "colors":   C_bse,       # [N, 3] float32，RGB [0,1]
    "mu":       mu_scene,    # [3]    float32，场景均值（用于反归一化）
    "sigma":    sigma_scene, # 标量 float32，场景标准差
}
```

向后兼容：如有下游代码需求，可同时提供 `points`/`pooled_points` 与
`features`/`pooled_features` 的键别名。

---

## 验证清单

- [ ] 阶段 1 冒烟测试：在 7-Scenes Chess 上运行，验证 `P_norm` 落在 `[-3, 3]`
- [ ] 阶段 3 压缩率：`len(P_bse) / total_input_pixels < 0.05`
- [ ] 边界保持性：使用 Open3D 可视化检查
- [ ] 回归验证：将新 `.pt` 输入 `trainer_dinov2_lmc.py`，确认 loss 可收敛
- [ ] 消融实验：在 Indoor6 scene3/scene4a 上比较 `--use_bse False` 与 `--use_bse True`

---

## 设计备注

- v1 中采用二分（主类/离群）可接受；后续可扩展为迭代 k-means
- 将 τ=0.90 作为 `--bse_tau` 暴露，便于做消融
- 反投影阶段有 OOM 风险 → 采用分块处理（见 `memory.txt` 设计）
- 不引入新的可训练参数——全部为代数运算
