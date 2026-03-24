# 标题：面向视觉重定位的场景无关记忆压缩双边超体素提取方法

## 1. 问题定义

给定 N 张训练图像 {I_i}，已知位姿 {T_i ∈ SE(3)} 和相机内参 {K_i}，
记忆提取流水线生成紧凑场景记忆 M，供下游 ACE DINOv2 LMC 重定位系统使用。

**当前流水线（范式 A — 均匀体素池化）：**
- 将每个像素反投影到 3D：P_raw ∈ R^{H×W×3}
- 将特征 F ∈ R^{H×W×C} 体素池化到固定网格 → M_voxel

**局限性：**
- 均匀体素网格忽略语义边界 → 物体边缘特征模糊
- 无全局度量归一化 → 场景相关坐标尺度破坏零样本迁移
- 无光线方向存储 → 下游 Transformer 无法推理视角相关效果
- 与 mapanything/Hydra 生态系统紧耦合 → 无法在 ace_depth 数据集上独立运行

**目标定义（范式 C — BSE + 全局归一化）：**

输入：{I_i, T_i, K_i}
输出：M = {P_norm, D_bse, F_bse, C_bse, μ_scene, σ_scene}

其中：
- P_norm = (P_raw − μ_scene) / σ_scene  ∈ R^{N×3}  （零均值，单位标准差）
- D_bse  = normalize(P_raw − O_cam)     ∈ R^{N×3}  （单位光线方向）
- F_bse  ∈ R^{N×C}  （双边池化后的 DINOv2 特征）
- C_bse  ∈ R^{N×3}  （池化后的 RGB 颜色）
- μ_scene ∈ R^3, σ_scene ∈ R  （场景统计量，用于反归一化）

目标：在实现 >95% 点压缩率的同时最大化语义边界保留。

## 2. 文献综述与动机

### 代表性方法

**ACE（Cavallari 等，CVPR 2023）**
- 假设：场景坐标可从局部块特征回归，无需显式 3D 地图
- 局限：无持久场景记忆；每个场景需从头完整重训练
- 差距：无法利用跨场景几何先验

**Map-Anything / ACE-G（内部，2025）**
- 假设：预提取的体素池化记忆 M_voxel 可指导迭代 LMC 训练
- 局限：均匀体素网格 → 语义边界模糊；Hydra/mapanything 耦合
- 差距：记忆质量直接限制重定位精度；无原则性特征-几何对齐

**PointNeRF（Xu 等，CVPR 2022）**
- 假设：带逐点特征的神经点云可实现视图合成
- 局限：需要稠密深度；逐场景优化；非为重定位设计
- 差距：特征池化策略不可迁移到稀疏重定位场景

### 研究差距

现有记忆提取流水线在池化时均匀对待所有 3D 点。
这忽略了语义边界（物体间边缘）对重定位携带不成比例的判别信息这一事实。
此外，场景特定坐标尺度阻止了下游回归器的零样本迁移。

**本想法同时解决两个差距：**
1. 语义感知池化：双边聚类保留高频边界
2. 全局度量归一化：实现场景无关的下游训练

### 为何非平凡

- 3D 点云中的双边滤波需要高效 GPU 实现，且不依赖 torch_scatter
- 二值分割（主/离群）必须通过 τ 校准以避免过分割
- 全局归一化必须在池化前计算以避免数据泄漏
- 从 Hydra/mapanything 迁移到独立 argparse 需要仔细的依赖分析

## 3. 设计空间与选定架构

### 范式 A — 均匀体素池化（当前基线）
- 简单网格量化，每体素平均特征
- 优点：O(N) 时间，除 voxel_size 外无超参数
- 缺点：模糊语义边界；无特征-几何对齐

### 范式 B — FPS + kNN 池化
- 最远点采样选择代表点，kNN 聚合邻居
- 优点：均匀空间覆盖
- 缺点：>1M 点时 O(N²) FPS 极慢；丢失边界结构

### 范式 C — 双边超体素提取（已选）
- 两级聚类：粗粒度几何哈希 → 按特征相似度细粒度双边分割
- 优点：保留语义边界；O(N) 向量化；无新依赖
- 缺点：二值分割对复杂边界可能过粗（v1 可接受）

**选择范式 C 的理由：**
- 理论：双边滤波是保留不连续性的原则性方法
- 实践：可用 index_add_（代码库中已有）实现，无需 torch_scatter
- 消融友好：τ 作为 CLI 参数暴露；可用 --use_bse False 禁用 BSE

### 核心算法：BSE

```
步骤 1：coarse_voxel_hash(P, voxel_size) → cluster_ids  [O(N)]
步骤 2：scatter_mean(F, cluster_ids) → F_cluster_mean   [O(N) via index_add_]
步骤 3：cosine_sim(F_i, F_cluster_mean[cluster_ids[i]]) → sim_i  [O(N)]
步骤 4：is_outlier_i = (sim_i < τ)
步骤 5：sub_id_i = cluster_ids[i] * 2 + is_outlier_i
步骤 6：scatter_mean({P,D,F,C}, sub_id) → 池化输出  [O(N)]
步骤 7：L2 归一化池化后的 D
```

### 全局度量归一化

在反投影后、池化前计算：
```
μ_scene = mean(P_raw, dim=0)          # [3]
σ_scene = std(P_raw - μ_scene).item() # 标量
P_norm  = (P_raw - μ_scene) / σ_scene
```

这确保典型室内场景的 P_norm ∈ [-3, 3]（3σ 覆盖）。

## 4. 评估与验证计划

### 数据集
- **7-Scenes Chess**：冒烟测试（小、快、理解充分）
- **Indoor6 scene3, scene4a**：主要消融（dino_lmc_base 中有现有评估结果）

### 指标
- **压缩率**：len(P_bse) / total_input_pixels（目标：<0.05）
- **边界保留**：Open3D 定性可视化
- **重定位精度**：Indoor6 上的中位平移误差（cm）和旋转误差（°）
- **归一化检查**：断言 P_norm ∈ [-5, 5]（99.9% 的点）

### 基线
1. `--use_bse False`（均匀体素池化，当前行为）
2. `--use_bse True --bse_tau 0.90`（提议的 BSE）
3. `--use_bse True --bse_tau 0.70`（激进分割，消融）
4. `--use_bse True --bse_tau 0.95`（保守分割，消融）

### 消融研究
- τ 扫描：{0.70, 0.80, 0.90, 0.95} 在 Indoor6 scene3 上
- voxel_size 扫描：{0.03, 0.05, 0.10} 在 7-Scenes Chess 上
- 有/无全局归一化（P_raw vs P_norm 作为下游训练器输入）

## 5. 预期失败模式与工程风险

| 风险 | 可能性 | 缓解措施 |
|------|--------|---------|
| ace 环境中无 torch_scatter | 高 | 使用 index_add_ + 计数除法（代码库中已验证） |
| 大场景反投影时 OOM | 中 | 分块处理（chunk_size=500k） |
| τ=0.90 过度分割平坦墙面 | 中 | 暴露为 --bse_tau；默认保守 |
| 输出 >100k 点（压缩不足） | 低 | 自适应回退：加倍 voxel_size 并重试 |
| mapanything 导入路径失效 | 高 | 用直接等价物替换所有 mapanything.* 导入 |
| WAI 数据集格式不兼容 | 中 | 先用 ace 后端测试；WAI 支持可选 |

## 6. 复用计划

### 继承自 dino_lmc_base
- `../trainer_dinov2_lmc.py`：DINOv2 模型加载模式（第 1-80 行展示导入结构）
- `../options_dinov2_lmc.py`：argparse 构建模式，`_strtobool`，`--data_backend` 参数
- `../train_ace_dinov2_lmc.py`：脚本入口点结构（sys.path，设备设置）

### 来自项目根目录
- `../../dataset_dinov2.py`：`CamLocDatasetDINOv2` — RGB 加载，内参，位姿加载
- `../../ace_util.py`：`to_homogeneous` — 坐标变换

### 新文件（仅此想法）
- `memory_extraction/__init__.py` — 空包标记
- `memory_extraction/bse_pooling.py` — 独立 BSE 算法（除 torch 外无外部依赖）
- `memory_extraction/run_memory_extraction.py` — 迁移 + 增强的提取脚本
- `memory_extraction/extract_memory.sh` — 适配的 bash 启动器（无 Hydra）

### 迁移源（只读参考）
- `map-anything/mapanything/tasks/run_memory_extraction.py`
- `map-anything/bash_scripts/ace/fps_memory.sh`
