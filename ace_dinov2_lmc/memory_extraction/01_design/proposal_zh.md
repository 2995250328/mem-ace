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
- **自适应阈值选择**：固定 τ 在不同对比度场景下有过分割/欠分割风险。Otsu 方法提供有原则的、数据驱动的阈值选择。
- **分块处理与全局归一化**：在分块数据上计算 μ_scene、σ_scene 需要流式算法（Welford）以保持数学一致性。
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
- **自适应阈值**：使用 Otsu 方法从余弦相似度直方图中找到最优分裂点 τ_otsu（无固定超参数）
- 优点：保留语义边界；O(N) 向量化；无新依赖；适应场景对比度
- 缺点：假设双峰相似度分布（通过单峰保护缓解）

**选择范式 C 的理由：**
- 理论：双边滤波是保留不连续性的原则性方法
- 实践：可用 index_add_（代码库中已有）实现，无需 torch_scatter
- 自适应：Otsu 方法根据局部特征分布自动为每个块找到最优阈值
- 消融友好：可用 --use_otsu False 禁用 Otsu 回退到固定 τ

### 详细管线步骤

**步骤 1 — 全光反投影**

对每个有效像素 (u, v)，深度 Z，内参 K，旋转 R，平移 t：
```
P_cam = Z · K⁻¹ · [u, v, 1]ᵀ
P_raw = R · P_cam + t          # 世界坐标 [N, 3]
O_cam = t                       # 相机光心世界坐标 [3]
D_raw = (P_raw - O_cam) / ‖P_raw - O_cam‖₂   # 单位射线方向 [N, 3]
```

**步骤 2 — 粗几何哈希**
```
V_idx = floor(P_raw / voxel_size).long()   # [N, 3]
_, cluster_ids = torch.unique(V_idx, dim=0, return_inverse=True)  # [N]
```

**步骤 3 — 双边特征聚类（保边分裂，核心步骤）**

必须避免 Python 循环，全向量化：
```
# 通过 index_add_ 计算簇均值特征（无需 torch_scatter）
F_fp32 = F_raw.float()                          # 转 fp32 保证数值稳定
F_mean = zeros(M, C).index_add_(0, cluster_ids, F_fp32) / count  # [M, C]

# 每个点与其簇均值的余弦相似度
F_i    = F_fp32                                 # [N, C]
F_ci   = F_mean[cluster_ids]                    # [N, C]
sim_i  = (F_i * F_ci).sum(1) / (‖F_i‖ · ‖F_ci‖ + 1e-6)  # [N], fp32

# Otsu 自适应阈值（自动找到最优分裂点）
hist, bins = histogram(sim_i, bins=256, range=[0, 1])
τ_otsu = otsu_threshold(hist, bins)             # 最大化类间方差

# 单峰保护：如果分布过于均匀则跳过分裂
if std(sim_i) < 0.02:
    τ_otsu = -1.0  # 强制所有点归入主簇

# 二值分裂：语义与簇主体差异大的点标记为离群点
is_outlier = (sim_i < τ_otsu).long()            # [N], 0 或 1
sub_id = cluster_ids * 2 + is_outlier           # [N]
```

**Otsu 方法**：将相似度分布视为两类混合（主簇 vs. 边界）。搜索最大化类间方差的阈值，等价于最小化类内方差。对双峰分布证明最优。

**步骤 4 — 细粒度池化**
```
# 用 sub_id 对 {P_raw, D_raw, F_raw, colors} 做 scatter_mean
P_bse, D_bse, F_bse, C_bse = scatter_mean_all(sub_id)
D_bse = D_bse / (‖D_bse‖₂ + 1e-6)             # 重新归一化射线方向
```

**步骤 5 — 全局度量归一化**

在 BSE 池化之后计算（对 P_bse，而非 P_raw——避免离群点污染统计量）：
```
μ_scene = mean(P_bse, dim=0)                    # [3]
σ_scene = std(‖P_bse - μ_scene‖₂).item()       # 标量
P_norm  = (P_bse - μ_scene) / σ_scene           # [N, 3]
```

典型室内场景的 P_norm ∈ [-3, 3]（3σ 覆盖）。

### OOM 缓解：Welford 流式归一化

对于大场景（>1000 张高分辨率图像），全量反投影超出 GPU 显存。
使用 Welford 在线算法实现数值稳定的 O(1) 内存全局统计：

```
第一阶段（逐块流式）：每次处理 B 张图像：
    反投影 → D_raw → BSE（步骤 1-4）→ P_bse_chunk

    # Welford 在线更新（FP64 保证数值稳定）
    for each point p in P_bse_chunk:
        count += 1
        delta = p - mean
        mean += delta / count
        M2 += delta * (p - mean)  # 平方偏差的累积和

    # 下盘到磁盘（临时 .pt 文件）
    save(P_bse_chunk, F_bse_chunk, D_bse_chunk, f"temp_chunk_{i}.pt")

第二阶段（全局缩放）：所有块处理完毕后：
    μ_scene = mean                          # [3], FP64 → FP32
    σ_scene = sqrt(M2 / count)              # 标量, FP64 → FP32

    # 流式归一化每个块
    for each temp_chunk_{i}.pt:
        load(P_bse_chunk, F_bse_chunk, D_bse_chunk)
        P_norm_chunk = (P_bse_chunk - μ_scene) / σ_scene
        追加到最终 buffer

    # 保存最终记忆
    save(P_norm, F_bse, D_bse, C_bse, μ_scene, σ_scene, "pooled_memory.pt")
    清理临时文件
```

**关键优势**：
- O(1) 内存：仅存储运行统计量，不存储所有点
- 数值稳定：Welford 算法避免方差计算中的灾难性相消
- 数学精确：产生与全批次计算相同的 μ、σ（FP64 精度内）
- 无双重池化：BSE 每块仅运行一次，归一化是简单标量操作

默认：B = 16 张图像/块，临时文件写入 /dev/shm（RAM 盘）以实现快速 I/O。

### 自适应回退

若 BSE 后 len(P_bse) > 100,000（压缩不足）：
- 将 voxel_size 加倍，重新执行步骤 2-4
- 记录警告日志，包含原始和新的点数

## 4. 评估与验证计划

### 数据集
- **7-Scenes Chess**：冒烟测试（小、快、理解充分）
- **Indoor6 scene3, scene4a**：主要消融（dino_lmc_base 中有现有评估结果）

### 指标
- **压缩率**：len(P_bse) / total_input_pixels（目标：<0.05）
- **FVR（特征方差保留率）**：Var(F_bse) / Var(F_raw) — 测量特征空间中的信息保留（目标：>0.90）
- **SCD（语义倒角距离）**：联合空间 [P_xyz, λ·F_dino] 中原始点云与池化点云的倒角距离（目标：<0.5× 基线）
- **重定位精度**：Indoor6 上的中位平移误差（cm）和旋转误差（°）
- **归一化检查**：断言 P_norm ∈ [-5, 5]（99.9% 的点）

### 基线
1. `--use_bse False`（均匀体素池化，当前行为）
2. `--use_bse True --use_otsu True`（提议：Otsu 自适应 BSE）
3. `--use_bse True --use_otsu False --bse_tau 0.90`（固定阈值 BSE，用于比较）

### 消融研究
- Otsu vs. 固定 τ：在 Indoor6 scene3（不同对比度）上比较 FVR 和 SCD
- voxel_size 扫描：{0.03, 0.05, 0.10} 在 7-Scenes Chess 上
- 单峰保护：测量同质区域（平坦墙面）上的错误分裂率
- Welford 精度：验证 μ_scene、σ_scene 与全批次计算匹配到小数点后 4 位

## 5. 预期失败模式与工程风险

| 风险 | 可能性 | 缓解措施 |
|------|--------|---------|
| ace 环境中无 torch_scatter | 高 | 使用 index_add_ + 计数除法（代码库中已验证） |
| 大场景反投影时 OOM | 中 | Welford 流式处理，O(1) 内存，临时文件写入 /dev/shm |
| FP16 余弦相似度数值不稳定 | 高 | 计算前转 fp32；分母加 ε=1e-6 |
| Welford M2 累加器 FP32 精度损失 | 高 | mean 和 M2 使用 FP64，仅在输出时转 FP32 |
| Otsu 在单峰分布上失效 | 中 | 单峰保护：std(sim) < 0.02 时跳过分裂 |
| Otsu 假设双峰分布 | 中 | 对 DINOv2 特征可接受（语义边界创造自然双峰性） |
| 临时文件下盘的磁盘 I/O 瓶颈 | 低 | 写入 /dev/shm（RAM 盘）而非 HDD；异步 torch.save |
| 输出 >100k 点（压缩不足） | 低 | 自适应回退：加倍 voxel_size 并重试一次 |
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

## 7. 实施行动路线图

三步渐进式冲刺——每步均可独立测试：

**冲刺 1 — Welford 流式归一化（解决数学矛盾）**
- 实现带 FP64 累加器（count、mean、M2）的 WelfordMeter 类
- 重写 run_memory_extraction.py 中的主循环以实现两阶段处理
- 第一阶段：流式处理块，更新 Welford 状态，下盘到临时文件
- 第二阶段：加载临时文件，应用归一化，组装最终记忆
- 验收：验证 μ_scene、σ_scene 与全批次计算匹配到小数点后 4 位

**冲刺 2 — Otsu 自适应阈值（解决超参数质疑）**
- 实现向量化的 otsu_threshold(sim_tensor, bins=256) 函数
- 添加单峰保护：std(sim) < 0.02 时跳过分裂
- 将 `is_outlier = sim < 0.90` 替换为 `is_outlier = sim < τ_otsu`
- 记录每个块的 τ_otsu 以观察动态适应
- 验收：验证 τ_otsu 在不同对比度的块之间变化

**冲刺 3 — 定量评估指标（提供反驳弹药）**
- 实现 eval_boundary_metrics.py 脚本
- 计算 FVR = Var(F_bse) / Var(F_raw) 用于边界保留
- 在联合 [P, λ·F] 空间中使用倒角距离计算 SCD
- 生成比较表：均匀体素 vs. Otsu-BSE
- 验收：Otsu-BSE 的 FVR > 0.90，均匀基线的 FVR < 0.50
