# 系统架构：双分支记忆提取

**作者**：高级 ML 系统架构师
**阶段**：3 — 架构设计（修订版）
**日期**：2026-03-26

---

## 架构概览：两个并行分支

```
                    run_memory_extraction.py
                            |
                ┌───────────┴───────────┐
                |                       |
         分支 A（主线）           分支 B（备选）
      Map-Anything 流程        DINOv2 单尺度
      ├─ 多尺度特征              ├─ DINOv2-ViT-L/14
      ├─ 真实深度支持            ├─ Depth Anything V2
      ├─ 视图选择                ├─ 无视图选择
      └─ 已验证有效              └─ 无深度选项
```

**设计理由**：
- **分支 A**：已验证的 map-anything 方法，多尺度特征 + 真实深度
- **分支 B**：无深度数据集的通用框架（未来工作）
- 用户通过 `--backend` 标志选择：`mapanything`（默认）或 `dinov2`

---

## 分支 A：Map-Anything 流程（主线）

### A.1 核心组件

**1. 视图选择**（`select_optimal_memory_indices`）
- 复用：`mapanything.tasks.ace.memory_selection.select_optimal_memory_indices`
- 输入：完整训练数据集
- 输出：N_MEMORY 个索引（如 100 个视图）覆盖场景几何
- 策略：基于 FPS 的空间覆盖 + 位姿多样性

**2. 模型推理**（`model.infer`）
- 模型：Map-Anything Transformer（通过 `mapanything.models.init_model` 加载）
- 输入：选定视图，包含 RGB、深度、位姿、内参
- 输出：多尺度中间特征保存到 `.pt` 文件
- 关键参数：
  - `memory_efficient_inference=True`
  - `use_amp=True, amp_dtype="bf16"`
  - `ignore_depth_inputs=False`（使用真实深度）
  - `ignore_pose_inputs=False`（使用 GT 位姿）

**3. 多尺度特征处理**
- 特征：Map-Anything Transformer 的中间层 + 最终层
- 处理选项：
  - **网格对齐**（默认）：将所有尺度对齐到最大网格，拼接
  - **AnyUp 上采样**（可选）：学习上采样到图像分辨率
- 输出：网格分辨率或图像分辨率的拼接特征

**4. BSE 池化 + Welford 归一化**
- 与分支 B 设计相同（复用 BSEPooler 和 WelfordNormalizer）
- 输入：带多尺度特征的反投影 3D 点
- 输出：带归一化坐标的压缩记忆

### A.2 数据流

```python
# 步骤 1：视图选择
memory_indices = select_optimal_memory_indices(train_dataset, N_MEMORY)

# 步骤 2：准备输入视图
input_views = []
for idx in memory_indices:
    raw_data = train_dataset[idx]
    processed = prepare_batch_input(raw_data, device, mode="memory")
    input_views.append(processed)

# 步骤 3：模型推理（保存多尺度特征到 .pt）
predictions = model.infer(
    input_views,
    memory_efficient_inference=True,
    use_amp=True,
    amp_dtype="bf16",
    save_filename=features_path,
    ignore_depth_inputs=False,  # 使用真实深度
)

# 步骤 4：加载特征并反投影
features_dict = torch.load(features_path)
for view_idx, view_data in enumerate(input_views):
    # 提取该视图的多尺度特征
    feat_list = features_dict[f'view_{view_idx}']['features']

    # 对齐到网格或上采样到图像分辨率
    if use_anyup:
        features = process_multiscale_features_anyup(feat_list, ...)
    else:
        features = process_multiscale_features_to_grid(feat_list)

    # 用真实深度反投影
    points, ray_dirs, colors = unproject_with_real_depth(view_data, features)

    # BSE 池化
    pooled = bse_pooler.pool(points, features, colors, ray_dirs)
    welford.update(pooled['points'])
    save_temp_chunk(pooled)

# 步骤 5：Welford 归一化（与分支 B 相同）
mu, sigma = welford.finalize()
normalize_and_save_final_memory(mu, sigma)
```

### A.3 关键技术细节

**真实深度处理**：
- 支持稠密和稀疏深度
- 稀疏深度：仅反投影有效像素（depth > 0）
- 分支 A 中无深度预测回退（无深度时使用分支 B）

**多尺度特征维度**：
- 典型：4-5 个中间层 + 1 个最终层
- 总通道数：~5120（取决于模型架构）
- 网格分辨率：通常为 37×37 或 74×74

**视图选择策略**：
- `select_optimal_memory_indices` 在相机位置上使用 FPS
- 确保场景几何的空间覆盖
- 默认 N_MEMORY：100 个视图（可通过配置调整）

---

## 分支 B：纯学习式记忆提取（未来工作）

**目的**：无深度数据集的通用框架，基于纯学习的场景表示。

**与分支 A 的关键区别**：
- 无视图选择（处理所有训练图像）
- 无深度依赖（既不用真实深度也不用预测深度）
- 纯学习式特征压缩和场景表示
- 参考：SceneTok 项目的学习式场景标记化方法

**何时使用**：
- 数据集无深度标注
- Map-anything 模型不可用
- 需要完全端到端的学习式方法

**设计方向**（待细化）：
- 借鉴 SceneTok 的场景标记化思想
- 学习式 3D 场景表示而非几何反投影
- 端到端训练特征提取和压缩
- 与 ACE 训练流程解耦

**实现**：推迟到分支 A 验证后。详细设计见未来的 `system_design_branch_b_learned.md`。

---

## 统一文件组织

```
memory_extraction/
├── __init__.py
├── backends/
│   ├── __init__.py
│   ├── mapanything_backend.py    # 分支 A 实现
│   └── learned_backend.py         # 分支 B 实现（未来）
├── common/
│   ├── bse_pooling.py             # 共享 BSE 算法
│   ├── welford_meter.py           # 共享归一化
│   └── utils.py                   # 共享工具
├── run_memory_extraction.py       # 主入口（分发到后端）
└── extract_memory.sh              # Bash 启动器
```

**后端接口**（两个分支都实现）：
```python
class MemoryBackend:
    def select_views(self, dataset, n_views) -> List[int]
    def extract_features(self, views) -> Dict[str, Tensor]
    def unproject(self, view, features) -> Tuple[Tensor, ...]
```

---

## 实施路线图（修订版）

### Sprint 1：共享基础设施
- `bse_pooling.py`、`welford_meter.py`（与之前相同）
- 后端接口定义

### Sprint 2：分支 A（Map-Anything）— 优先
- `mapanything_backend.py`
- 集成 `select_optimal_memory_indices`
- 用适当的错误处理包装 `model.infer`
- 多尺度特征处理（网格对齐 + 可选 AnyUp）

### Sprint 3：分支 B（纯学习式）— 未来工作
- `learned_backend.py`
- 研究 SceneTok 方法
- 设计端到端学习式场景表示

### Sprint 4：评估
- `eval_boundary_metrics.py`
- 在有真实深度的场景上比较分支 A 与原始方法

---

## 从 map-anything 迁移

**要保留的依赖**：
- `mapanything.models.init_model`
- `mapanything.datasets.{SevenScenesWAI, Indoor6WAI}`
- `mapanything.tasks.ace.memory_selection.select_optimal_memory_indices`

**要替换的依赖**：
- Hydra 配置 → argparse
- 硬编码路径 → CLI 参数
- 可视化辅助工具 → 可选（保留用于调试）

**向后兼容性**：
- 输出 `.pt` 格式与 map-anything 匹配，用于下游 ACE 训练
- 可以加载现有的 map-anything 提取记忆
