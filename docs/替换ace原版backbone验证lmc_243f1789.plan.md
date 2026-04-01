---
name: 替换ACE原版Backbone验证LMC
overview: 用ACE原版FCN encoder替换DINOv2 backbone，验证LMC的memory特征融合机制在不同backbone上的通用性。需要创建ACE encoder适配器、修改训练脚本、运行对比实验并分析结果。
todos:
  - id: create_ace_adapter
    content: 创建 ace_network_ace.py，实现 ACEEncoder 适配器类（支持 RGB 输入、灰度转换、feature_dim 属性）
    status: pending
  - id: modify_trainer_dynamic_dim
    content: 修改 trainer_dinov2_lmc.py 第 212 行，从 regressor.encoder.feature_dim 动态获取 backbone 维度（默认 1024，保证 DINOv2 行为不变）
    status: pending
  - id: regression_verify_dinov2
    content: 修改后重跑一次 DINOv2+LMC 或 DINOv2 vanilla，确认结果与修改前一致（回归验证）
    status: pending
  - id: create_train_script
    content: 创建 train_ace_lmc.py，复制 train_ace_dinov2_lmc.py 并修改为使用 ACE encoder
    status: pending
  - id: run_ace_baseline
    content: 运行实验 E1：ACE vanilla baseline（28 轮迭代，heads 场景）
    status: pending
  - id: check_memory_compatibility
    content: 检查现有 DINOv2 memory 是否可复用，如不兼容则生成 ACE memory
    status: pending
  - id: run_ace_lmc
    content: 运行实验 E2：ACE + LMC（28 轮迭代，heads 场景）
    status: pending
  - id: analyze_results
    content: 对比分析 E1 vs E2 vs E3 vs E4，量化 LMC 在不同 backbone 上的增益
    status: pending
  - id: create_experiment_doc
    content: 创建 docs/lmc_backbone_generalization_experiment.md，记录实验设计、结果和结论
    status: pending
isProject: false
---

# 替换 ACE 原版 Backbone 验证 LMC 通用性

## 背景与目标

**当前状态：** 项目使用 DINOv2 (ViT-L/14) 作为 backbone，已验证 LMC 在 heads 场景上有效（89.7% vs 88.6% iterative baseline）。

**目标：** 用 ACE 原版 FCN encoder 替换 DINOv2，验证 LMC 的 memory 特征融合机制是否对不同 backbone 架构具有广泛有效性。

**核心问题：**

- ACE 原版 encoder 输入为**灰度图 (1 channel)**，输出 **512-dim** 特征，下采样 **8x**
- DINOv2 输入为 **RGB (3 channels)**，输出 **1024-dim** 特征，下采样 **14x**
- 需要创建适配层，确保 LMC 模块能无缝切换 backbone

---

## 约束：不影响原有功能

**所有修改必须保持对现有 DINOv2 路径的完全兼容，不得改变原有行为或结果。**

- **新增代码**：只新增文件（如 `ace_network_ace.py`、`train_ace_lmc.py`），不删除、不替换现有 DINOv2 相关代码；现有入口（`train_ace_dinov2_lmc.py`、`train_ace_dinov2_iterative.py` 等）保持可用且行为不变。
- **修改现有文件**：仅做最小必要改动；对 `trainer_dinov2_lmc.py` 的修改必须**向后兼容**：
  - 使用 `getattr(self.regressor.encoder, 'feature_dim', 1024)`，当 encoder 无 `feature_dim` 时默认 1024，保证 DINOv2 路径数值与逻辑与修改前一致。
- **验证要求**：修改后，在相同配置下重跑一次 DINOv2 + LMC（或 DINOv2 vanilla）实验，结果应与修改前一致（或仅在数值误差范围内）；若有 CI/回归测试，须通过。

---

## 架构对比


| 特性        | ACE 原版 Encoder                                          | DINOv2 Encoder         |
| --------- | ------------------------------------------------------- | ---------------------- |
| **架构**    | 自定义 FCN (轻量级)                                           | ViT-L/14 (Transformer) |
| **输入**    | 灰度图 `[B, 1, H, W]`                                      | RGB `[B, 3, H, W]`     |
| **输出维度**  | **512**                                                 | **1024**               |
| **下采样倍数** | **8x**                                                  | **14x**                |
| **预训练权重** | `/home/xwh/project/ace_depth/ace_encoder_pretrained.pt` | DINOv2 checkpoint      |
| **参数量**   | ~小 (几 MB)                                               | ~大 (300+ MB)           |


---

## 实施步骤

### 1. 创建 ACE Encoder 适配器类

**文件：** 新建 `[ace_depth/ace_network_ace.py](ace_depth/ace_network_ace.py)`

**目标：** 创建与 `DINOv2Encoder` 接口一致的 `ACEEncoder` 类

**关键设计：**

```python
class ACEEncoder(nn.Module):
    """
    ACE 原版 FCN encoder 的适配器，接口与 DINOv2Encoder 一致
    
    输入: RGB 图像 [B, 3, H, W]
    输出: 特征图 [B, 512, H//8, W//8]
    """
    def __init__(self, pretrained_path, out_channels=512, freeze_backbone=True):
        super().__init__()
        
        self.out_channels = out_channels
        self.patch_size = 8  # ACE 下采样倍数
        self.freeze_backbone = freeze_backbone
        self.feature_dim = 512  # 关键：LMC 通过此属性获取维度
        
        # 导入 ACE 原版 Encoder
        from ace_network import Encoder
        self.encoder = Encoder(out_channels=out_channels)
        
        # 加载预训练权重
        state_dict = torch.load(pretrained_path, map_location='cpu')
        self.encoder.load_state_dict(state_dict, strict=False)
        
        # 冻结 backbone
        if freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # RGB -> 灰度转换（使用标准权重）
        self.register_buffer('rgb_weights', torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1))
    
    def forward(self, x):
        """
        Args:
            x: RGB 图像 [B, 3, H, W]
        Returns:
            features: [B, 512, H//8, W//8]
        """
        # RGB -> 灰度
        gray = (x * self.rgb_weights).sum(dim=1, keepdim=True)  # [B, 1, H, W]
        
        # 提取特征
        with torch.set_grad_enabled(not self.freeze_backbone):
            features = self.encoder(gray)
        
        return features
```

**同时创建对应的 Regressor 和 Head 类：**

- 复用 `[ace_depth/ace_network.py](ace_depth/ace_network.py)` 中的 `Head` 类（已支持 `in_channels` 参数）
- 创建 `RegressorACE` 类，结构与 `ace_network_dinov2.py` 中的 `Regressor` 一致

---

### 2. 修改 LMC Trainer 支持动态 Backbone 维度

**文件：** `[ace_depth/trainer_dinov2_lmc.py](ace_depth/trainer_dinov2_lmc.py)`

**修改位置：** 第 212 行

**当前代码：**

```python
backbone_feature_dim = 1024  # 硬编码
```

**修改为：**

```python
# 动态从 regressor 获取 backbone 特征维度；无此属性时默认 1024，保证 DINOv2 路径行为不变
backbone_feature_dim = getattr(self.regressor.encoder, 'feature_dim', 1024)
_logger.info("[LMC] backbone_feature_dim=%d (from regressor.encoder)", backbone_feature_dim)
```

**说明：**

- DINOv2：`encoder.feature_dim` 已为 1024，修改前后 `backbone_feature_dim` 恒为 1024，**原有功能不变**。
- ACE：在 `ACEEncoder` 上设置 `feature_dim = 512`，LMC 自动适配。
- LMC 的 `fusion` 模块通过 `query_feature_dim` 支持不同维度。

---

### 3. 创建 ACE + LMC 训练脚本

**文件：** 新建 `[ace_depth/train_ace_lmc.py](ace_depth/train_ace_lmc.py)`

**策略：** 复制 `[train_ace_dinov2_lmc.py](ace_depth/train_ace_dinov2_lmc.py)`，修改以下部分：

**关键修改：**

```python
# 1. 导入 ACE 网络
from ace_network_ace import RegressorACE

# 2. 修改 get_lmc_train_parser()
parser.add_argument('--encoder_path', type=Path, 
                    default=Path(__file__).parent / "ace_encoder_pretrained.pt",
                    help='ACE encoder pretrained weights')

# 3. 修改 run_vanilla_iterative_baseline()
def run_vanilla_iterative_baseline(args):
    # 使用 RegressorACE 替代 Regressor
    regressor = RegressorACE.create_from_split(
        args.scene,
        encoder_path=args.encoder_path,
        num_head_blocks=args.num_head_blocks,
        use_homogeneous=args.use_homogeneous,
    )
    # ... 其余逻辑不变

# 4. 修改 TrainerACELMC 初始化
class TrainerACELMC(TrainerACEDINOv2LMC):
    # 继承 LMC 逻辑，仅替换 regressor 构建
    pass
```

---

### 4. 准备实验配置

**目标场景：** 先在 **heads** 上验证（已有 DINOv2 baseline 和 memory）

**实验对比矩阵：**


| 实验 ID  | Backbone | LMC | 预期 pct5     | 目的                        |
| ------ | -------- | --- | ----------- | ------------------------- |
| **E1** | ACE FCN  | ❌   | ~75-80%?    | ACE baseline（参考原版 ACE 性能） |
| **E2** | ACE FCN  | ✅   | **目标: >E1** | 验证 LMC 对 ACE backbone 有效  |
| **E3** | DINOv2   | ❌   | 88.6% (已有)  | DINOv2 baseline           |
| **E4** | DINOv2   | ✅   | 89.7% (已有)  | DINOv2 + LMC              |


**关键对比：**

- **E2 vs E1**：量化 LMC 对 ACE backbone 的增益
- **E2 vs E4**：对比不同 backbone + LMC 的性能上限
- **E1 vs E3**：对比 ACE 原版 vs DINOv2 的 baseline 性能

---

### 5. 运行实验

#### **实验 E1：ACE Vanilla Baseline (Iterative)**

**命令：**

```bash
cd /home/xwh/project/ace_depth

python train_ace_lmc.py \
    /mnt/storage/xwh/7Scenes/pgt_7scenes_heads \
    ace_heads_vanilla_baseline.pt \
    --experiment_root output \
    --encoder_path ace_encoder_pretrained.pt \
    --device cuda:3 \
    --use_lmc False \
    --vanilla_iterations 28 \
    --training_buffer_size 2560000 \
    --use_half False \
    --samples_per_image 384 \
    --batch_size 5120 \
    --image_resolution 480 \
    --epochs 24 \
    --learning_rate_max 5e-3 \
    --num_head_blocks 4 \
    --eval_each_iteration True \
    --keep_best_only True
```

**预计时间：** 6-8 小时（28 轮迭代）

---

#### **实验 E2：ACE + LMC**

**前置条件：** 需要为 ACE backbone 生成 memory（使用 heads 场景的现有 memory 或重新生成）

**命令：**

```bash
python train_ace_lmc.py \
    /mnt/storage/xwh/7Scenes/pgt_7scenes_heads \
    ace_heads_lmc.pt \
    --experiment_root output \
    --encoder_path ace_encoder_pretrained.pt \
    --device cuda:3 \
    --use_lmc True \
    --memory_path output/7Scenes/pgt_7scenes_heads/memory_bank/heads_memory.pt \
    --vanilla_iterations 28 \
    --training_buffer_size 2560000 \
    --use_half False \
    --samples_per_image 384 \
    --batch_size 5120 \
    --image_resolution 480 \
    --epochs 24 \
    --learning_rate_max 5e-3 \
    --num_head_blocks 4 \
    --lmc_mode global \
    --num_latent_tokens 64 \
    --eval_each_iteration True \
    --keep_best_only True
```

**注意：** 如果现有 memory 的特征维度与 ACE 不匹配，需要重新生成 memory：

```bash
# 使用 ACE encoder 提取 memory 特征
python scripts/create_memory_bank.py \
    /mnt/storage/xwh/7Scenes/pgt_7scenes_heads \
    output/7Scenes/pgt_7scenes_heads/memory_bank/heads_memory_ace.pt \
    --encoder_type ace \
    --encoder_path ace_encoder_pretrained.pt
```

---

### 6. 结果分析与对比

**分析维度：**

1. **LMC 通用性验证**
  - 对比 E2 vs E1：LMC 是否对 ACE backbone 有增益？
  - 对比 (E2-E1) vs (E4-E3)：LMC 在不同 backbone 上的增益是否一致？
2. **Backbone 架构影响**
  - 对比 E1 vs E3：轻量级 FCN vs 大型 ViT 的 baseline 性能差距
  - 对比 E2 vs E4：加上 LMC 后性能差距是否缩小？
3. **收敛效率**
  - ACE (8x 下采样) vs DINOv2 (14x 下采样) 的训练速度
  - 不同 backbone 达到最优性能所需的迭代轮数
4. **Memory 兼容性**
  - 如果复用 DINOv2 的 memory：验证 LMC 是否能跨 backbone 工作
  - 如果重新生成 ACE memory：对比不同特征空间的 memory 质量

**输出文档：** `[ace_depth/docs/lmc_backbone_generalization_experiment.md](ace_depth/docs/lmc_backbone_generalization_experiment.md)`

---

### 7. 可选扩展实验

**如果 E1-E2 验证成功，可进一步测试：**

1. **其他场景验证**
  - 在 stairs、chess 上重复 E1-E2 实验
  - 验证结论的跨场景一致性
2. **混合 Backbone 实验**
  - Memory 用 DINOv2 提取，Query 用 ACE encoder
  - 验证 LMC 的跨模态融合能力
3. **轻量化研究**
  - 对比 ACE (512-dim) vs DINOv2 (1024-dim) 的内存占用和推理速度
  - 评估 LMC 在资源受限场景下的实用性

---

## 关键文件清单

**新建文件：**

- `[ace_depth/ace_network_ace.py](ace_depth/ace_network_ace.py)` - ACE encoder 适配器
- `[ace_depth/train_ace_lmc.py](ace_depth/train_ace_lmc.py)` - ACE + LMC 训练脚本
- `[ace_depth/test_ace_lmc.py](ace_depth/test_ace_lmc.py)` - ACE + LMC 测试脚本（可选）

**修改文件：**

- `[ace_depth/trainer_dinov2_lmc.py](ace_depth/trainer_dinov2_lmc.py)` - 第 212 行，动态获取 backbone 维度

**参考文件：**

- `[ace_depth/ace_network.py](ace_depth/ace_network.py)` - ACE 原版网络定义
- `[ace_depth/ace_network_dinov2.py](ace_depth/ace_network_dinov2.py)` - DINOv2 网络定义（参考模板）
- `[ace_depth/train_ace_dinov2_lmc.py](ace_depth/train_ace_dinov2_lmc.py)` - DINOv2 训练脚本（参考模板）

**预训练权重：**

- `[ace_depth/ace_encoder_pretrained.pt](ace_depth/ace_encoder_pretrained.pt)` - ACE encoder 权重（已存在）

**实验输出：**

- `ace_depth/output/7Scenes/pgt_7scenes_heads/ace_vanilla/` - E1 结果
- `ace_depth/output/7Scenes/pgt_7scenes_heads/ace_lmc/` - E2 结果

---

## 时间线与资源规划


| 阶段          | 任务                   | 预计时间   | GPU    |
| ----------- | -------------------- | ------ | ------ |
| **Phase 1** | 创建 ACE encoder 适配器   | 1-2 小时 | -      |
| **Phase 2** | 修改 trainer 和训练脚本     | 1-2 小时 | -      |
| **Phase 3** | 运行 E1 (ACE baseline) | 6-8 小时 | cuda:3 |
| **Phase 4** | 生成 ACE memory（如需要）   | 1-2 小时 | cuda:3 |
| **Phase 5** | 运行 E2 (ACE + LMC)    | 6-8 小时 | cuda:3 |
| **Phase 6** | 结果分析与文档撰写            | 2-3 小时 | -      |


**总计：** 约 17-25 小时（大部分是自动训练）

---

## 成功标准

**最低标准（验证通过）：**

- ✅ E2 (ACE + LMC) > E1 (ACE baseline)，增益 ≥ 1%
- ✅ 代码无报错，训练流程稳定

**理想标准（强验证）：**

- ✅ E2 vs E1 的增益与 E4 vs E3 相当（±0.5%）
- ✅ ACE + LMC 在 3 个场景上均有一致增益
- ✅ Memory 可跨 backbone 复用（无需重新生成）

**失败标准（需重新审视）：**

- ❌ E2 ≤ E1（LMC 对 ACE 无效或负面影响）
- ❌ 训练不稳定（loss 爆炸、NaN 等）
- ❌ Memory 维度不兼容且无法修复

---

## 风险与应对

**风险 1：ACE encoder 输入灰度图，特征表达能力可能不足**

→ **应对：**

- 先运行 E1 验证 ACE baseline 性能
- 如果 E1 << E3（差距过大），考虑修改 ACE encoder 第一层为 3 通道输入
- 或使用 RGB 三通道分别提取特征后融合

**风险 2：Memory 特征维度不匹配**

→ **应对：**

- 优先尝试复用现有 DINOv2 memory（LMC fusion 支持不同维度）
- 如果不兼容，重新生成 ACE memory（需要额外 1-2 小时）

**风险 3：下采样倍数不同（8x vs 14x）导致空间分辨率差异**

→ **应对：**

- ACE 输出更高分辨率特征图（H//8 vs H//14），理论上有利
- 需要确保 Head 网络能正确处理不同分辨率的输入
- 可能需要调整 `image_resolution` 参数以对齐

**风险 4：学习率等超参数需要重新调优**

→ **应对：**

- E1 先使用 ACE 原版的默认超参数（`learning_rate_max=5e-3`）
- 如果性能不佳，参考 DINOv2 的配置进行调整
- 记录所有超参数变化，便于后续分析

---

## 后续行动（如果验证成功）

1. **扩展到其他 Backbone**
  - ResNet-50/101
  - EfficientNet
  - Swin Transformer
2. **优化 LMC 架构**
  - 研究最优的 `num_latent_tokens` 配置
  - 测试不同的 fusion 模式（global vs hierarchical）
3. **发表论文/技术报告**
  - 标题：「LMC: A Backbone-Agnostic Memory Compression Framework for Scene Coordinate Regression」
  - 核心贡献：证明 LMC 对多种 backbone 架构具有通用性

