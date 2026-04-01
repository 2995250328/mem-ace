# 任务 Prompt：用 ACE 原版 Backbone 替换 DINO，验证现有逻辑通用性

**请在新对话中复制本段作为第一条消息，以充分发挥模型能力并保证修改质量。**

---

## 你的角色与准则

- 你正在协助一个**三维视觉 / 相机重定位 / 深度估计**的深度学习项目（ace_depth）。代码严谨性、数学正确性与实验可复现性优先。
- **证据优于断言**：在声称“修复完成”或“逻辑不变”前，必须有可执行的验证命令与期望结果说明；不主动运行本仓库中的长时间验收/端到端脚本，而是给出完整命令由用户本地执行后贴回结果。
- **谋定而后动**：在改核心架构或新增 backbone 路径前，先理清接口契约与影响范围，再动手写代码。

---

## 任务目标（请严格遵循）

1. **用 ACE 原版 backbone 替换 DINO**  
   - 当前：backbone 为 DINOv2（ViT-L/14，1024 维，patch 14，RGB 输入）。  
   - 目标：增加一条**可选**的、使用 **ACE 原版 FCN encoder**（`ace_network_depth.py` 中的 `Encoder`，512 维，8x 下采样，灰度输入，权重 `ace_encoder_pretrained.pt`）的完整训练与测试流程。

2. **其他流程完全保持**  
   - LMC（GeoLMC）迭代训练、buffer 填充、fusion、评估脚本、数据格式、多场景工作流等**一律不改动逻辑**，仅通过“选择 backbone”的方式切换 DINO 与 ACE 原版。

3. **验证现有逻辑的通用性**  
   - 目标：证明当前 LMC/训练/评估流程与 backbone 解耦，换 ACE 原版后仍能跑通并可比对（vanilla vs LMC）。

4. **完全不影响现有逻辑**  
   - **禁止**删除或弱化现有 DINOv2 相关代码与入口。  
   - 现有入口（如 `train_ace_dinov2_lmc.py`、`train_ace_dinov2_iterative.py`、`test_ace_dinov2.py`、`test_ace_dinov2_lmc.py`）行为必须与修改前一致。  
   - 对现有文件的修改仅限于**最小必要**且**向后兼容**（例如用 `getattr(encoder, 'feature_dim', 1024)` 等保证 DINO 路径不变）。

---

## 你必须使用的 Skills（按顺序）

1. **先阅读并执行**  
   - `~/.cursor/skills/using-superpowers/SKILL.md`：确认本任务涉及到的 skills 均被调用。  
   - `~/.cursor/skills/brainstorming/SKILL.md`：在写代码前，先探索项目上下文、明确 backbone 接口契约（输入/输出维度、分辨率约束）、提出 2–3 种实现方式并给出推荐，形成设计并得到用户确认后再实现。  
   - 若已有实现计划，可选用 `~/.cursor/skills/writing-plans/SKILL.md`：将任务拆成带具体文件路径与测试步骤的 bite-sized 计划（保存到 `docs/plans/`），再逐步执行。

2. **实现与验证阶段**  
   - 若涉及多步实现：使用 `~/.cursor/skills/test-driven-development/SKILL.md`（先写失败测试，再最小实现）。  
   - 在声称“现有逻辑不受影响”前：使用 `~/.cursor/skills/verification-before-completion/SKILL.md`，列出回归验证命令与通过标准，由用户本地执行并贴回结果后再收尾。

3. **可选**  
   - 若你希望本对话自动采用“3D 视觉 + 严谨实现”的身份准则，可先应用 `@identity-prompt-by-context` 中与路径/关键词匹配的 prompt（如 research-advisor-3dv 或 default）。

---

## 关键代码与文档位置（供你检索）

- **DINO 当前接入**：`ace_network_dinov2.py`（`DINOv2Encoder`、`Regressor`）、`trainer_dinov2.py`（`TrainerACEDINOv2`）、`trainer_dinov2_lmc.py`（`TrainerACEDINOv2LMC`，含 `backbone_feature_dim`）、`options_dinov2_lmc.py`（`dinov2_path`、`freeze_backbone`）。  
- **ACE 原版 Encoder**：`ace_network_depth.py` 中的 `Encoder` 类（灰度输入、512 维、8x 下采样）、`ace_encoder_pretrained.pt`。  
- **已有设计参考**：`ace_depth/docs/替换ace原版backbone验证lmc_243f1789.plan.md`（内含接口对比、约束“不影响原有功能”、适配器类设计要点与步骤）。  
- **工作流与数据**：`ace_depth/docs/indoor6_multi_scene_workflow.md`。

---

## 接口契约（Backbone 抽象）

新增的 ACE 原版路径必须与现有 LMC/训练流程兼容，建议统一 backbone 接口（与 `DINOv2Encoder` 对齐）：

- **输入**：`[B, C, H, W]`（若 ACE 原版为灰度，在适配器内做 RGB→灰度转换，对外仍可接收 RGB 以复用现有 dataset）。  
- **输出**：特征图 `[B, feature_dim, H', W']`，且 encoder 具 `feature_dim`、`patch_size`（或等效下采样倍数）属性，供 trainer 与 fusion 使用。  
- **权重**：通过 `encoder_path`（或等效参数）加载 `ace_encoder_pretrained.pt`，不写死路径。

---

## 验收与回归（由用户本地执行）

- **回归**：在相同配置下运行一次现有 DINOv2 + LMC 或 DINOv2 vanilla 流程（例如 `train_ace_dinov2_lmc.py` 或 `train_ace_dinov2_iterative.py` 的典型命令），结果应与修改前一致（或仅在数值误差范围内）。  
- **新路径**：能使用 ACE 原版 backbone 完成至少一场景的 vanilla 与（若适用）LMC 训练与评估，并得到可比较的指标。  
- 请你在完成实现后**给出上述回归与新路径的完整命令与期望输出/通过标准**，不要主动执行长时间验收脚本。

---

## 小结（复制到新对话时可直接用下面一段）

**任务**：在 ace_depth 中增加“ACE 原版 backbone”的可选路径，替换当前使用的 DINO，其余流程（LMC、训练、评估、数据）完全保持，以验证现有逻辑的通用性；且**不得影响**现有 DINO 相关代码与行为。  
**要求**：先按 `brainstorming` 做设计与接口确认，再按 `writing-plans` 或 TDD 拆解实现；对现有文件的修改须最小且向后兼容；验收与回归命令由你给出、用户本地执行。  
**参考**：`ace_depth/docs/替换ace原版backbone验证lmc_243f1789.plan.md`、`ace_network_dinov2.py`、`ace_network_depth.py`、`trainer_dinov2_lmc.py`。
