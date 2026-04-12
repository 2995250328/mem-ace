# Task Prompts

本目录集中存放**新对话用任务 Prompt**：复制对应文档（或其中的小结段落）到新会话作为第一条消息，可驱动模型按约定角色、Skills 顺序与验收标准完成任务。

This folder contains **task prompts for new conversations**. Copy a document (or its summary block) as the first message in a new chat to drive the model with the intended role, skill order, and acceptance criteria.

---

## 目录 / Index

| 文件 | 说明 |
|------|------|
| [prompt_替换ACE原版backbone.md](prompt_替换ACE原版backbone.md) | 用 ACE 原版 backbone 替换 DINO，验证 LMC 通用性（中文） |
| [prompt_replace_dino_with_ace_backbone.md](prompt_replace_dino_with_ace_backbone.md) | Replace DINO with ACE original backbone, verify LMC generality (English) |
| [prompt_sampling_logic.md](prompt_sampling_logic.md) | 基于原版 ACE：先 e2e 单独训练采样器（无 buffer），再替换 buffer 随机采样；可选邻域/Part A/B（参考 ace_trainer_full SuperPoint）；新代码入子目录（中英双语） |
| [prompt_scenetok_replace_lmc.md](prompt_scenetok_replace_lmc.md) | 用 SceneTok 替换 LMC：场景非结构 token 作全局 memory 与 query 融合；新代码归入专用子目录（中英双语） |

---

## 相关文档（位于 `docs/`）

- **设计/计划**：`docs/替换ace原版backbone验证lmc_243f1789.plan.md`（backbone 替换的详细步骤与约束）
- **工作流**：`docs/indoor6_multi_scene_workflow.md`（多场景 memory 提取与训练流程）
- **后续思路**：`docs/后续.md`（各任务简述与对应 prompt 链接）
