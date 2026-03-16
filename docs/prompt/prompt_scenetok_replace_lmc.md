# 用 SceneTok 替换 LMC / Replace LMC with SceneTok — 任务 Prompt（中英）

**在新对话中复制对应语言段落作为第一条消息，以充分发挥模型能力并保证修改质量。**

---

# 中文

## 你的角色与准则

- 你正在协助一个**三维视觉 / 相机重定位 / 深度估计**的深度学习项目（ace_depth）。代码严谨性、数学正确性与实验可复现性优先。
- **证据优于断言**：在声称“修复完成”或“逻辑正确”前，必须有可执行的验证命令与期望结果说明；不主动运行本仓库中的长时间验收/端到端脚本，而是给出完整命令由用户本地执行后贴回结果。
- **谋定而后动**：在替换 LMC 或接入 SceneTok 前，先理清 SceneTok 的 token 接口、与现有 memory/fusion 的对应关系，以及新代码的目录契约，再动手写代码。

---

## 任务目标（请严格遵循）

1. **用 SceneTok 替换当前的 LMC 模块**  
   - 当前 LMC：基于深度/几何的 memory 压缩（如 FPS、pooled 点云特征）与 query 做 cross-attention 融合（见 `ace_compressor.py`、`ace_fusion.py`、`trainer_dinov2_lmc.py`）。  
   - 目标：用**预训练好的新视角合成方法 SceneTok** 得到的**场景非结构表示**（压缩的、可扩散的 1D scene tokens）作为**全局 memory**，与 query 特征进行融合，实现特征增强；相当于用 SceneTok 的 encoder/compressor 输出替代或补充现有 LMC 的 memory 来源。

2. **其他流程尽量保持**  
   - 主训练流程（backbone、head、buffer、迭代方式）与现有入口可保持一致；仅将“memory 来源 + 与 query 的融合方式”从当前 LMC 换为 SceneTok 驱动的路径；现有 DINO+LMC 路径保持可用（可选切换或独立入口）。

3. **新代码全部归入专用子目录（强制）**  
   - **禁止**在 ace_depth 主目录下继续散落 ace-scenetok 相关新文件（避免重蹈当前 LMC 相关文件散落在根目录的覆辙）。  
   - 所有与“ACE + SceneTok”集成相关的**新代码**必须放在一个**专用子文件夹**内，例如：  
     - `ace_depth/ace_scenetok/` 或 `ace_depth/scenetok_integration/`  
   - 该目录内应自洽组织：例如 `model/`（SceneTok 适配器、fusion 模块）、`options.py`、`trainer.py` 或训练入口、`README.md` 说明依赖与用法；对外仅通过少量入口（如根目录下一个 thin 脚本或 `train_ace_scenetok.py` 调用子目录）接入，保持根目录整洁。  
   - 具体子目录名称与内部结构在 brainstorming 阶段与用户确认后确定，并在 writing-plans 中写明，实现时严格遵循。

4. **不破坏现有 LMC 行为**  
   - 现有 `train_ace_dinov2_lmc.py`、`trainer_dinov2_lmc.py`、`ace_compressor.py`、`ace_fusion.py` 等在不启用 SceneTok 路径时行为不变；回归验证通过后再收尾。

---

## 你必须使用的 Skills（按顺序）

1. **先阅读并执行**  
   - `~/.cursor/skills/using-superpowers/SKILL.md`：确认本任务涉及到的 skills 均被调用。  
   - `~/.cursor/skills/brainstorming/SKILL.md`：在写代码前，先探索 ace_depth 与 SceneTok 项目结构、明确（1）SceneTok 的输入/输出（view set → scene tokens）、（2）与现有 memory 的接口对应（token 维度、序列长度、是否需 FPS/池化）、（3）fusion 形式（cross-attention query=backbone feature, memory=scene tokens）、（4）**专用子目录的命名与内部模块划分**，提出 2–3 种实现方式并给出推荐，形成设计并得到用户确认后再实现。  
   - 若已有实现计划，可选用 `~/.cursor/skills/writing-plans/SKILL.md`：将任务拆成带**具体子目录路径与文件清单**的 bite-sized 计划（保存到 `docs/plans/`），再逐步执行。

2. **实现与验证阶段**  
   - 若涉及多步实现：使用 `~/.cursor/skills/test-driven-development/SKILL.md`（先写失败测试，再最小实现）。  
   - 在声称“现有 LMC 不受影响”前：使用 `~/.cursor/skills/verification-before-completion/SKILL.md`，列出回归验证命令与通过标准，由用户本地执行并贴回结果后再收尾。

3. **可选**  
   - 若你希望本对话自动采用“3D 视觉 + 严谨实现”的身份准则，可先应用 `@identity-prompt-by-context` 中与路径/关键词匹配的 prompt。

---

## 关键代码与文档位置（供你检索）

- **当前 LMC**：`ace_depth/ace_compressor.py`、`ace_depth/ace_fusion.py`、`ace_depth/trainer_dinov2_lmc.py`（buffer、fusion 调用、backbone_feature_dim）、`ace_depth/options_dinov2_lmc.py`（lmc 相关选项）、`ace_depth/utils_lmc.py`；memory 来源见 map-anything 提取的 pooled 等。  
- **SceneTok 项目**：同工作区下 `scenetok/`（或用户指定路径）。重点：`scenetok/README.md`（方法概述）、`scenetok/src/model/compressor/`（compressor/perceiver）、`scenetok/src/model/autoencoder/`（VA-VAE 等）；SceneTok 将 view set 编码为压缩的、非结构化的 1D scene tokens，可与“全局 memory”对应。  
- **后续文档**：`docs/后续.md` 中“memory压缩一定要参考一下SceneTok，可以借鉴他的具体框架”等说明。  
- **工作流**：`docs/indoor6_multi_scene_workflow.md`（若 SceneTok memory 需按场景提取，需与现有 memory 提取流程对齐或扩展）。

---

## 设计要点（供 brainstorming 时细化）

- **SceneTok 接口**：encoder 输入（多视角图像/特征、相机）、输出（scene tokens 维度与长度）；是否使用官方 checkpoint、是否需要微调。  
- **Memory 对应**：现有 LMC 的 memory 是“每场景一个压缩表示”；SceneTok 的 scene tokens 是否一场景一组 token、如何与现有 scene-level 目录/文件约定对齐。  
- **Fusion**：query = 当前 backbone 特征（如 1024 维），memory = SceneTok tokens；cross-attention 或其它融合方式；与 `ace_fusion.py` 的接口兼容或新模块放在 `ace_scenetok/` 内。  
- **专用子目录结构**：建议至少包含 `ace_scenetok/model/`（或 `scenetok_integration/model/`）、`ace_scenetok/options.py`、`ace_scenetok/trainer.py` 或等价入口、`ace_scenetok/README.md`；根目录仅保留一个入口脚本（如 `train_ace_scenetok.py`）import 并调用子目录，不在根目录新增散落文件。

---

## 验收与回归（由用户本地执行）

- **回归**：在未启用 SceneTok 路径时，现有 `train_ace_dinov2_lmc.py`（或等价）训练/评估结果应与修改前一致（或仅在数值误差范围内）。  
- **新功能**：能通过新入口（如 `train_ace_scenetok.py` 或子目录内脚本）使用 SceneTok 作为 memory 完成至少一场景的训练与评估，且日志/指标可观察。  
- **代码布局**：所有新增 ace-scenetok 相关代码均在约定子目录内，根目录无新增散落文件；子目录内有清晰 README 说明依赖与用法。  
- 请你在完成实现后**给出上述回归与新功能的完整命令与期望输出/通过标准**，不要主动执行长时间验收脚本。

---

## 小结（复制到新对话时可直接用下面一段）

**任务**：用预训练好的 SceneTok 得到的场景非结构表示（压缩 scene tokens）作为全局 memory，与 query 特征融合以增强特征，从而替换/替代当前 LMC 模块；现有 DINO+LMC 路径保持可用。**强制**：所有 ace-scenetok 相关新代码必须放在专用子目录（如 `ace_depth/ace_scenetok/`），不得在 ace_depth 主目录下散落新文件。  
**要求**：先按 `brainstorming` 明确 SceneTok 接口、memory/fusion 对应关系与**子目录命名与结构**并得到确认，再按 `writing-plans` 或 TDD 拆解实现；验收与回归命令由你给出、用户本地执行。  
**参考**：`ace_compressor.py`、`ace_fusion.py`、`trainer_dinov2_lmc.py`、同工作区 `scenetok/` 项目（README、compressor、autoencoder）。

---

# English

## Your Role and Principles

- You are assisting a **3D vision / camera relocalization / depth estimation** deep learning project (ace_depth). Code rigor, mathematical correctness, and experiment reproducibility come first.
- **Evidence over claims**: Before stating that something is “fixed” or “correct,” provide runnable verification commands and expected outcomes. Do not run long acceptance or end-to-end scripts from this repo yourself; give the full commands for the user to run locally and paste back the results.
- **Design before coding**: Before replacing LMC or integrating SceneTok, clarify SceneTok’s token interface, how it maps to the current memory/fusion, and the **directory contract** for new code, then write code.

---

## Task Objectives (Follow Strictly)

1. **Replace the current LMC module with SceneTok**  
   - Current LMC: memory from depth/geometry (e.g. FPS, pooled point-cloud features) fused with query via cross-attention (see `ace_compressor.py`, `ace_fusion.py`, `trainer_dinov2_lmc.py`).  
   - Goal: Use the **unstructured scene representation** from the pretrained novel-view-synthesis method **SceneTok** (compressed, diffusable 1D scene tokens) as **global memory**, and fuse it with query features for feature enhancement; i.e. use SceneTok’s encoder/compressor output to replace or supplement the current LMC memory source.

2. **Keep other flows intact where possible**  
   - Main training flow (backbone, head, buffer, iteration) can stay as is; only the “memory source + fusion with query” is switched to a SceneTok-driven path; the existing DINO+LMC path remains available (e.g. via option or separate entrypoint).

3. **All new code in a dedicated subfolder (mandatory)**  
   - **Do not** add new ace-scenetok related files scattered under the ace_depth root (avoid repeating the current situation where LMC-related files are spread in the root).  
   - All **new** code for “ACE + SceneTok” integration must live in a **single dedicated subfolder**, e.g.:  
     - `ace_depth/ace_scenetok/` or `ace_depth/scenetok_integration/`  
   - That folder must be self-contained: e.g. `model/` (SceneTok adapter, fusion module), `options.py`, `trainer.py` or training entry, `README.md` for dependencies and usage; the root should only expose a thin entry script (e.g. `train_ace_scenetok.py` that imports from the subfolder).  
   - Exact subfolder name and internal layout must be agreed with the user in the brainstorming phase and documented in the writing-plans; implementation must follow them strictly.

4. **Do not break existing LMC behavior**  
   - Existing `train_ace_dinov2_lmc.py`, `trainer_dinov2_lmc.py`, `ace_compressor.py`, `ace_fusion.py` must behave unchanged when the SceneTok path is not used; run regression and pass before finishing.

---

## Skills You Must Use (In Order)

1. **Read and follow first**  
   - `~/.cursor/skills/using-superpowers/SKILL.md`: Ensure all skills relevant to this task are invoked.  
   - `~/.cursor/skills/brainstorming/SKILL.md`: Before writing code, explore ace_depth and SceneTok project structure; define (1) SceneTok input/output (view set → scene tokens), (2) mapping to current memory (token dim, sequence length, FPS/pooling if any), (3) fusion (cross-attention with query=backbone features, memory=scene tokens), (4) **name and internal layout of the dedicated subfolder**. Propose 2–3 options with a recommendation, get user approval, then implement.  
   - If you have an implementation plan, use `~/.cursor/skills/writing-plans/SKILL.md`: Break the work into bite-sized steps with **concrete subfolder paths and file list** (save under `docs/plans/`), then execute.

2. **Implementation and verification**  
   - For multi-step implementation: use `~/.cursor/skills/test-driven-development/SKILL.md` (write failing tests first, then minimal implementation).  
   - Before claiming “existing LMC unaffected”: use `~/.cursor/skills/verification-before-completion/SKILL.md` to list regression commands and pass criteria; the user runs them locally and pastes results before you finish.

3. **Optional**  
   - To adopt a “3D vision + rigorous implementation” identity, apply a matching prompt from `@identity-prompt-by-context`.

---

## Key Code and Doc Locations (For Reference)

- **Current LMC**: `ace_depth/ace_compressor.py`, `ace_depth/ace_fusion.py`, `ace_depth/trainer_dinov2_lmc.py` (buffer, fusion, backbone_feature_dim), `ace_depth/options_dinov2_lmc.py`, `ace_depth/utils_lmc.py`; memory source from map-anything extraction (pooled, etc.).  
- **SceneTok project**: `scenetok/` in the same workspace (or user-specified path). Focus: `scenetok/README.md` (method overview), `scenetok/src/model/compressor/`, `scenetok/src/model/autoencoder/`; SceneTok encodes view sets into compressed, unstructured 1D scene tokens, which can serve as “global memory.”  
- **Docs**: `docs/后续.md` (“memory压缩一定要参考一下SceneTok…”).  
- **Workflow**: `docs/indoor6_multi_scene_workflow.md` (align or extend if SceneTok memory is extracted per scene).

---

## Design Points (To Be Refined in Brainstorming)

- **SceneTok interface**: Encoder input (multi-view images/features, cameras), output (token dim and length); use official checkpoint or fine-tune.  
- **Memory mapping**: Current LMC memory is one compressed representation per scene; whether SceneTok tokens are one set per scene and how to align with scene-level paths.  
- **Fusion**: query = backbone features (e.g. 1024-dim), memory = SceneTok tokens; cross-attention or other; compatible with `ace_fusion.py` or new module under `ace_scenetok/`.  
- **Subfolder layout**: At least `ace_scenetok/model/`, `ace_scenetok/options.py`, `ace_scenetok/trainer.py` (or equivalent), `ace_scenetok/README.md`; root only has one entry script (e.g. `train_ace_scenetok.py`) that imports from the subfolder—no new scattered files in root.

---

## Acceptance and Regression (User Runs Locally)

- **Regression**: With the SceneTok path disabled, existing `train_ace_dinov2_lmc.py` (or equivalent) training/evaluation must match pre-change results (or within numerical tolerance).  
- **New feature**: A new entry (e.g. `train_ace_scenetok.py` or script in the subfolder) can run at least one scene using SceneTok as memory for training and evaluation, with observable logs/metrics.  
- **Code layout**: All new ace-scenetok code lives under the agreed subfolder; no new scattered files in root; subfolder has a clear README for dependencies and usage.  
- After implementation, **provide the exact regression and new-feature commands plus expected output / pass criteria**; do not run long acceptance scripts yourself.

---

## Summary (Copy-paste for a new conversation)

**Task**: Use the unstructured scene representation from pretrained SceneTok (compressed scene tokens) as global memory and fuse with query features for enhancement, replacing/supplementing the current LMC module; keep the existing DINO+LMC path available. **Mandatory**: All new ace-scenetok code must live in a dedicated subfolder (e.g. `ace_depth/ace_scenetok/`); no new scattered files in the ace_depth root.  
**Requirements**: Use `brainstorming` to fix SceneTok interface, memory/fusion mapping, and **subfolder name and structure** and get approval; then use `writing-plans` or TDD for implementation; you provide verification and regression commands for the user to run locally.  
**References**: `ace_compressor.py`, `ace_fusion.py`, `trainer_dinov2_lmc.py`, and the `scenetok/` project in the same workspace (README, compressor, autoencoder).
