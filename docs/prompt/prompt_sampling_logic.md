# 采样逻辑 / Sampling Logic — 任务 Prompt（中英）

**在新对话中复制对应语言段落作为第一条消息，以充分发挥模型能力并保证修改质量。**

---

# 中文

## 你的角色与准则

- 你正在协助一个**三维视觉 / 相机重定位 / 深度估计**的深度学习项目（ace_depth）。代码严谨性、数学正确性与实验可复现性优先。
- **证据优于断言**：在声称“修复完成”或“逻辑正确”前，必须有可执行的验证命令与期望结果说明；不主动运行本仓库中的长时间验收/端到端脚本，而是给出完整命令由用户本地执行后贴回结果。
- **谋定而后动**：在改训练流程或新增采样模块前，先理清“真值”定义、训练阶段划分与采样模块的接口，再动手写代码。

---

## 任务目标（请严格遵循）

1. **基于原版 ACE 进行修改，先验证逻辑可行性**  
   - 本任务以**原版 ACE**（`ace_network_depth.py`、`train_ace.py`、`ace_trainer.py`，灰度、512 维、8× 下采样）为基础，不依赖 DINO/LMC。目标先验证“用重投影误差训练采样器并用于 buffer 采样”这一逻辑是否可行。

2. **训练流程分两阶段**  
   - **阶段一（单独训练采样器）**：不使用 buffer，**直接端到端（e2e）训练**采样模块。**数据**：必须使用**训练集**（与主训练同一场景的 train 划分）。即：用已训练好的 ACE 模型在（可加噪）原图上推理 → 计算重投影误差图 → 以“重投影误差小”的点作为稳定点真值/权重，监督训练一个采样器网络（输出每点置信度/权重），e2e 一个阶段内完成，无需先填 buffer。是否复用原版 ACE 训练时的**数据增强**（如旋转、缩放、aug_rotation / aug_scale 等）及相同的数据加载方式，在设计中明确并作为可选项实现，以保持与主训练的数据分布一致。  
   - **阶段二（接入 buffer 收集）**：将**训练好的采样器**用来**替换** buffer 收集过程中的**随机采样**。在填充 buffer 时，依据采样器的输出（高置信度/高权重的点）来选择要采样的位置，从而让 buffer 中更多是“稳定且有意义的点”；原有随机采样作为 fallback 或可选项保留。

3. **采样器接入时的可选逻辑：邻域采样等**  
   - 在阶段二“用采样器替换随机采样”时，可**参考** `ace_trainer_full.py` 中的 **SuperPoint 采样逻辑**，将其中**邻域扩展、过滤与补充采样**等作为**可选项**保留。例如：  
     - 类似 `expand_neighbors_gpu(coords, H, W, patch_size, include_diagonal)` 的邻域扩展（在采样器给出的高置信度点周围再采 patch 中心），以增加局部覆盖；  
     - 类似 Part A（高置信度/SuperPoint 类点）+ Part B（按策略补充或随机补充）的两段式结构，使“采样器主导 + 补充”可配置；  
   - 具体是否启用邻域扩展、Part A/B 比例等，在设计中明确并作为选项暴露。

4. **所有修改放在单独子文件夹下（强制）**  
   - **禁止**在 ace_depth 主目录下散落本任务新增的采样器/训练脚本。所有与“ACE 原版 + 可学习采样器”相关的**新代码**必须放在一个**专用子文件夹**内，例如：  
     - `ace_depth/ace_sampler/` 或 `ace_depth/sampling_module/`  
   - 该目录内自洽组织：例如 `model/`（采样器网络）、`train_sampler_e2e.py`（阶段一 e2e 训练）、`options.py`、与 buffer 填充的对接接口（供原版 ACE trainer 或独立 buffer 脚本调用）、`README.md`；根目录至多保留一个薄入口脚本（如 `train_ace_sampler.py`）调用子目录。子目录名称与内部结构在 brainstorming 阶段与用户确认，并在设计文档或对话中写明。

5. **不破坏现有流程**  
   - 现有原版 ACE 训练（`train_ace.py`）、DINO/ACE LMC 等流程保持可用；新采样逻辑通过**子目录内入口 + 可选开关**使用，默认行为与当前一致。

---

## 你必须使用的 Skills（按顺序）

1. **先阅读并执行**  
   - `~/.cursor/skills/using-superpowers/SKILL.md`：确认本任务涉及到的 skills 均被调用。  
   - `~/.cursor/skills/brainstorming/SKILL.md`：在写代码前，先探索项目上下文、明确（1）阶段一 e2e 的训练数据与 loss（重投影误差→稳定点真值/权重）、（2）采样器输入/输出与阶段二如何替换 buffer 内随机采样、（3）是否采用邻域扩展与 Part A/B 结构及可选接口、（4）**专用子目录的命名与内部模块划分**，提出 2–3 种实现方式并给出推荐，形成设计并得到用户确认后再实现。  
   - 实现时将任务拆成带**具体子目录路径与文件清单**的步骤，在文档或对话中写明后逐步执行（勿使用 writing-plans skill）。

2. **实现与验证阶段**  
   - 若涉及多步实现：使用 `~/.cursor/skills/test-driven-development/SKILL.md`（先写失败测试，再最小实现）。  
   - 在声称“现有逻辑不受影响”前：使用 `~/.cursor/skills/verification-before-completion/SKILL.md`，列出回归验证命令与通过标准，由用户本地执行并贴回结果后再收尾。

3. **可选**  
   - 若你希望本对话自动采用“3D 视觉 + 严谨实现”的身份准则，可先应用 `@identity-prompt-by-context` 中与路径/关键词匹配的 prompt。

---

## 关键代码与文档位置（供你检索）

- **原版 ACE**：`ace_network_depth.py`（`Encoder`、`Regressor`）、`train_ace.py`、`ace_trainer.py`（原版 ACE 训练与模型加载、**训练集**使用方式与**数据增强**如 aug_rotation / aug_scale / use_aug）、`ace_encoder_pretrained.pt`。  
- **SuperPoint 与邻域采样（可参考）**：`ace_trainer_full.py` 中的 `expand_neighbors_gpu()`（邻域扩展）、`create_training_buffer()` 的 Part A（SuperPoint 采样 + 邻域扩展 + sky 过滤）与 Part B（按 strategy 权重补充）、`get_strategy_mask()`；`superpoint.py`（SuperPointNet / SuperPointFrontend）。  
- **重投影误差**：`ace_trainer_full.py` 中的 `compute_reprojection_error_map()`（ACE 前向 → 场景坐标 → 重投影 → 误差图）；坐标系与单位需在注释中明确。  
- **Buffer 与采样（若对接现有 buffer 流程）**：原版 ACE 的 buffer 填充逻辑在 `ace_trainer.py`；DINO 侧为 `options_dinov2_lmc.py`（`samples_per_image`）、`trainer_dinov2.py` / `trainer_dinov2_lmc.py`。本任务以原版 ACE 为主，buffer 对接可在子目录内封装。  
- **工作流**：`docs/indoor6_multi_scene_workflow.md`。  
- **训练好的权重**：保存在 `ace_depth/output/ace_models` 中；阶段一采样器 checkpoint 与阶段二加载路径可约定置于该目录或子目录下。

---

## 设计要点（供 brainstorming 时细化）

- **阶段一 e2e**：**数据**必须为**训练集**（与主训练同场景 train 划分）；数据循环（逐图或小 batch）、ACE 推理、重投影误差图、真值/权重（阈值或 exp(-error)）、采样器 loss（二分类或回归权重）；无需 buffer，直接 e2e 训练采样器。**推荐**：采样器输出为 **per-pixel 置信度图（float 0–1）**，监督目标用 exp(-α·error) ，用 MSE 算 loss，便于平滑梯度传播；阶段二可用该置信度做 top-k 或加权采样。**数据增强**：可复用原版 ACE 训练时的增强（如 `train_ace.py` / `ace_trainer.py` 中的 aug_rotation、aug_scale、use_aug 等），作为可选项实现，以与主训练数据分布一致。  
- **阶段二对接**：训练好的采样器如何被调用（输入图像/特征，输出 per-point 置信度或权重）；在 buffer 填充时如何用该输出替代随机采样。**推荐**：采用 **Part A (sampler top-k) + Part B (random supplement)** 结构，Part A 取采样器高置信度点（可选邻域扩展），Part B 随机补充剩余配额，比例由参数（如 `--sampler_ratio`）控制；这与现有 SuperPoint 的 Part A/B 结构最契合，也更稳健。  
- **邻域与 Part A/B（可选）**：是否在采样器给出的点上做邻域扩展（类似 `expand_neighbors_gpu`）、Part A（采样器高置信度点）+ Part B（随机/策略补充）的比例与开关；参考 `ace_trainer_full.py` 的 `sp_ratio`、`expand_neighbors_gpu(..., include_diagonal=4 或 8)`。  
- **专用子目录结构**：例如 `ace_sampler/model/`、`ace_sampler/train_sampler_e2e.py`、`ace_sampler/options.py`、`ace_sampler/buffer_sampler_impl.py`（供外部调用的采样接口）、`ace_sampler/README.md`；根目录仅保留一个入口脚本。

---

## 验收与回归（由用户本地执行）

- **回归**：在未启用新采样逻辑时，现有原版 ACE 训练/评估（如 `train_ace.py`）结果应与修改前一致（或仅在数值误差范围内）。  
- **新功能**：  
  - 阶段一：能通过子目录内入口（如 `train_sampler_e2e.py`）在不使用 buffer 的前提下 e2e 训练采样器，并得到可用的 checkpoint。  
  - 阶段二：能使用训练好的采样器替换 buffer 收集时的随机采样，完成至少一场景的 buffer 填充或训练，且日志/指标可观察（如采样点分布、loss）。  
- **代码布局**：所有新增代码均在约定子目录内，根目录无新增散落文件；子目录内有清晰 README 说明两阶段用法与依赖。  
- 请你在完成实现后**给出上述回归与新功能的完整命令与期望输出/通过标准**，不要主动执行长时间验收脚本。

---

## 小结（复制到新对话时可直接用下面一段）

**任务**：基于**原版 ACE**，先验证“用重投影误差训练采样器并用于 buffer 采样”的逻辑可行性。**训练流程**：（1）阶段一：单独训练采样器，**不使用 buffer**，直接 **e2e 训练**；（2）阶段二：用训练好的采样器**替换** buffer 收集过程的**随机采样**，按采样器输出选择高置信度点；可**可选**参考 `ace_trainer_full.py` 的 SuperPoint 采样逻辑，保留邻域扩展（如 `expand_neighbors_gpu`）与 Part A/B 补充结构。**强制**：所有新代码放在专用子目录（如 `ace_depth/ace_sampler/`），不在主目录散落。  
**要求**：先按 `brainstorming` 明确两阶段数据/loss/接口、邻域与 Part A/B 可选设计及子目录结构并得到确认，再按 TDD 或分步实现拆解；验收与回归命令由你给出、用户本地执行。  
**参考**：`ace_network_depth.py`、`ace_trainer.py`、`ace_trainer_full.py`（SuperPoint、`expand_neighbors_gpu`、`compute_reprojection_error_map`、Part A/B 采样结构）、与重投影/场景坐标相关的 loss 与评估代码。

---

# English

## Your Role and Principles

- You are assisting a **3D vision / camera relocalization / depth estimation** deep learning project (ace_depth). Code rigor, mathematical correctness, and experiment reproducibility come first.
- **Evidence over claims**: Before stating that something is “fixed” or “correct,” provide runnable verification commands and expected outcomes. Do not run long acceptance or end-to-end scripts from this repo yourself; give the full commands for the user to run locally and paste back the results.
- **Design before coding**: Before changing the training pipeline or adding a sampling module, clarify the definition of “ground truth,” the two-phase training flow, and the sampling module’s interface, then write code.

---

## Task Objectives (Follow Strictly)

1. **Base changes on original ACE; verify feasibility first**  
   - This task is based on **original ACE** (`ace_network_depth.py`, `train_ace.py`, `ace_trainer.py`, grayscale, 512-dim, 8× downsampling), not DINO/LMC. The goal is to first verify that the logic “train a sampler from reprojection error and use it for buffer sampling” is feasible.

2. **Two-phase training flow**  
   - **Phase 1 (train sampler only)**: Do **not** use a buffer; train the sampling module **end-to-end (e2e)**. **Data**: must use the **training set** (same scene train split as main ACE training). That is: run a trained ACE model on (optionally noisy) images → compute reprojection error map → treat points with “small reprojection error” as stable-point ground truth/weights → supervise a sampler network (outputting per-point confidence/weight) in a single e2e phase, with no buffer filling. Whether to reuse **data augmentation** from original ACE training (e.g. rotation, scale, aug_rotation / aug_scale) and the same data-loading setup should be decided in design and implemented as an option, so the sampler sees the same data distribution as main training.  
   - **Phase 2 (plug into buffer collection)**: Use the **trained sampler** to **replace** **random sampling** during buffer collection. When filling the buffer, select positions according to the sampler’s output (high-confidence / high-weight points) so that the buffer contains more “stable and meaningful points”; keep original random sampling as fallback or optional.

3. **Optional logic when plugging in: neighborhood sampling, etc.**  
   - In Phase 2, when replacing random sampling with the sampler, **reference** the **SuperPoint sampling logic** in `ace_trainer_full.py` and keep **neighborhood expansion, filtering, and supplement sampling** as **optional**. For example:  
     - Neighborhood expansion like `expand_neighbors_gpu(coords, H, W, patch_size, include_diagonal)` around sampler-predicted high-confidence points to improve local coverage;  
     - A two-part structure similar to Part A (high-confidence / SuperPoint-like points) + Part B (strategy-based or random supplement), so that “sampler-led + supplement” is configurable;  
   - Whether to enable neighborhood expansion, Part A/B ratio, etc., must be explicit in the design and exposed as options.

4. **All new code in a dedicated subfolder (mandatory)**  
   - **Do not** add new sampler/training files scattered under the ace_depth root. All **new** code for “original ACE + learnable sampler” must live in a **single dedicated subfolder**, e.g.:  
     - `ace_depth/ace_sampler/` or `ace_depth/sampling_module/`  
   - That folder must be self-contained: e.g. `model/` (sampler network), `train_sampler_e2e.py` (Phase 1 e2e training), `options.py`, and a buffer-fill interface for the original ACE trainer or a standalone buffer script, plus `README.md`; the root may expose at most one thin entry script (e.g. `train_ace_sampler.py`) that calls into the subfolder. Subfolder name and internal layout must be agreed with the user in the brainstorming phase and documented in the design or in chat.

5. **Do not break existing flows**  
   - Existing original ACE training (`train_ace.py`), DINO/ACE LMC, etc. must remain usable; the new sampling logic is used via **subfolder entry + optional switch**, with default behavior unchanged.

---

## Skills You Must Use (In Order)

1. **Read and follow first**  
   - `~/.cursor/skills/using-superpowers/SKILL.md`: Ensure all skills relevant to this task are invoked.  
   - `~/.cursor/skills/brainstorming/SKILL.md`: Before writing code, explore project context and define (1) Phase 1 e2e data and loss (reprojection error → stable-point labels/weights), (2) sampler inputs/outputs and how Phase 2 replaces random sampling inside buffer filling, (3) whether to use neighborhood expansion and Part A/B structure and their optional interface, (4) **name and internal layout of the dedicated subfolder**. Propose 2–3 options with a recommendation, get user approval, then implement.  
   - When implementing, break the work into steps with **concrete subfolder paths and file list**, document them in a file or in chat, then execute step by step (do not use the writing-plans skill).

2. **Implementation and verification**  
   - For multi-step implementation: use `~/.cursor/skills/test-driven-development/SKILL.md` (write failing tests first, then minimal implementation).  
   - Before claiming “existing logic unaffected”: use `~/.cursor/skills/verification-before-completion/SKILL.md` to list regression commands and pass criteria; the user runs them locally and pastes results before you finish.

3. **Optional**  
   - To adopt a “3D vision + rigorous implementation” identity, apply a matching prompt from `@identity-prompt-by-context`.

---

## Key Code and Doc Locations (For Reference)

- **Original ACE**: `ace_network_depth.py` (`Encoder`, `Regressor`), `train_ace.py`, `ace_trainer.py` (original ACE training and model loading, **training set** usage and **data augmentation** e.g. aug_rotation / aug_scale / use_aug), `ace_encoder_pretrained.pt`.  
- **SuperPoint and neighborhood sampling (reference)**: In `ace_trainer_full.py`, `expand_neighbors_gpu()` (neighborhood expansion), `create_training_buffer()` Part A (SuperPoint sampling + neighborhood expansion + sky filtering) and Part B (strategy-weighted supplement), `get_strategy_mask()`; `superpoint.py` (SuperPointNet / SuperPointFrontend).  
- **Reprojection error**: `ace_trainer_full.py`’s `compute_reprojection_error_map()` (ACE forward → scene coords → reproject → error map); document coordinate frame and units in comments.  
- **Buffer and sampling (if wiring to existing buffer flow)**: Original ACE buffer filling is in `ace_trainer.py`; DINO side: `options_dinov2_lmc.py` (`samples_per_image`), `trainer_dinov2.py` / `trainer_dinov2_lmc.py`. This task is centered on original ACE; buffer wiring can be encapsulated inside the subfolder.  
- **Workflow**: `docs/indoor6_multi_scene_workflow.md`.  
- **Trained weights**: Saved under `ace_depth/output/ace_models`; Phase 1 sampler checkpoints and Phase 2 load paths can be placed under this directory or a subdirectory thereof.

---

## Design Points (To Be Refined in Brainstorming)

- **Phase 1 e2e**: **Data** must be the **training set** (same scene train split as main training); data loop (per image or small batch), ACE inference, reprojection error map, labels/weights (threshold or exp(-error)), sampler loss (binary or weight regression); no buffer, train sampler e2e directly. **Recommended**: Sampler outputs a **per-pixel confidence map (float 0–1)** with target exp(-α·error) using MSE loss for smooth gradients; Phase 2 can use it for top-k or weighted sampling. **Data augmentation**: Optionally reuse original ACE training augmentation (e.g. aug_rotation, aug_scale, use_aug in `train_ace.py` / `ace_trainer.py`) as an option so the sampler matches main training data distribution.  
- **Phase 2 integration**: How the trained sampler is called (input image/features, output per-point confidence or weight); how buffer filling uses this to replace random sampling. **Recommended**: Use **Part A (sampler top-k) + Part B (random supplement)** where Part A takes sampler high-confidence points (with optional neighborhood expansion) and Part B randomly fills the remaining quota, controlled by a ratio (e.g. `--sampler_ratio`); this mirrors the SuperPoint Part A/B structure and is more robust.  
- **Neighborhood and Part A/B (optional)**: Whether to expand neighborhood around sampler points (like `expand_neighbors_gpu`), Part A (sampler high-confidence) + Part B (random/strategy supplement) ratio and switch; reference `ace_trainer_full.py`’s `sp_ratio`, `expand_neighbors_gpu(..., include_diagonal=4 or 8)`.  
- **Subfolder layout**: e.g. `ace_sampler/model/`, `ace_sampler/train_sampler_e2e.py`, `ace_sampler/options.py`, `ace_sampler/buffer_sampler_impl.py` (sampling API for external call), `ace_sampler/README.md`; root keeps only one entry script.

---

## Acceptance and Regression (User Runs Locally)

- **Regression**: With the new sampling logic disabled, existing original ACE training/evaluation (e.g. `train_ace.py`) must match pre-change results (or within numerical tolerance).  
- **New feature**:  
  - Phase 1: A subfolder entry (e.g. `train_sampler_e2e.py`) can train the sampler e2e **without** using a buffer and produce a usable checkpoint.  
  - Phase 2: The trained sampler can replace random sampling during buffer collection for at least one scene (buffer fill or training), with observable logs/metrics (e.g. sampling point distribution, loss).  
- **Code layout**: All new code lives under the agreed subfolder; no new scattered files in root; subfolder has a clear README for both phases and dependencies.  
- After implementation, **provide the exact regression and new-feature commands plus expected output / pass criteria**; do not run long acceptance scripts yourself.

---

## Summary (Copy-paste for a new conversation)

**Task**: Based on **original ACE**, first verify that the logic “train a sampler from reprojection error and use it for buffer sampling” is feasible. **Training flow**: (1) Phase 1: Train the sampler **alone**, **without buffer**, with **e2e training**; (2) Phase 2: Use the trained sampler to **replace random sampling** in buffer collection and select high-confidence points from the sampler output; **optionally** reference `ace_trainer_full.py`’s SuperPoint sampling (e.g. `expand_neighbors_gpu`, Part A/B supplement structure). **Mandatory**: All new code in a dedicated subfolder (e.g. `ace_depth/ace_sampler/`), no scattered files in root.  
**Requirements**: Use `brainstorming` to fix Phase 1/2 data/loss/interface, optional neighborhood and Part A/B design, and subfolder layout and get approval; then use TDD or stepwise implementation; you provide verification and regression commands for the user to run locally.  
**References**: `ace_network_depth.py`, `ace_trainer.py`, `ace_trainer_full.py` (SuperPoint, `expand_neighbors_gpu`, `compute_reprojection_error_map`, Part A/B sampling structure), and loss/evaluation code for reprojection and scene coordinates.
