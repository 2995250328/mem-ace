# Task Prompt: Replace DINO with ACE Original Backbone and Verify Logic Generality

**Copy this document as the first message in a new conversation to get the most out of the model and keep changes correct.**

---

## Your Role and Principles

- You are assisting a **3D vision / camera relocalization / depth estimation** deep learning project (ace_depth). Code rigor, mathematical correctness, and experiment reproducibility come first.
- **Evidence over claims**: Before stating that something is "fixed" or "logic unchanged," you must provide runnable verification commands and expected outcomes. Do not run long acceptance or end-to-end scripts from this repo yourself; instead give the full commands for the user to run locally and paste back the results.
- **Design before coding**: Before changing core architecture or adding a new backbone path, clarify the interface contract and impact, then write code.

---

## Task Objectives (Follow Strictly)

1. **Replace DINO with ACE original backbone**  
   - Current: backbone is DINOv2 (ViT-L/14, 1024-dim, patch 14, RGB input).  
   - Goal: Add an **optional** path that uses the **ACE original FCN encoder** (the `Encoder` in `ace_network_depth.py`: 512-dim, 8× downsampling, grayscale input, weights in `ace_encoder_pretrained.pt`) for the full training and testing pipeline.

2. **Keep all other flows unchanged**  
   - LMC (GeoLMC) iterative training, buffer filling, fusion, evaluation scripts, data format, and multi-scene workflow must **not** change in behavior; only the "backbone choice" (DINO vs ACE original) should switch.

3. **Verify generality of existing logic**  
   - Goal: Show that the current LMC / training / evaluation pipeline is backbone-agnostic and still runs and is comparable when using the ACE original backbone (vanilla vs LMC).

4. **Do not affect existing behavior**  
   - **Do not** remove or weaken existing DINOv2 code or entrypoints.  
   - Existing entrypoints (e.g. `train_ace_dinov2_lmc.py`, `train_ace_dinov2_iterative.py`, `test_ace_dinov2.py`, `test_ace_dinov2_lmc.py`) must behave exactly as before.  
   - Changes to existing files must be **minimal** and **backward compatible** (e.g. use `getattr(encoder, 'feature_dim', 1024)` so the DINO path is unchanged).

---

## Skills You Must Use (In Order)

1. **Read and follow first**  
   - `~/.cursor/skills/using-superpowers/SKILL.md`: Ensure all skills relevant to this task are invoked.  
   - `~/.cursor/skills/brainstorming/SKILL.md`: Before writing code, explore project context, define the backbone interface contract (input/output dimensions, resolution constraints), propose 2–3 implementation options with a recommendation, get user approval on the design, then implement.  
   - If you already have an implementation plan, use `~/.cursor/skills/writing-plans/SKILL.md`: Break the work into bite-sized steps with concrete file paths and test steps (save under `docs/plans/`), then execute step by step.

2. **Implementation and verification**  
   - For multi-step implementation: use `~/.cursor/skills/test-driven-development/SKILL.md` (write failing tests first, then minimal implementation).  
   - Before claiming "existing logic unaffected": use `~/.cursor/skills/verification-before-completion/SKILL.md` to list regression commands and pass criteria; the user runs them locally and pastes results before you finish.

3. **Optional**  
   - To adopt a "3D vision + rigorous implementation" identity for this conversation, apply a matching prompt from `@identity-prompt-by-context` (e.g. research-advisor-3dv or default).

---

## Key Code and Doc Locations (For Reference)

- **Current DINO wiring**: `ace_network_dinov2.py` (`DINOv2Encoder`, `Regressor`), `trainer_dinov2.py` (`TrainerACEDINOv2`), `trainer_dinov2_lmc.py` (`TrainerACEDINOv2LMC`, including `backbone_feature_dim`), `options_dinov2_lmc.py` (`dinov2_path`, `freeze_backbone`).  
- **ACE original encoder**: The `Encoder` class in `ace_network_depth.py` (grayscale input, 512-dim, 8× downsampling), and `ace_encoder_pretrained.pt`.  
- **Existing design reference**: `ace_depth/docs/替换ace原版backbone验证lmc_243f1789.plan.md` (interface comparison, "no impact on existing behavior" constraints, adapter design and steps).  
- **Workflow and data**: `ace_depth/docs/indoor6_multi_scene_workflow.md`.

---

## Interface Contract (Backbone Abstraction)

The new ACE-original path must be compatible with the existing LMC/training pipeline. Align the backbone interface with `DINOv2Encoder`:

- **Input**: `[B, C, H, W]` (if the ACE encoder is grayscale, do RGB→grayscale inside the adapter so callers can still pass RGB and reuse existing datasets).  
- **Output**: Feature map `[B, feature_dim, H', W']`, with the encoder exposing `feature_dim` and `patch_size` (or equivalent downsampling factor) for the trainer and fusion.  
- **Weights**: Load `ace_encoder_pretrained.pt` via `encoder_path` (or equivalent); do not hardcode the path.

---

## Acceptance and Regression (User Runs Locally)

- **Regression**: Run the existing DINOv2 + LMC or DINOv2 vanilla pipeline once with the same config (e.g. the usual commands for `train_ace_dinov2_lmc.py` or `train_ace_dinov2_iterative.py`); results must match the pre-change run (or only within numerical tolerance).  
- **New path**: The ACE-original backbone must support at least one scene for vanilla and (if applicable) LMC training and evaluation, with comparable metrics.  
- After implementation, **provide the exact regression and new-path commands plus expected output / pass criteria**; do not run long acceptance scripts yourself.

---

## Summary (Copy-paste for a new conversation)

**Task**: Add an optional "ACE original backbone" path in ace_depth, replacing DINO for that path while keeping all other flows (LMC, training, evaluation, data) unchanged to verify generality of the current logic; **do not** change existing DINO code or behavior.  
**Requirements**: Use `brainstorming` for design and interface agreement first, then `writing-plans` or TDD for implementation; keep edits to existing files minimal and backward compatible; you provide verification and regression commands for the user to run locally.  
**References**: `ace_depth/docs/替换ace原版backbone验证lmc_243f1789.plan.md`, `ace_network_dinov2.py`, `ace_network_depth.py`, `trainer_dinov2_lmc.py`.
