---
name: dl-vision-experiment-orchestration
description: Use when coordinating multiple deep learning or vision experiments across scenes, datasets, GPUs, seeds, ablations, or evaluation phases. Emphasize scheduling, naming, provenance, and reproducibility.
---

# DL / Vision Experiment Orchestration

Use this skill when the task spans more than one run and needs disciplined organization.

## Workflow

1. Define the experiment matrix.
   Enumerate baselines, variants, datasets/scenes, seeds, and evaluation phases.
2. Define naming and output rules.
   Encode timestamp, scene, contract/preset, and major runtime knobs in a stable format.
3. Allocate resources.
   Map runs to GPUs, phases, and dependencies. Distinguish parallel-safe vs sequential stages.
4. Define result collection.
   Specify which logs, summaries, checkpoints, and artifacts count as the official outputs.
5. Define stop/resume policy.
   Document when to resume, restart, or mark a run as diagnostic only.

## Deliverables

- experiment matrix
- naming convention
- GPU scheduling plan
- result collection checklist
- stop/resume policy

## Rules

- Do not mix baseline and diagnostic runs without labeling them.
- Keep run names machine-parsable.
- Record dependencies between extraction, training, eval, and ensemble phases explicitly.
