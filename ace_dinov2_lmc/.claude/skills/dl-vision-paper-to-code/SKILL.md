---
name: dl-vision-paper-to-code
description: Use when translating a deep learning or vision paper into implementation tasks, ablations, configs, and verification steps. Convert method sections into code contracts rather than loose summaries.
---

# DL / Vision Paper To Code

Use this skill when a paper, benchmark description, or method note must become an implementation plan.

## Workflow

1. Extract the method contract.
   Model architecture, input format, losses, augmentations, schedule, optimizer, normalization, and evaluation protocol.
2. Separate mandatory details from optional details.
   Core reproduction items vs improvements, conveniences, or engineering substitutions.
3. Map paper concepts to repository locations.
   Which parts belong in model code, data pipeline, training loop, eval logic, config presets, or reporting.
4. Build an ablation matrix.
   Baseline, target method, reduced variants, stress tests, and expected directional outcomes.
5. Define verification criteria.
   Metrics, plots, qualitative outputs, and sanity checks required to claim the implementation is correct.

## Deliverables

- implementation checklist
- repository mapping
- ablation matrix
- verification plan

## Rules

- Do not treat the paper abstract as an implementation guide.
- Record image resolution, augmentation, and metric protocol explicitly for vision papers.
- If a detail is missing, mark it as unresolved instead of guessing silently.
