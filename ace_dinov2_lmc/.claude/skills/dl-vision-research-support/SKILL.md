---
name: dl-vision-research-support
description: Use when a deep learning or vision task depends on papers, benchmarks, ablations, or method reproduction. Turn literature into implementation plans, experiment matrices, and structured technical summaries.
---

# DL / Vision Research Support

Use this skill for paper reading, benchmark alignment, method reproduction, ablation planning, and connecting literature claims to code changes.

## Workflow

1. Clarify the research objective.
   Is the user asking for reproduction, comparison, adaptation, or literature survey?
2. Extract the method contract.
   Identify model family, data assumptions, losses, augmentation, training schedule, metrics, and evaluation protocol.
3. Translate paper language into implementation tasks.
   Separate what is architecture, data, optimization, runtime, and reporting.
4. Build an experiment matrix.
   Define baseline, target variant, ablations, expected outputs, and stopping conditions.
5. Align reporting.
   Ensure plots, tables, and qualitative examples reflect the benchmark or paper convention.

## Deliverables

- one implementation checklist
- one experiment matrix
- one benchmark/metric note
- one list of open technical uncertainties

## Rules

- Do not treat paper claims as implementation facts until matched to code or appendix details.
- Separate mandatory reproduction details from optional improvements.
- When the paper is vision-heavy, include dataset splits, image resolution, augmentation, and metric protocol explicitly.
