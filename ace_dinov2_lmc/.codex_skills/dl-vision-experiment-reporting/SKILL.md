---
name: dl-vision-experiment-reporting
description: Use when aggregating deep learning experiment results from training logs, eval logs, summaries, checkpoints, or run directories. Normalize metric sources, compare runs, and produce reproducible summaries with exact file provenance.
---

# DL / Vision Experiment Reporting

Use this skill when experiment outputs are spread across multiple logs and summary files and the user needs a reliable comparison.

## Workflow

1. Inventory the run directories.
2. Identify metric sources.
3. Normalize the metric definition.
4. Resolve conflicts.
5. Produce best-of reporting.
6. Preserve provenance.

## Rules

- Do not mix incompatible metric definitions across repos or phases.
- If the user asks for “latest” or “best”, specify the exact timestamp or iteration.
- Prefer exact paths and exact filenames.
- If a run was interrupted, say so explicitly instead of silently excluding it.
