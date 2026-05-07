---
name: dl-vision-experiment-reporting
description: Use when aggregating deep learning experiment results from training logs, eval logs, summaries, checkpoints, or run directories. Normalize metric sources, compare runs, and produce reproducible summaries with exact file provenance.
---

# DL / Vision Experiment Reporting

Use this skill when experiment outputs are spread across multiple logs and summary files and the user needs a reliable comparison.

## Workflow

1. Inventory the run directories.
   Capture experiment name, timestamp, device, checkpoint names, and sibling summary files.
2. Identify metric sources.
   Distinguish raw training logs, eval logs, per-iteration summaries, post-train summaries, and final tables.
3. Normalize the metric definition.
   Confirm exact names, units, directionality, and whether a metric is best-max or best-min.
4. Resolve conflicts.
   If multiple files disagree, prefer the file closest to the actual evaluation step and note the discrepancy.
5. Produce best-of reporting.
   State whether “best” means best checkpoint, best iteration, post-train best, or best across all saved summaries.
6. Preserve provenance.
   Every conclusion should point to exact files, not memory.

## Output

- one concise comparison table
- one source note per run explaining which files were scanned
- one short note on anomalies, missing files, or metric-definition changes

## Rules

- Do not mix incompatible metric definitions across repos or phases.
- If the user asks for “latest” or “best”, specify the exact timestamp or iteration.
- Prefer exact paths and exact filenames.
- If a run was interrupted, say so explicitly instead of silently excluding it.
