---
name: ace-result-aggregation
description: Use when aggregating ACE-DINOv2-LMC experiment results across training logs, eval logs, saved summaries, and reference markdown tables. Focus on best-of selection, source consistency, and exact run provenance.
---

# ACE Result Aggregation

Use this skill for result collation in this repository.

## Scope

- `*_eval_log.txt`
- `eval_summary_*.txt`
- training logs
- checkpoint/output directories
- result markdown files under `memory_extraction/03_implementation`

## Workflow

1. Inventory all result sources for a run.
   Check the experiment directory for all eval logs and summary files before summarizing.
2. Normalize the aggregation rule.
   State whether the run uses “best across `*_eval_log.txt` and `eval_summary_*.txt`” or another rule.
3. Resolve best values carefully.
   Identify which file and iteration produced each best metric.
4. Compare against references.
   Update or check repository markdown tables and progress notes using the same aggregation rule.
5. Preserve provenance.
   Every table row should be traceable to exact files and iterations.

## Output

- best-of metrics
- source note
- comparison against baseline/reference runs
- update checklist for result docs

## Rules

- Never summarize a run from one file if the directory contains multiple relevant eval artifacts.
- If a run is incomplete, mark it incomplete instead of silently ranking it.
- Use exact experiment directory names when comparing similar runs.
