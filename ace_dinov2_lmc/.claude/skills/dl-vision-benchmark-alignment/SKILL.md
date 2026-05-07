---
name: dl-vision-benchmark-alignment
description: Use when comparing a deep learning or vision repository against a paper, benchmark, baseline, or prior internal result. Normalize data protocol, metric definitions, runtime conditions, and reporting rules before drawing conclusions.
---

# DL / Vision Benchmark Alignment

Use this skill when users want fair comparison instead of raw numbers only.

## Workflow

1. Identify the benchmark contract.
   Dataset split, preprocessing, input resolution, augmentation, metric, and evaluation protocol.
2. Identify the repository contract.
   Actual train/eval code paths, defaults, and reporting semantics.
3. Compare the two contracts.
   List all mismatches that affect fairness.
4. Define acceptable equivalence.
   Decide which deviations are harmless and which invalidate comparison.
5. Produce a normalized comparison note.

## Output

- benchmark contract
- repository contract
- mismatch list
- fair-comparison judgement

## Rules

- Never compare runs across different metric definitions silently.
- Surface evaluation-time differences separately from training-time differences.
- If comparison is only approximate, label it approximate.
