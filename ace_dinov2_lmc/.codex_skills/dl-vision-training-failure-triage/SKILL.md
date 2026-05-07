---
name: dl-vision-training-failure-triage
description: Use when a deep learning or vision training job crashes, hangs, diverges, or resumes incorrectly. Triage failures systematically across config, data, device placement, distributed runtime, checkpointing, and metric regressions.
---

# DL / Vision Training Failure Triage

Use this skill when training fails before start, during warmup, mid-epoch, at evaluation, or during resume.

## Workflow

1. Reconstruct the failing command.
2. Locate the first authoritative error.
3. Classify the stage.
4. Check contract mismatches.
5. Decide whether the failure is deterministic.
6. Propose the narrowest safe fix.

## Rules

- Do not summarize a long log before finding the first real exception.
- Distinguish crash from divergence from stall.
- For resume issues, check optimizer, scheduler, scaler, iteration counters, and output directory semantics separately.
