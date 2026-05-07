---
name: dl-vision-training-failure-triage
description: Use when a deep learning or vision training job crashes, hangs, diverges, or resumes incorrectly. Triage failures systematically across config, data, device placement, distributed runtime, checkpointing, and metric regressions.
---

# DL / Vision Training Failure Triage

Use this skill when training fails before start, during warmup, mid-epoch, at evaluation, or during resume.

## Failure Classes

- startup failure
- dataloader / dataset failure
- CUDA / device mismatch
- OOM / fragmentation
- NaN / Inf divergence
- distributed deadlock or partial worker failure
- checkpoint / resume corruption
- eval-time failure after successful training steps

## Workflow

1. Reconstruct the failing command.
   Capture exact CLI, env vars, device mapping, output directory, and timestamp.
2. Locate the first authoritative error.
   Prefer the earliest real stack trace over wrapper noise.
3. Classify the stage.
   Startup, data load, forward, backward, optimizer step, checkpoint save, eval, or resume.
4. Check contract mismatches.
   Paths, tensor devices, shapes, dtypes, generator devices, checkpoint schema, and metric expectations.
5. Decide whether the failure is deterministic.
   Reproducible every run, data-dependent, iteration-dependent, or hardware/load dependent.
6. Propose the narrowest safe fix.
   Prefer the minimal code or flag change that resolves the specific failure mode.

## Output

- failing stage
- root cause hypothesis
- exact evidence
- minimal fix
- verification command

## Rules

- Do not summarize a long log before finding the first real exception.
- Distinguish crash from divergence from stall.
- For resume issues, check optimizer, scheduler, scaler, iteration counters, and output directory semantics separately.
- If the failure happened after a user throughput tweak, separate runtime issues from training-dynamics issues.
