---
name: dl-vision-checkpoint-resume-integrity
description: Use when a deep learning or vision project saves, resumes, swaps, or compares checkpoints. Verify checkpoint schema, optimizer/scheduler state, scaler state, iteration counters, and best-model semantics before trusting resumed results.
---

# DL / Vision Checkpoint And Resume Integrity

Use this skill whenever checkpoint handling affects correctness.

## Workflow

1. Identify checkpoint types.
   Best, latest, iter, epoch, EMA, post-train, ensemble, or partial-stage checkpoints.
2. Inspect schema expectations.
   Model state, optimizer state, scheduler state, scaler state, RNG state, iteration/epoch counters, and metadata.
3. Verify resume semantics.
   Determine which states are restored and which are intentionally reset.
4. Verify output semantics.
   Confirm whether resumed runs overwrite, append, branch, or create a new run directory.
5. Define trust level.
   Decide whether resumed metrics are directly comparable to uninterrupted runs.

## Output

- checkpoint type map
- schema compatibility note
- resume correctness note
- comparability note

## Rules

- Do not assume loading model weights implies true training resume.
- Separate “fine-tune from weights” from “resume optimizer/scheduler state”.
- If best-checkpoint logic changes after resume, call that out explicitly.
