---
name: dl-vision-repo-workflow
description: Use when working with an unfamiliar deep learning or computer vision repository. Build a fast understanding of entrypoints, dataset/config contracts, train/eval flows, outputs, and the minimum reproducible commands before changing code.
---

# DL / Vision Repo Workflow

Use this skill when the codebase is a training, evaluation, or inference repo for deep learning, especially vision-heavy projects.

## Objectives

- identify the real train / eval / inference entrypoints
- map dataset, config, checkpoint, and output contracts
- find where metrics are produced and persisted
- produce a minimal runnable workflow before deeper changes

## Workflow

1. Find primary entrypoints.
   Look for `train*`, `test*`, `eval*`, `infer*`, `options*`, `trainer*`, and dataset modules.
2. Map config surfaces.
   Record CLI flags, presets, config files, env vars, device flags, batch-size knobs, and output directory rules.
3. Map data contracts.
   Identify scene / dataset roots, annotation formats, split conventions, expected folder layouts, and loader backends.
4. Map training flow.
   Locate model build, optimizer/scheduler setup, checkpoint save logic, eval triggers, and resume behavior.
5. Map evaluation flow.
   Identify which script writes which metric files, what counts as best, and how summaries differ from raw logs.
6. Produce a runbook.
   Write the shortest commands that can reproduce train, eval, and result lookup.

## Deliverables

- one short architecture summary
- one table of key entry files and their responsibilities
- one minimal command set for train / eval / inspect results
- one list of high-risk assumptions or missing prerequisites

## Checks

- Do not assume `README` is correct. Prefer code and actual paths.
- Prefer exact file references over narrative summaries.
- Distinguish between current project behavior and upstream / intended behavior.
- If multiple logs disagree, note which one is authoritative and why.

## Strong Patterns

- Use semantic code search if available, but verify with file reads.
- Treat dataset path handling and output path handling as first-class contracts.
- When training is iterative, note which metrics are per-iter, post-train, or best-checkpoint only.
