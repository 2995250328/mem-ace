---
name: agent-skill-orchestration
description: Use when a request is ambiguous, costly, multi-step, research-heavy, experiment-heavy, debugging-heavy, or likely to benefit from structured requirement clarification, retrieval, planning, critique, and verification. This is the default routing skill for choosing and sequencing multiple skills automatically instead of answering in one shot.
---

# Agent Skill Orchestration

Use this skill when the task should be handled as a deliberate multi-skill workflow rather than a single-shot answer.

## Routing Rule

If there is a reasonable chance the task would benefit from more than one specialized skill, use this skill first to choose the chain.

## Core Principle

Do not jump straight from request to implementation when the task is ambiguous, high-cost, cross-file, research-heavy, or evaluation-sensitive.

Instead, deliberately compose skills in phases:

1. clarify
2. retrieve
3. plan
4. challenge
5. execute
6. verify

## Default Chains

### 1. Requirement Clarification / Design Thinking

Use this chain when the request is underspecified, has multiple possible solutions, or needs stronger framing before action:

- `superpowers-using-superpowers`
- `superpowers-brainstorming`
- `superpowers-writing-plans` if implementation will follow
- `grill-me` to pressure-test the chosen plan
- `karpathy` while converting the plan into concrete work

### 2. Research / Retrieval / Technical Judgment

Use this when the task needs literature, codebase lookup, benchmark context, or synthesis:

- `dl-vision-repo-workflow`
- `paper-lookup`
- `literature-review`
- `dl-vision-paper-to-code`
- `grill-me`
- `karpathy`

### 3. Deep Learning / Vision Experiment Work

Use this when the task is training, evaluation, metrics, datasets, or experiment orchestration:

- `dl-vision-experiment-orchestration`
- `dl-vision-benchmark-alignment`
- `dl-vision-experiment-reporting`
- `ace-training-command-composer` or `ace-memory-extraction-workflow`
- `grill-me`

### 4. Debugging / Failure Analysis

Use this when the task is a bug, training instability, mismatch, or failing evaluation:

- `superpowers-systematic-debugging`
- `dl-vision-training-failure-triage`
- `dl-vision-checkpoint-resume-integrity`
- `dl-vision-gpu-performance`
- `superpowers-verification-before-completion`
- `karpathy`

### 5. Large Multi-Step Execution

Use this when there is an approved plan and multiple mostly-independent tasks:

- `superpowers-writing-plans`
- `superpowers-dispatching-parallel-agents` or `superpowers-subagent-driven-development`
- `superpowers-verification-before-completion`
- `ace-result-aggregation`

## Fast Routing Heuristics

- unclear build/change request
  -> `superpowers-brainstorming`
- explicit plan request
  -> `superpowers-writing-plans`
- pressure-test / review-the-plan request
  -> `grill-me`
- debugging / instability / mismatch
  -> `superpowers-systematic-debugging`
- repo understanding / code search / codebase orientation
  -> `dl-vision-repo-workflow`
- paper-to-implementation task
  -> `paper-lookup` + `literature-review` + `dl-vision-paper-to-code`
- training / extraction / experiment command task
  -> `ace-training-command-composer` or `ace-memory-extraction-workflow`
- results / metrics / benchmark summary task
  -> `ace-result-aggregation` + `dl-vision-experiment-reporting`

Then:

- add `grill-me` before committing to important plans
- add `karpathy` during implementation tightening
- add verification skills before completion

## Repo-Specific Advice

In this repository, prefer these combinations:

- memory extraction changes:
  `ace-memory-extraction-workflow` -> `grill-me` -> `karpathy`
- training command changes:
  `ace-training-command-composer` -> `dl-vision-experiment-orchestration` -> `grill-me`
- result synthesis:
  `ace-result-aggregation` -> `dl-vision-experiment-reporting` -> `dl-vision-benchmark-alignment`
- runbook / next-step decisions:
  `ace-scene-runbook` -> `grill-me` -> `karpathy`

## Minimal Rule

If the task is important enough that a bad answer would waste hours, money, GPU time, or research effort, do not operate with a single skill. Use an explicit chain.
