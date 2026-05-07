---
name: agent-skill-orchestration
description: Use when a request is ambiguous, costly, multi-step, research-heavy, experiment-heavy, debugging-heavy, or likely to benefit from structured requirement clarification, retrieval, planning, critique, and verification. This is the default routing skill for choosing and sequencing multiple skills automatically instead of answering in one shot.
---

# Agent Skill Orchestration

Use this skill when the task should be handled as a deliberate multi-skill workflow rather than a single-shot answer.

## Routing Rule

If there is a reasonable chance the task would benefit from more than one specialized skill, use this skill first to choose the chain.

That includes:

- vague or underspecified requests
- requests with multiple possible designs
- GPU-expensive or time-expensive experiment decisions
- debugging where root cause is not obvious
- result aggregation or benchmark interpretation
- literature-to-code or repo-to-plan work
- any request where a shallow answer would create rework

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

Expected effect:

- requirements become explicit
- tradeoffs are surfaced
- weak assumptions get challenged before code is touched

### 2. Research / Retrieval / Technical Judgment

Use this when the task needs literature, codebase lookup, benchmark context, or synthesis:

- `dl-vision-repo-workflow` for repo orientation
- `paper-lookup` and `literature-review` for external technical context
- `dl-vision-paper-to-code` when turning papers into implementable steps
- `grill-me` to test whether the proposed interpretation really holds
- `karpathy` to keep the reasoning grounded and non-handwavy

Expected effect:

- better evidence
- clearer links between sources and code decisions
- less “sounds right” reasoning

### 3. Deep Learning / Vision Experiment Work

Use this when the task is training, evaluation, metrics, datasets, or experiment orchestration:

- `dl-vision-experiment-orchestration`
- `dl-vision-benchmark-alignment`
- `dl-vision-experiment-reporting`
- `ace-training-command-composer` or `ace-memory-extraction-workflow` when working in this repo
- `grill-me` before committing to a costly run

Expected effect:

- experiments are comparable
- command construction is less error-prone
- evaluation language stays aligned with actual evidence

### 4. Debugging / Failure Analysis

Use this when the task is a bug, training instability, mismatch, or failing evaluation:

- `superpowers-systematic-debugging`
- `dl-vision-training-failure-triage`
- `dl-vision-checkpoint-resume-integrity` if checkpoints/resume may be involved
- `dl-vision-gpu-performance` if performance or device behavior is involved
- `superpowers-verification-before-completion`
- `karpathy` during the actual fix to keep changes minimal

Expected effect:

- fewer premature fixes
- clearer root-cause isolation
- stronger final verification

### 5. Large Multi-Step Execution

Use this when there is an approved plan and multiple mostly-independent tasks:

- `superpowers-writing-plans`
- `superpowers-dispatching-parallel-agents` or `superpowers-subagent-driven-development`
- `superpowers-verification-before-completion`
- `ace-result-aggregation` if the output is experiment-heavy

Expected effect:

- work stays decomposed
- review is built into execution
- context does not collapse into an unstructured blob

## Fast Routing Heuristics

Map the user request to one of these entry points before doing anything substantial:

- "I want to build / add / change ..." with unclear scope
  -> `superpowers-brainstorming`
- "Give me a plan / steps / execution plan"
  -> `superpowers-writing-plans`
- "Review this plan / challenge this / what am I missing"
  -> `grill-me`
- "Find why this failed / debug / mismatch / crash / wrong result"
  -> `superpowers-systematic-debugging`
- "Search code / understand repo / explain where this lives"
  -> `dl-vision-repo-workflow`
- "Read papers / compare papers / turn method into implementation"
  -> `paper-lookup` + `literature-review` + `dl-vision-paper-to-code`
- "Prepare training command / extraction command / compare runs"
  -> `ace-training-command-composer` or `ace-memory-extraction-workflow`
- "Summarize results / update experiment table / compare metrics"
  -> `ace-result-aggregation` + `dl-vision-experiment-reporting`

After picking the entry point:

- use `grill-me` before committing to an important plan
- use `karpathy` during implementation or final recommendation tightening
- use verification skills before declaring completion

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

## Anti-Patterns

Do not:

- use `grill-me` after code is already committed as a substitute for planning
- use `karpathy` as a replacement for requirement clarification
- use benchmark/reporting skills before verifying the experiment source files
- use debugging skills without first identifying the failure surface

## Minimal Rule

If the task is important enough that a bad answer would waste hours, money, GPU time, or research effort, do not operate with a single skill. Use an explicit chain.
