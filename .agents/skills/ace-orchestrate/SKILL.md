---
name: ace-orchestrate
description: Use this skill when the user explicitly asks to use subagents, multi-model delegation, orchestration, routing, or parallel agent work for ACE/GeoLMC/Codex repository tasks. It defines cost-aware model routing, context packets, subagent boundaries, and a hard rule that xhigh reasoning requires explicit user approval.
metadata:
  short-description: Cost-aware multi-model subagent routing for ACE work
---

# ACE Orchestrate

Use this skill only when the user explicitly asks for subagents, parallel agents, delegation, orchestration, multi-model routing, or this skill by name. Do not activate it merely because a task is large or difficult.

## Hard Limits

- The root agent remains accountable for the final answer, integration, and verification.
- Default maximum capability is `gpt-5.5` with `reasoning_effort=high`.
- Never use `reasoning_effort=xhigh` unless the user explicitly approves it for this task.
- Do not delegate urgent blocking work when the root's next action depends on that result.
- Do not spawn agents for tiny edits, one-command checks, simple Q&A, or when the user asks not to use subagents.
- Workers must have disjoint write scopes. Explorers should usually be read-only.

## Routing Table

| Role | Model | Effort | Use For | Avoid For |
| --- | --- | --- | --- | --- |
| Root orchestrator | current/root model, usually `gpt-5.5` | high max | task decomposition, final decisions, core risky code, integration, verification | raw broad repo scans when scouts can do them |
| Doc scout | `gpt-5.4-mini` | low/medium | reading plans, extracting requirements, finding flags, summarizing logs/docs | architecture decisions, core model code |
| Code locator | `gpt-5.4-mini` or `gpt-5.3-codex` | medium | finding call chains, file:line anchors, config/checkpoint plumbing | writing high-risk patches |
| Bounded worker | `gpt-5.4` or `gpt-5.3-codex` | medium/high | scoped implementation in assigned files | shared core files unless explicitly assigned |
| High-risk reviewer | `gpt-5.5` | high | model architecture, checkpoint compatibility, training-path correctness, security/migration/perf review | routine docs or log summarization |

If `gpt-5.3-codex-spark` is available in a future session, use it only for ultra-fast scouting, mechanical edits, or simple targeted checks. If unavailable, use `gpt-5.4-mini`.

## Dispatch Pattern

Before spawning agents:

1. Identify the immediate root critical-path task.
2. Split only independent sidecar tasks that can run in parallel.
3. Assign each subagent a bounded role, scope, allowed actions, done condition, and return format.
4. Keep root-only routing rationale out of subagent prompts.
5. Continue useful non-overlapping work while subagents run.

## Minimal Context Packet

Every delegated task should include:

```text
Packet id: <short id>
Role: <explorer|worker|reviewer>
Objective: <one concrete outcome>
Scope: <files/modules/commands allowed>
Non-goals: <what not to touch>
Context handles: <paths, plan files, error snippets, commands>
Allowed actions: <read-only or exact write set>
Constraints: <repo rules, do not revert user edits, no unrelated refactor>
Done condition: <what must be returned>
Return format: <bullets with file:line evidence, changed files, tests run, blockers>
```

## ACE/GeoLMC Defaults

For this repository, prefer this split:

- Doc scout: read `ace_g_global_refactor/steps/*.md`, extract experiment contracts and CLI/config fields.
- Code locator: find `options_dinov2_lmc.py`, `trainer_dinov2_lmc.py`, `test_ace_dinov2_lmc*.py`, and `/home/xwh/project/ace_depth/ace_compressor.py` call sites.
- Root or high-risk worker: modify `ace_compressor.py` attention/model internals.
- Bounded worker: modify eval/checkpoint restore files only when write scope is isolated.
- Root: run `py_compile`, minimal `GeoLMC` smoke tests in `mapanything`, and prepare final training commands.

## Lifecycle Rules

- A subagent result is evidence, not completion.
- If a subagent is stuck, timed out, or returns vague output, repair the packet or redelegate a narrower task.
- If a worker changed files, root must inspect the diff before finalizing.
- Final answer must state what was delegated, what was changed, and what validation passed or could not run.
