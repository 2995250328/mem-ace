---
name: ace-paper-writing-workflow
description: Use when writing, revising, outlining, positioning, or reviewer-proofing an academic paper about ACE-DINOv2-LMC, GLACE-LMC, ACE-FCN-LMC, memory compression, scene conditioning, relocalization experiments, or this repo's paper claims.
---

# ACE Paper Writing Workflow

Use this skill to turn this repository's research, logs, and experiment matrices into a defensible CV/3D relocalization manuscript.

## Core Principle

Write claims from verified evidence, not from desired story. Every paper claim must map to exact experiments, tables, figures, logs, or research-stage documents.

## Workflow

1. **Frame the claim.**
   Decide whether the paper is claiming backbone superiority, memory-compression usefulness, scene-conditioning reliability, efficiency, or generality across SCR/APR-style pipelines.
2. **Inventory evidence.**
   Use `ace-result-aggregation` and `dl-vision-experiment-reporting` before quoting metrics. Require exact run paths, timestamps, metric definitions, and best-selection rules.
3. **Position related work.**
   Use `literature-review`, `paper-lookup`, and `citation-management`. Anchor comparisons to ACE, ACE-G, Map-relative pose regression, GLACE, DINOv2, memory/compression, SCR, APR, Indoor-6, Wayspots, Cambridge, and RIO/Naver-style benchmarks as relevant.
4. **Design paper tables and figures.**
   Use `dl-vision-figure-table-reporting` and `scientific-visualization`. Prefer paper-safe tables: main results, ablations, reliability/negative-transfer controls, efficiency, and qualitative/diagnostic figures.
5. **Write manuscript prose.**
   Use `scientific-writing`. Final manuscript sections should be flowing prose, not bullet lists. Keep limitations explicit where SquareBench/Cambridge/global-feature negative transfer is known.
6. **Self-review before finalizing.**
   Use `peer-review` to check novelty, methodology, metric fairness, ablation completeness, unsupported claims, and reviewer objections.
7. **Adapt to venue.**
   Use `venue-templates` for CVPR/ICCV/ECCV/NeurIPS-style constraints, page limits, citation style, and reviewer expectations.

## Paper-Safe Claim Rules

- Do not claim universal improvement unless every reported dataset supports it.
- Separate DINO+LMC, GLACE+LMC, and ACE-FCN-LMC evidence; do not merge them into one unsupported method claim.
- Treat SquareBench global-feature negative transfer as an important limitation or reliability motivation, not as a hidden failure.
- State whether gains come from local memory, global conditioning, gate/reliability control, memory extraction, or evaluation protocol.
- Always report exact thresholds and units: median rotation/translation, Acc5, Acc10/5, pct5, hypotheses, dataset split, and best-checkpoint rule.

## Recommended Manuscript Backbone

| Section | Required Evidence |
|---|---|
| Introduction | problem gap, scene memory motivation, main empirical wins, honest boundary case |
| Related Work | ACE/ACE-G, SCR/APR, DINO/GLACE features, scene memory/compression, relocalization benchmarks |
| Method | memory extraction, LMC compressor/fusion, training stages, inference path, checkpoint/eval protocol |
| Experiments | datasets, baselines, metrics, main tables, ablations, efficiency, reliability analysis |
| Discussion | why memory helps, when global conditioning hurts, limitations, future reliability control |

## Common Mistakes

- Writing the story before aggregating results.
- Mixing per-iteration eval, post-train eval, and independent test metrics without saying so.
- Calling a manual per-scene gate an automatic solution.
- Overstating GLACE global conditioning when local-only or Stage1 baselines explain the gain.
- Citing papers from memory without DOI/BibTeX verification.
