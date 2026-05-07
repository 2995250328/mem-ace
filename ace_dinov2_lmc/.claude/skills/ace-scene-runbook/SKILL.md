---
name: ace-scene-runbook
description: Use when planning or documenting scene-level workflows for ACE-DINOv2-LMC, especially Indoor6 scenes such as scene2a/scene3. Focus on extraction, training, eval, clustering decisions, and next-step gating per scene.
---

# ACE Scene Runbook

Use this skill when the unit of work is a scene or a small scene set.

## Workflow

1. Define the scene objective.
   Baseline validation, C1 validation, diagnostic run, cluster-local run, or ensemble support.
2. Map the stage sequence.
   Extraction -> training -> eval -> aggregation -> next-step decision.
3. Record scene-specific caveats.
   Whether the scene is considered positive evidence, diagnostic only, cluster-preferred, or pending contract validation.
4. Record exact commands and outputs.
   Extraction command, training command, result paths, and expected summary files.
5. Define decision gates.
   What result is needed before advancing to the next phase or next scene.

## Output

- scene-specific plan
- command list
- result locations
- next-step gate

## Rules

- Do not mix scene-level positive evidence and diagnostic evidence.
- Preserve exact scene naming and timestamped output directories.
- If a scene has a known special interpretation, carry it into the runbook explicitly.
