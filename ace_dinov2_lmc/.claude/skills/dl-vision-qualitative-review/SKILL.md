---
name: dl-vision-qualitative-review
description: Use when inspecting qualitative vision outputs such as retrieved examples, visualizations, failure cases, overlays, attention maps, or scene-level diagnostics. Tie visual impressions back to exact runs and metric context.
---

# DL / Vision Qualitative Review

Use this skill for non-numeric result inspection that still needs rigor.

## Workflow

1. Identify the artifact type.
   Overlay, qualitative example, retrieval panel, trajectory plot, heatmap, scene visualization, or failure montage.
2. Link the artifact to provenance.
   Exact run, scene, checkpoint, and generation step.
3. Define the review question.
   Better localization? More stable predictions? Fewer catastrophic failures? Better calibration?
4. Compare with context.
   Numeric metrics, baseline outputs, and known failure modes.
5. Separate signal from anecdote.
   Highlight recurring patterns, not one-off attractive examples.

## Output

- artifact provenance
- qualitative findings
- consistency with metrics
- open ambiguities

## Rules

- Do not let one cherry-picked example override broad metric evidence.
- If visuals and metrics disagree, call out the disagreement directly.
- Prefer grouped failure patterns over isolated screenshots.
