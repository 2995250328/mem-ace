---
name: dl-vision-figure-table-reporting
description: Use when turning deep learning and vision experiment results into clean tables, plots, and concise result narratives. Emphasize metric consistency, exact provenance, and comparison clarity.
---

# DL / Vision Figure And Table Reporting

Use this skill when experiments need to be presented as tables, plots, or compact report text.

## Workflow

1. Normalize the source metrics.
   Confirm exact metric definitions, units, phases, and best-direction before plotting or tabulating.
2. Define the comparison axis.
   Baseline vs variant, per-scene, per-dataset, per-iteration, or per-ablation.
3. Choose the right artifact.
   Table for exact comparisons, line plot for iteration dynamics, bar plot for grouped comparisons, qualitative panel for vision outputs.
4. Preserve provenance.
   Every plotted or tabulated number should trace back to an exact file and run.
5. Write the shortest honest narrative.
   State what improved, what regressed, and what remains ambiguous.

## Output

- normalized result table plan
- plot recommendations
- narrative summary template
- source provenance note

## Rules

- Never combine incompatible metrics in one figure.
- Use absolute timestamps or run names when runs are easy to confuse.
- If a run is diagnostic rather than production-grade, label it as such.
