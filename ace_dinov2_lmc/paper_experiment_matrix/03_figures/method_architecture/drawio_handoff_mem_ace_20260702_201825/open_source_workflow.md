# Draw.io Figure Workflow

This file records the workflow for turning the Mem-ACE method specification into a draw.io paper figure.

## Source of Truth

Use `figure_spec.md` for the figure semantics. Use `method_architecture.mmd` only as a structural draft. Use `paper_code_attachment/provenance/SOURCE_MAP.md` only when a method fact needs to be checked against production code.

## Recommended Workflow

1. Read `figure_spec.md` and identify the two required panels.
2. Use `method_architecture.mmd` to check the node order and arrow directions.
3. Build an editable draw.io file with swimlanes or grouped panels.
4. Export a PNG preview without embedded diagram XML for visual inspection.
5. Export the final SVG/PDF/PNG with embedded draw.io XML after approval.

## Figure Policy

The final figure should be a compact method diagram, not a code-branch diagram. It should not expand all historical ablations, downstream-system-specific integration details, feature-layer choices, or launch-script settings.

The only method-specific details that must be visible are:

- pooled scene memory `{p_i, f_i}`;
- Geo-token compression into `(Z, P)`;
- scene-centered encoding `PE(P-c)`;
- value-geometry fusion as the default long-run-selected path;
- memory fusion before an SCR coordinate head;
- single-view and cross-view geometric supervision;
- semantic training phases: `Geometric Alignment` and `Cached Memory Adaptation`.
