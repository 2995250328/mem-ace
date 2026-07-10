# Mem-ACE Method Architecture Figure Handoff

This folder contains the paper-facing handoff materials for drawing the Mem-ACE method figure in draw.io.

Use these files as the source of truth:

1. `figure_spec.md`: final semantic specification for the two-panel method figure.
2. `method_architecture.mmd`: compact Mermaid draft for checking nodes and arrows before drawing.
3. `paper_code_attachment/DRAWIO_HANDOFF_CN.md`: Chinese draw.io handoff instructions for the next Codex.
4. `paper_code_attachment/configs/common_method.json`: machine-readable shared method facts.
5. `paper_code_attachment/provenance/SOURCE_MAP.md`: production-code source map for fact checking.

Do not use older downstream-system-specific wording as the figure title or layout. The main figure should show Mem-ACE as a scene-memory plugin for scene coordinate regression.

Current value-geometry figure policy: the long-run ablation selects `value_only_raw`, so the default fusion path should be drawn as `Z + PE(P-c)` entering Memory Fusion. `P` is a latent 3D anchor, not a final coordinate prediction or supervised coordinate output.

The generated handoff archive includes only the cleaned source-of-truth files and the lightweight code attachment used for fact checking. Historical drafts in this directory are not part of the draw.io handoff.
