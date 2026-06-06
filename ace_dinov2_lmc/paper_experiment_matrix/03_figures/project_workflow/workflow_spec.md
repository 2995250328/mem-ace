# Workflow Figure Spec: GLACE + GeoLMC Memory Research Pipeline

## Figure Role

This figure explains the overall research workflow for the current work, not just the model architecture. It should be useful for planning the paper and communicating how open-source figure workflows, project-specific method design, experiments, and reporting fit together.

## Open-Source Layer

The workflow should begin with open-source organization principles:

1. PaperBanana-style staged academic figure workflow.
2. GPT-Image2-Skill-style structured image prompt writing.
3. visual-skills-style diagram readability constraints.
4. Mermaid/Excalidraw-style editable structural drafting.

These are not the scientific contribution. They are the preparation process used to organize the figure and paper assets.

## Project-Specific Layer

The project-specific research work contains these branches:

- Method definition: GLACE + GeoLMC Memory for visual localization.
- Memory construction: scene memory / map codes from training images and geometry.
- Model path: ACE/GLACE encoder, GeoLMC compressor, memory-to-image fusion, local coordinate head, Stage-2 GLACE concat/global head, DSAC* pose solving.
- Experiments: Indoor6, Wayspots, GLACE baseline, GLACE+LMC, DINOv2 historical ACE-G context, token-count ablation.
- Stability and debugging: memory sanity, divergence checks, resume/big-buffer training, low-LR retry, OOM tracking.
- Reporting: metric aggregation, best-per-metric tables, curated paper tables, method figure, workflow figure, paper experiment matrix.

## Required Flow

Use this top-level order:

1. Open-source diagram workflow
2. Project method specification
3. Editable structural drafts
4. Image2 prompt / generated reference
5. Method implementation and training runs
6. Dataset-level experiment matrix
7. Failure/stability triage
8. Result aggregation
9. Paper-facing artifacts
10. Final paper narrative

## Required Labels

Use these labels exactly where possible:

- Open-source diagram workflow
- Project-specific method spec
- Editable structural draft
- Image2 reference prompt
- GLACE + GeoLMC Memory
- Scene memory / map codes
- Stage 1 local memory path
- Stage 2 GLACE concat/global head
- Indoor6 validation
- Wayspots validation
- DINOv2 ACE-G context
- Token-count ablation
- Stability triage
- Result aggregation
- Paper figures and tables
- Final paper narrative

## Visual Grammar

Use four colors:

- Gray: open-source preparation workflow.
- Blue: project method design.
- Green: experiments and validation.
- Orange: reporting and paper assets.
- Red or muted pink: failure/stability triage.

Keep the figure as a clean flowchart, not a neural-network architecture diagram.

## What To Avoid

- Do not include exact run directories.
- Do not include GPU IDs or shell commands.
- Do not include full metric tables.
- Do not duplicate the detailed model architecture figure.
- Do not imply that open-source diagram tools are part of the algorithm.

## Caption Draft

Workflow used to organize the GLACE + GeoLMC Memory study. Open-source academic-figure and diagramming workflows are first used to define an editable paper figure process. Project-specific method details are then mapped into architecture drafts and image-generation prompts. In parallel, Indoor6, Wayspots, DINOv2 ACE-G context experiments, token-count ablations, and stability triage feed into result aggregation and paper-facing figures, tables, and narrative.
