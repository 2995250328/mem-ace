# Image2 Prompt: Project Workflow Figure

Copy this into image2 if you want a polished visual reference for the research workflow figure. The editable source of truth is `project_workflow.mmd`.

```text
Use case: infographic-diagram
Asset type: academic project workflow figure for a computer vision paper

Primary request:
Create a clean CVPR / ICCV style workflow diagram titled "GLACE + GeoLMC Memory Research Workflow". The figure should show how open-source diagram workflows, project-specific method design, experiments, stability triage, result aggregation, and paper artifacts connect.

Canvas:
Landscape 16:9, white background, clean vector-like flowchart, readable text, thin arrows, restrained academic colors, no watermark.

Layout:
Use five grouped regions arranged left to right:

1. "Open-source figure workflow" in light gray.
Include blocks:
"Open-source diagram workflow"
"PaperBanana-style staged figure process"
"GPT-Image2 / visual-skills prompt structure"
"Mermaid / Excalidraw editable draft"

2. "Project-specific method design" in blue.
Include blocks:
"Project-specific method spec"
"GLACE + GeoLMC Memory"
"Scene memory / map codes"
"Stage 1 local memory path"
"Stage 2 GLACE concat/global head"
"DSAC* pose solving"

3. "Figure preparation" in blue-gray.
Include blocks:
"Editable structural draft"
"Image2 reference prompt"
"Generated reference figure"
"Vector cleanup / manual verification"

4. "Experiment matrix" in green.
Include blocks:
"Indoor6 validation"
"Wayspots validation"
"GLACE baseline comparison"
"DINOv2 ACE-G context"
"Token-count ablation"

5. "Paper reporting" in orange.
Include blocks:
"Result aggregation"
"Paper figures and tables"
"Final paper narrative"

Add a small red or muted pink side branch titled "Stability triage" connected into "Result aggregation".
It should include:
"Memory sanity checks"
"Divergence / NaN checks"
"Resume + big-buffer training"
"Low-LR retry / OOM tracking"

Arrow logic:
Open-source figure workflow -> Project-specific method spec -> Editable structural draft -> Image2 reference prompt -> Generated reference figure -> Vector cleanup / manual verification -> Paper figures and tables.
Project-specific method design -> Experiment matrix -> Result aggregation -> Paper figures and tables -> Final paper narrative.
Stability triage -> Result aggregation.

Style constraints:
This is a workflow figure, not a neural network architecture diagram.
Use simple rectangular modules, thin arrows, consistent spacing, readable sans-serif labels, and grouped sections with subtle headers.
Do not include shell commands, GPU IDs, run directories, raw metric tables, decorative neural-network layers, logos, watermarks, or unrelated modules.
All text should be spelled exactly as specified.
```
