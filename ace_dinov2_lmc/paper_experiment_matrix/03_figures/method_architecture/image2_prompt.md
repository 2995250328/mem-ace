# Image2 Prompt: GLACE + GeoLMC Memory Method Figure

Copy the prompt below into image2. It is organized using open-source image-prompt practices, then specialized for our GLACE + GeoLMC Memory method.

```text
Use case: infographic-diagram
Asset type: academic paper method figure, CVPR / ICCV style architecture overview

Open-source workflow intent:
Follow the style of publication-quality academic method diagrams: structured like a PaperBanana-style research visual, prompt-organized like GPT-Image2-Skill, with Mermaid-like clear nodes and arrows. The output should be a clean reference figure that can be recreated in a vector editor.

Primary request:
Create a clean publication-ready architecture diagram titled "GLACE + GeoLMC Memory for Visual Localization". The figure should explain how compact scene memory tokens condition an ACE / GLACE coordinate regression pipeline.

Canvas:
Landscape 16:9 layout, white background, clean vector-like academic diagram, strong hierarchy, readable labels, thin arrows, consistent spacing, no watermark.

Overall layout:
Use a left-to-right pipeline with three visually separated horizontal bands.

Band 1, top: "Offline Scene Memory Construction"
Show a small stack of posed training images with depth / sparse geometry.
Arrow to a block labeled exactly "Memory Extraction".
Arrow to a green token bank labeled exactly "Scene Memory / Map Codes".
Annotate the token bank with "K compact scene tokens" and "geometry + appearance".

Band 2, middle: "Query Image Feature Extraction"
Show a single query image entering a blue block labeled exactly "ACE / GLACE Encoder".
Split the encoder output into two branches:
1. "Dense Local Features"
2. "GLACE Global Descriptor"

Band 3, bottom: "GeoLMC Conditioning and Coordinate Regression"
From "Scene Memory / Map Codes", draw a green arrow into "GeoLMC Compressor".
Inside or below the compressor block, include the short phrases "geometry-aware token compression" and "scene-scale normalization".
Then draw an arrow into "Memory-to-Image Fusion".
The "Memory-to-Image Fusion" block must also receive an arrow from "Dense Local Features".
Then draw arrows to:
"Local ACE Coordinate Head" -> "Stage-2 GLACE Concatenation Head" -> "Refined Scene Coordinates" -> "DSAC* / RANSAC Pose Solver" -> "6-DoF Camera Pose".
The "Stage-2 GLACE Concatenation Head" must also receive an arrow from "GLACE Global Descriptor".

Training inset:
Add a compact inset on the right titled exactly "Two-stage training".
Inside it, show three short lines:
"Stage 1: local ACE + GeoLMC memory path"
"Stage 2: freeze local stack + train GLACE concat/global head"
"Best checkpoint selected by pose accuracy"

Color and style:
Use restrained academic colors:
blue for query image / feature path,
green for scene memory / map-code path,
orange for coordinates / pose output,
light gray for the two-stage training inset.
Use flat rectangular modules with subtle rounded corners, simple token-bank glyphs, and thin directional arrows.
Typography should be clean sans-serif, black or dark gray, with major labels clearly readable.

Constraints:
All text must be spelled exactly as specified.
Avoid tiny text.
Avoid decorative gradients, shadows, photorealistic renderings, cartoon icons, unrelated neural network layers, dataset result tables, GPU details, optimizer details, batch-size details, or excessive equations.
Do not imply that GLACE global features are used in Stage 1.
Do not show token ablation plots.
Do not add logos, watermarks, or unrelated text.
```

## Shorter Fallback Prompt

Use this if image2 struggles with dense text:

```text
Create a clean CVPR-style method diagram titled "GLACE + GeoLMC Memory for Visual Localization". Landscape 16:9, white background, vector-like academic style.

Show this left-to-right pipeline:
Training RGB-D / Sparse-Depth Images -> Memory Extraction -> Scene Memory / Map Codes -> GeoLMC Compressor -> Memory-to-Image Fusion.
In parallel, show Query Image -> ACE / GLACE Encoder -> Dense Local Features -> Memory-to-Image Fusion.
Also show ACE / GLACE Encoder -> GLACE Global Descriptor -> Stage-2 GLACE Concatenation Head.
Then show Memory-to-Image Fusion -> Local ACE Coordinate Head -> Stage-2 GLACE Concatenation Head -> Refined Scene Coordinates -> DSAC* / RANSAC Pose Solver -> 6-DoF Camera Pose.

Add a small inset:
Two-stage training:
Stage 1: local ACE + GeoLMC memory path
Stage 2: freeze local stack + train GLACE concat/global head

Use blue for query features, green for memory tokens, orange for output pose. Keep labels readable, no tiny text, no decorative elements, no watermark.
```
