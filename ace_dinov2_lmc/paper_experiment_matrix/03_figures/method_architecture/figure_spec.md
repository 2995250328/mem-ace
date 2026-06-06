# Method Figure Spec: GLACE + GeoLMC Memory

This document defines the intended content of the paper method figure before using image generation. It combines an open-source figure-generation workflow with the project-specific GLACE + GeoLMC architecture.

## Figure Goal

Show how compact scene memory, represented as scene tokens / map codes, conditions an ACE/GLACE coordinate regression pipeline for visual localization.

The figure should emphasize:

- Offline construction of scene memory from posed training images and geometry.
- Query-time extraction of local image features and optional GLACE global image descriptor.
- GeoLMC compression of scene memory into compact tokens.
- Memory-to-image feature fusion before coordinate regression.
- Two-stage training: local memory-conditioned model first, GLACE global refinement second.
- Final 6-DoF pose recovered from scene coordinate predictions with DSAC* / RANSAC.

## Open-Source Organization Layer

The diagram should follow this open-source-inspired workflow:

1. Use PaperBanana-style staged generation: method spec -> structured diagram draft -> generated reference -> visual critique -> vector cleanup.
2. Use GPT-Image2-Skill-style prompt structure: intended use -> canvas -> layout -> labels -> constraints.
3. Use visual-skills-style text control: readable typography, explicit labels, no unnecessary decoration.
4. Use Mermaid-style structural validation before image generation.

## Project Method Layer

Main paper-facing architecture:

- Backbone / local coordinate path: ACE-FCN local encoder and coordinate head.
- Memory path: sparse scene memory built from training images and depth / sparse geometry.
- Memory module: GeoLMC compressor + LMCFeatureFusion.
- Stage 1: train local ACE + GeoLMC memory path without GLACE global concatenation.
- Stage 2: load the Stage-1 local checkpoint, freeze the local stack, concatenate GLACE image-level global feature, and train the global head.

The figure may mention the broader DINOv2 ACE-G path only as a side note or appendix figure. The main figure should not mix DINOv2 historical ablations with the GLACE + LMC method figure.

## Required Blocks

Use these block labels exactly where possible:

- Training RGB-D / Sparse-Depth Images
- Memory Extraction
- Scene Memory / Map Codes
- Query Image
- ACE / GLACE Encoder
- Dense Local Features
- GLACE Global Descriptor
- GeoLMC Compressor
- Memory-to-Image Fusion
- Local ACE Coordinate Head
- Stage-2 GLACE Concatenation Head
- Refined Scene Coordinates
- DSAC* / RANSAC Pose Solver
- 6-DoF Camera Pose

## Required Arrows

Offline path:

1. Training RGB-D / Sparse-Depth Images -> Memory Extraction
2. Memory Extraction -> Scene Memory / Map Codes
3. Scene Memory / Map Codes -> GeoLMC Compressor

Query path:

1. Query Image -> ACE / GLACE Encoder
2. ACE / GLACE Encoder -> Dense Local Features
3. ACE / GLACE Encoder -> GLACE Global Descriptor
4. Dense Local Features -> Memory-to-Image Fusion
5. GeoLMC Compressor -> Memory-to-Image Fusion
6. Memory-to-Image Fusion -> Local ACE Coordinate Head
7. Local ACE Coordinate Head -> Stage-2 GLACE Concatenation Head
8. GLACE Global Descriptor -> Stage-2 GLACE Concatenation Head
9. Stage-2 GLACE Concatenation Head -> Refined Scene Coordinates
10. Refined Scene Coordinates -> DSAC* / RANSAC Pose Solver
11. DSAC* / RANSAC Pose Solver -> 6-DoF Camera Pose

## Two-Stage Training Inset

Add a compact inset titled "Two-stage training".

Text:

- Stage 1: local ACE + GeoLMC memory path
- Stage 2: freeze local stack + train GLACE concat/global head
- Select best checkpoint by pose accuracy

## Visual Grammar

Use three color-coded flows:

- Blue: image/query feature path.
- Green: scene memory / map-code path.
- Orange: coordinate / pose output path.

Use restrained academic styling:

- White background.
- Thin arrows.
- Rectangular modules.
- Minimal icons.
- No decorative gradients.
- No neural-network spaghetti.
- No photorealistic renderings.
- No dataset screenshots unless added manually later.

## What To Avoid

- Do not imply that GLACE global features are part of Stage 1.
- Do not show millions of ACE-G map codes in the main figure; our paper-facing branch uses compact K scene tokens.
- Do not mix Indoor6 / Wayspots experimental tables into the architecture figure.
- Do not show token ablations in this figure; token count belongs in an ablation plot.
- Do not include optimizer, batch size, GPU, or run directory details.

## Optional Caption Draft

Overview of the proposed GLACE + GeoLMC memory architecture. A compact scene memory is constructed offline from posed training images and geometry, then compressed into scene tokens by GeoLMC. At query time, dense local features from the ACE/GLACE encoder are fused with the compressed memory to produce memory-conditioned coordinate features. Stage 1 trains the local memory-conditioned coordinate head, while Stage 2 freezes the local stack and trains a GLACE-conditioned global head. The refined scene coordinates are converted to a 6-DoF camera pose by DSAC* / RANSAC.
