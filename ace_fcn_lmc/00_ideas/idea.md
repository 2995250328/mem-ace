# Idea: ACE FCN + GeoLMC (ace_fcn_lmc)

## Core Idea

Combine the original ACE FCN encoder with the GeoLMC (Geometric Latent Memory Compression)
two-stage training framework. The goal is to surpass the ACE-G baseline while preserving
ACE's original inference speed (~30 FPS) by leveraging an external 3D point-cloud memory
(pooled memory) to improve scene coordinate regression accuracy.

## Motivation

- ACE vanilla: fast training and inference, but accuracy is limited by single-stage buffer training
- DINOv2 LMC: high accuracy but slow inference (~10-15 FPS), requires RGB input
- Target: replace DINOv2 with the FCN encoder (grayscale, 8x downsampling, 512-dim),
  inherit the LMC two-stage iterative training framework, and surpass ACE-G in low-resource settings

## Key Design Decisions

1. **Encoder swap**: ACEEncoder wraps the original FCN encoder, exposing the same interface as DINOv2
2. **Trainer inheritance**: TrainerACEFCN/TrainerACEFCNLMC only override `_create_regressor()`;
   all training logic is inherited from TrainerACEDINOv2/TrainerACEDINOv2LMC
3. **Two training modes**:
   - Vanilla: single-stage buffer training (`--use_lmc False`)
   - LMC: two-stage iterative training (`--use_lmc True --memory_path <pooled.pt>`)
4. **No resolution constraint**: FCN encoder supports arbitrary resolution (no patch-size restriction)

## Validation Status

- [x] 7-Scenes Chess: both vanilla and LMC modes validated
- [x] Indoor6 Scene3: full LMC training validated, surpasses ACE-G baseline
- [x] Multi-round iterative training (vanilla_iterations=3) validated
