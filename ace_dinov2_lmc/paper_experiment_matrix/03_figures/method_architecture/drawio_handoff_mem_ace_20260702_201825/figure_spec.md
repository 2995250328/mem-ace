# Method Figure Spec: Mem-ACE Scene-Compressed Memory

This document defines the content of the main Mem-ACE method figure before drawing it in draw.io. The figure should explain the stable method scope: compact scene memory, scene-anchored latent tokens, memory fusion, scene-coordinate regression, and geometric supervision.

## Figure Goal

Show how Mem-ACE turns a pooled scene memory into compact anchored scene tokens and injects those tokens into an existing scene coordinate regression (SCR) pipeline.

The figure should emphasize:

- Offline construction of pooled scene memory from posed training images, SfM/STGS geometry, and feature pooling.
- Compression of a large memory bank into a small set of anchored tokens `(Z, P)`.
- Explicit use of latent positions `P` in fusion through scene-centered positional encoding `PE(P-c)`.
- Query-time memory fusion before a standard SCR coordinate head.
- Stable geometric supervision: single-view reprojection and cross-view track reprojection.
- A semantic two-phase training schedule: `Geometric Alignment` followed by `Cached Memory Adaptation`.

Long-run ablation note: use the value-geometry fusion path as the default. The figure should show `Z + PE(P-c)` entering Memory Fusion as the main path, not as an optional branch. Keep the claim modest: `P` supplies a centered geometry channel to fusion, but the figure should not imply that latent anchors alone explain all gains.

## Recommended Layout

Use a two-panel method figure.

### Panel (a): Mem-ACE Framework

Main left-to-right path:

```text
Posed Training Images + SfM/STGS Geometry
  -> Offline Memory Construction
  -> Pooled Scene Memory M = {(p_i, f_i)}, scene center c
  -> Geo-token Compressor
  -> Anchored Scene Tokens (Z, P)

Query Image
  -> SCR Backbone / Encoder
  -> Dense Query Features Q

Dense Query Features Q + Anchored Scene Tokens (Z, P) + PE(P-c)
  -> Memory Fusion
  -> SCR Coordinate Head
  -> Dense Scene Coordinates X_hat(u)
  -> downstream PnP/RANSAC Pose Solver (small, secondary geometry block)
```

Training band below the main path:

```text
Geometric Alignment:
  single-view reprojection + cross-view track reprojection

Cached Memory Adaptation:
  cache anchored scene tokens per scene, freeze compressor, adapt fusion/head
```

Draw training signals with dashed arrows into `Dense Scene Coordinates` or the coordinate head. The training band should not dominate the figure.

### Panel (b): Scene-Anchored Memory Module

This smaller panel explains the internal interface of the compressor and fusion block:

```text
Memory Points X and Memory Features F
  -> Geo-token Compressor
  -> latent features Z
  -> latent positions P

P and scene center c
  -> PE(P-c)

Query Features Q + Z + PE(P-c)
  -> Memory Fusion
  -> Fused Query Features
  -> Coordinate Head
```

The panel must make three facts visually explicit:

1. The compressor outputs both feature tokens `Z` and latent positions `P`.
2. Fusion consumes `P` through `PE(P-c)`; `P` is not a final coordinate prediction.
3. The scene center `c` aligns the memory-token coordinate convention with the SCR coordinate convention.

## Required Labels

Use these labels exactly or with only minor typographic shortening:

- Mem-ACE
- Offline Memory Construction
- SfM/STGS Geometry
- Pooled Scene Memory
- Geo-token Compressor
- Anchored Scene Tokens `(Z, P)`
- Scene Center `c`
- Centered Encoding `PE(P-c)`
- Dense Query Features `Q`
- Memory Fusion
- SCR Coordinate Head
- Dense Scene Coordinates
- Geometric Alignment
- Cached Memory Adaptation
- Single-view Reprojection
- Cross-view Track Reprojection

## Visual Grammar

Use three restrained flows:

- Blue: query image and dense feature path.
- Green: scene memory and anchored token path.
- Orange: scene coordinates, projection, and pose geometry.
- Gray: training schedule and offline metadata.

Use solid arrows for forward computation and dashed arrows for supervision or offline-only dependencies. Keep labels short and place explanatory detail in the caption.

## What To Avoid

Do not draw or label the main figure with:

- Code-stage labels or launch-script terminology.
- Downstream global-refinement internals or branch-specific head variants.
- DINO layer selection, key/value source asymmetry, scale tokens, or feature-layer ablations.
- Local/hierarchical/learned compressor branches.
- Cascade, reread, coord-prior, or other fusion variants.
- Dataset-specific lanes or benchmark-result panels.
- Optimizer, batch size, GPU, run folder, or checkpoint file details.

## Caption Draft

Mem-ACE builds a pooled scene memory from posed training images and SfM/STGS geometry, then compresses the memory into a fixed set of anchored scene tokens. Each token contains a feature vector `Z` and a latent 3D position `P`; fusion reads the tokens together with a scene-centered positional encoding `PE(P-c)`. The fused query features are decoded by a standard SCR head into dense scene coordinates. Training combines single-view reprojection with cross-view track reprojection, then adapts the regressor using cached scene tokens for stable memory-conditioned coordinate prediction.
