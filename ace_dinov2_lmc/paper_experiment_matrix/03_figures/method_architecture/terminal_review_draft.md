# Terminal Review Draft: Latent Scene Memory Method Figure

Date: 2026-06-28
Scope: ACE + DINOv2 + GeoLMC / ACE-FCN + GLACE method figure planning.
Purpose: terminal-review specification before generating TikZ / draw.io / Mermaid.

## 0. Source-of-truth decision needed

The current figure folder says the main figure should focus on GLACE + GeoLMC and keep the DINOv2 ACE-G path as a side note or appendix. The current paper draft is broader:

- Abstract/introduction claim a plug-in latent scene-memory interface.
- Method section explicitly has two instantiations:
  - one-level two-stage: DINOv2 + MapAnything memory on Indoor6;
  - two-level two-stage: ACE-FCN memory first, then GLACE-style global-local regressor.
- Method section also includes GVCS visibility-based global-conditioning selection.

Recommendation:

- Main Figure 1 should be a unified latent scene-memory framework with two instantiation lanes.
- Detailed GLACE + GeoLMC implementation can be either the lower panel of Figure 1 or Figure 2.
- The old GLACE-only figure spec is still useful, but it is now too narrow for the current paper claim.

If the paper should instead emphasize only the final GLACE+ACE-FCN system, then abstract/method wording should be narrowed. Otherwise, update the figure spec as below.

## 1. Figure title

Preferred title:

Latent Scene Memory Conditioning for Scene Coordinate Relocalization

Short labels:

- LMC = Latent Memory Compression / Conditioning
- GeoLMC = geometric latent memory compressor
- SCR = scene coordinate regression

## 2. Main figure layout

Use a 3-band architecture with a right-side decision inset.

Band A: Offline scene memory construction

Training posed images / RGB-D or sparse depth
  -> memory view selection / feature extraction
  -> pooled scene memory bank
  -> GeoLMC compressor
  -> K latent scene tokens

Band B: Query-time coordinate regression

Query image
  -> backbone / encoder
  -> dense query features
  -> memory-to-image fusion
  -> coordinate head / global conditioning head
  -> dense scene coordinates
  -> DSAC* / RANSAC
  -> 6-DoF pose

Band C: Training and routing inset

- Stage 1 local memory training.
- Stage 2 global refinement when applicable.
- GVCS visibility route: keep global or fall back to local/hierarchical.

Recommended color grammar:

- Query path: blue.
- Memory path: green.
- Geometry / pose solver: orange.
- Training-only losses / frozen modules / routing: gray.
- Solid arrows: forward computation.
- Dashed arrows: supervision, routing, or offline-only dependency.

## 3. Exact node list

### 3.1 Offline memory construction nodes

Node M0: Posed training images

Label:
Posed training images
RGB-D / sparse depth / SfM depth

Details to keep implicit:
ACE loader or WAI loader, camera intrinsics, camera-to-world poses.

Node M1: View selection and extraction

Label:
Memory view selection + feature extraction

Possible small text:
FPS / covisibility / ASB; MapAnything, DINOv2, GLACE encoder, or ACE-FCN

Do not over-expand this in the main figure.

Node M2: Pooled scene memory bank

Label:
Pooled scene memory bank
{points p_i, features f_i}

Shape note:
p_i in R^3, f_i in R^D, N pooled points

Important correction to old spec:
Do not label this as "K compact tokens". K tokens are produced after GeoLMC, not before it.

Node M3: Optional metadata side chips

Label chips:
scene center
poses / intrinsics
scale tokens
rays / Plucker rays

These are optional side inputs. Keep them visually small.

Node M4: GeoLMC compressor

Label:
GeoLMC compressor
geometry-aware cross-attention

Internal callouts:
- FPS samples latent coordinates P in R^{K x 3}.
- 3D PE of P - scene center seeds latent queries.
- Key = selected layer slice or scalar mix of memory features.
- Value = full concatenated memory features.
- Distance bias / RBF / CRPB / PointRoPE can modulate attention.
- Optional scale tokens add a global scene shift.

Node M5: Latent scene tokens

Label:
K latent scene tokens
Z in R^{K x d}, P in R^{K x 3}

Important:
At evaluation, these are compressed once per scene and cached for all query frames.

### 3.2 Query-time nodes

Node Q0: Query image

Label:
Query image I

Node Q1: Backbone / encoder

Use branch labels:

Branch 1 label:
DINOv2 / ACE-G backbone

Branch 2 label:
ACE-FCN / GLACE encoder

Node Q2: Dense query features

Label:
Dense query features B(I)

For ACE-FCN:
stride 8, 512-dim local feature grid.

For DINOv2 / MapAnything memory path:
feature dimension follows backbone / memory contract.

Node Q3: Optional GLACE global descriptor

Label:
GLACE global descriptor g(I)

This is a side branch from dataset-aligned `features.npy` / GLACE encoder output, not part of GeoLMC memory.

Node Q4: Memory-to-image fusion

Label:
LMCFeatureFusion
query features attend to latent memory

Internal callouts:
- Query = dense image features.
- Key = latent token features Z.
- Value = Z + PE(P - scene center) by default.
- `geokey_norm` can inject geometry into key; default is value-only geometry.
- Outputs fused dense features.

Node Q5: Coordinate prediction head

Label options:

If branch is local-only:
Local ACE coordinate head

If branch is ACE-FCN + GLACE Stage2:
Global conditioning head
concat / residual / FiLM

If final experiment fixes one variant, replace with the exact selected head:
- concat: broadcast g(I), concatenate with local features, GLACE-style head.
- residual: frozen local prediction + gated global coordinate residual.
- FiLM: global feature modulates coordinate-head hidden state.

Node Q6: Dense scene coordinates

Label:
Dense scene coordinates X(u)

Node Q7: Pose solver

Label:
DSAC* / RANSAC pose solver

Node Q8: Output pose

Label:
6-DoF camera pose

## 4. Instantiation lane A: DINOv2 + MapAnything / ACE-G

Purpose:
Show the one-level two-stage pipeline used for DINOv2 + MapAnything memory experiments.

Minimal visual lane:

MapAnything / DINOv2 extracted memory
  -> pooled scene memory bank
  -> GeoLMC
  -> LMCFeatureFusion with DINOv2 query features
  -> ACE coordinate head
  -> DSAC* pose

Training inset:

Stage 1:
raw feature buffer
  -> compressor + fusion + head
  -> reprojection / invalid loss

Stage 2-G:
compressor frozen and cached
raw backbone buffer
  -> on-the-fly fusion
  -> train coordinate head
  -> optionally train fusion in R2 mode

Hard constraints from code:

- ACE-G default S2 buffer stores raw backbone features, not fused features.
- S2 compresses memory with no grad, once per iteration.
- Compressor is trained in Stage 1 only.
- Fusion is trained in Stage 1; in Stage 2 only if `ace_g_fusion_in_s2=True`.
- Supervision is coordinate reprojection / invalid loss, not feature-level alignment.

Optional special case:

`stage1_fused` exists for ACE-FCN Stage2 and stores frozen Stage1 fused features. Do not show it as the default ACE-G S2 behavior.

## 5. Instantiation lane B: ACE-FCN memory + GLACE-style Stage2

Purpose:
Show the two-level two-stage pipeline described by the current paper draft.

Level 1: local ACE-FCN-LMC

Posed training images + GT/sparse depth
  -> frozen ACE-FCN encoder
  -> patch-level features and world scene coordinates
  -> ACE-FCN memory bank
  -> GeoLMC + fusion
  -> local ACE coordinate head
  -> local Stage1 checkpoint

Memory contract:

- memory feature = frozen ACE-FCN encoder feature
- stride = 8
- feature dim = 512
- memory point = patch-level world scene coordinate from GT/sparse/COLMAP depth
- feature_source = ace_fcn
- layers_idx = [ace_fcn]

Level 2: GLACE-style global refinement

Load local Stage1 checkpoint
  -> freeze ACE encoder + compressor + fusion by default
  -> query local features / optionally Stage1 fused features
  -> add GLACE image-level global descriptor
  -> train global conditioning head
  -> refined dense scene coordinates
  -> DSAC* pose

Global head modes:

- `glace_concat`: broadcast global descriptor and concatenate with local features.
- `glace_residual`: frozen Stage1 local prediction plus gated coordinate residual.
- `glace_film`: global descriptor applies FiLM to head hidden state.

Hard constraints from code:

- Stage2 global modes require `--ace_lmc_local_checkpoint_path`.
- That checkpoint must be a pure local Stage1 checkpoint (`ace_lmc_global_head_mode=none`).
- `ace_lmc_freeze_local_stack=True` freezes ACE encoder + compressor + fusion.
- `stage1_fused` feature source is allowed only for `glace_concat` or `glace_residual`, requires frozen local stack, and requires `ace_g_fusion_in_s2=False`.

## 6. GVCS routing inset

Add this because the method section claims visibility-guided configuration selection.

Node G0:
Memory visibility probe

Inputs:
pooled memory points + training camera poses

Computation label:
front-facing visibility score

Formula label:
score = median_front_ratio - low_pose_front_ratio_fraction

Decision:
if requested global and score < threshold
  -> local / hierarchical fallback
else
  -> global mode

Visual:
small decision diamond near the global-conditioning branch.

Important:
This should be drawn as a reliability router, not as a learned neural module.

## 7. Loss and supervision arrows

Main dashed supervision:

Dense scene coordinates
  --reprojection loss with GT pose + intrinsics-->
training objective

Invalid-coordinate penalty:

Invalid / out-of-frame predictions
  --invalid loss-->
training objective

Optional dashed side losses:

- C1 reference-frame auxiliary metric loss.
- multi-frame reprojection / SfM track losses.
- relative-depth consistency.
- Stage2 local teacher consistency / guard losses.
- GLACE anti-regression guard.

Recommendation:
In the main figure, draw only:

- reprojection loss
- invalid loss
- optional small "auxiliary consistency losses" chip

Put the full list in supplementary or caption.

## 8. What not to draw

Do not draw the pooled scene memory bank as K compact tokens before GeoLMC.

Do not imply the compressor runs per query image at evaluation.
Correct: compressor runs once per scene, cached for all test frames.

Do not draw feature-level alignment loss as the core objective.
Correct: main supervision is reprojection / invalid scene-coordinate loss.

Do not draw GLACE global features as part of the memory bank.
Correct: global features are per-query/image-level descriptors.

Do not imply global conditioning always helps.
Correct: paper reports both gains and negative transfer; GVCS routes unreliable global modes away.

Do not mix ACE-FCN memory and MapAnything memory as one identical branch.
Correct: both satisfy pooled memory interface, but their feature sources and stride/dim contracts differ.

Do not show S2 training the compressor.
Correct: compressor is frozen / no-grad in S2.

Do not show default ACE-G S2 buffer as fused features.
Correct: default S2 buffer stores raw backbone features; fusion happens per batch.

Do not expand optimizer, scheduler, GPU, run folder, or log details in the paper figure.

## 9. Revised Mermaid-level draft

This is for quick review only, not the final paper figure.

```mermaid
flowchart LR
  subgraph Offline[Offline scene memory construction]
    M0[Posed training images<br/>RGB-D / sparse depth / SfM depth]
    M1[Memory view selection<br/>+ feature extraction]
    M2[Pooled scene memory bank<br/>{p_i, f_i}, i=1..N]
    M3[Metadata<br/>scene center / poses / scale tokens]
    M4[GeoLMC compressor<br/>geometry-aware cross-attention]
    M5[K latent scene tokens<br/>Z in R^{Kxd}, P in R^{Kx3}]
    M0 --> M1 --> M2 --> M4 --> M5
    M3 -.-> M4
  end

  subgraph Query[Query-time coordinate regression]
    Q0[Query image I]
    Q1[Backbone / encoder<br/>DINOv2 or ACE-FCN/GLACE]
    Q2[Dense query features B(I)]
    Q3[GLACE global descriptor g(I)<br/>optional Stage2 branch]
    Q4[LMCFeatureFusion<br/>dense features attend to latent memory]
    Q5[Coordinate head<br/>local ACE or global conditioning head]
    Q6[Dense scene coordinates X(u)]
    Q7[DSAC* / RANSAC]
    Q8[6-DoF pose]
    Q0 --> Q1 --> Q2 --> Q4 --> Q5 --> Q6 --> Q7 --> Q8
    Q1 -.-> Q3 -.-> Q5
    M5 --> Q4
  end

  subgraph Train[Training / routing]
    T1[Stage 1 local memory training<br/>compressor + fusion + head]
    T2[Stage 2 global refinement<br/>freeze local stack, train global head]
    T3[Reprojection + invalid losses]
    G0[GVCS visibility score<br/>median front ratio - low-pose fraction]
    G1{global reliable?}
    G2[Use global conditioning]
    G3[Fallback to local / hierarchical]
    T1 --> T2
    Q6 -.-> T3
    M2 -.-> G0 --> G1
    G1 --> G2
    G1 --> G3
    G2 -.-> Q5
    G3 -.-> M4
  end
```

## 10. Exact caption draft

Proposed caption:

We build a scene-level pooled memory bank from posed training images and compress it with a geometry-aware latent memory compressor. GeoLMC samples K latent 3D anchors, seeds latent queries with position encodings relative to the scene center, and attends to memory features under geometric bias. At test time, the compressed latent scene tokens are cached once per scene and fused into dense query features by cross-attention before scene-coordinate regression and DSAC*/RANSAC pose solving. The same interface supports a one-level DINOv2/MapAnything instantiation and a two-level ACE-FCN/GLACE instantiation. In the latter, a local ACE-FCN-LMC checkpoint is first trained and then reused by a GLACE-style global conditioning head. A visibility-guided selector disables unreliable global conditioning when memory points are poorly visible from the training poses.

## 11. Concrete edits needed to existing figure files

File: figure_spec.md

- Replace "Scene Memory / Map Codes (K compact tokens)" with "Pooled Scene Memory Bank (N points, features)".
- Add "GeoLMC output: K latent scene tokens (Z, P)" as a separate node.
- Replace "Stage-2 GLACE Concatenation Head" with "Stage-2 Global Conditioning Head (concat / residual / FiLM)" unless the final paper locks to concat only.
- Add GVCS visibility routing inset.
- Decide whether main figure is unified two-lane or GLACE-only.

File: method_architecture.mmd

- Add separate node between `SceneMem` and `Compressor` for `Pooled Memory Bank`.
- Move "K compact tokens" label to the compressor output.
- Add DINOv2/MapAnything lane or explicitly mark the current figure as GLACE-detail-only.
- Add GVCS decision diamond.
- Add note "memory compressed once per scene at eval".

File: image2_prompt.md

- If using image generation as visual reference, update prompt to request a unified framework with two instantiation lanes.
- Keep final publication output vector-based, because AI raster generation is only a layout/style reference.

## 12. Recommended next artifact sequence

Step 1:
Update `figure_spec.md` with the corrected unified spec.

Step 2:
Generate a clean Mermaid draft for logic approval.

Step 3:
Generate draw.io XML for editable layout.

Step 4:
Generate final TikZ standalone figure for LaTeX.

Step 5:
Export SVG/PDF and inspect labels, arrows, and spacing.

## 13. Final review questions for the user

Question 1:
Should Figure 1 be unified two-lane, or should Figure 1 stay GLACE+ACE-FCN only and move DINOv2+MapAnything to an appendix figure?

Question 2:
Which Stage2 global head is the final paper-facing variant: concat, residual, FiLM, or should the figure keep all three as supported modes?

Question 3:
Should GVCS be inside the main method figure, or separated into a small ablation/routing figure?
