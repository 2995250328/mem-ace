# Draw.io Generation Prompt: Mem-ACE Method Figure

Use this prompt as the high-level instruction for a Codex agent that will create a draw.io figure. The final asset should be vector/editable, not a raster-only generated image.

```text
Create a clean academic method diagram for the paper method "Mem-ACE".

Canvas and style:
- Landscape two-panel figure on a white background.
- Panel (a) is the main framework and takes about 65% of the width.
- Panel (b) is a smaller module detail panel and takes about 35% of the width.
- Use blue for query features, green for scene memory/tokens, orange for coordinates/projection, and gray for training supervision.
- Use short labels, thin arrows, no decorative gradients, no dataset tables, no GPU or optimizer details.

Panel (a): Mem-ACE framework
Draw this main flow:
Posed training images + SfM/STGS geometry -> Offline Memory Construction -> Pooled Scene Memory {p_i, f_i}, scene center c -> Geo-token Compressor -> Anchored Scene Tokens (Z, P).
In parallel draw:
Query Image -> SCR Backbone / Encoder -> Dense Query Features Q.
Then combine:
Dense Query Features Q + Anchored Scene Tokens (Z, P) + PE(P-c) -> Memory Fusion -> SCR Coordinate Head -> Dense Scene Coordinates.
Optionally add a small secondary block after Dense Scene Coordinates: PnP/RANSAC Pose Solver. It must look downstream, not like the main innovation.

Training band under Panel (a):
Geometric Alignment: single-view reprojection.
Cross-view Track Reprojection: source-target SfM/STGS tracks.
Cached Memory Adaptation: cache anchored scene tokens, adapt fusion/head.
Use dashed arrows from Dense Scene Coordinates to the supervision band.

Panel (b): Scene-anchored memory module
Draw:
Memory Points X + Memory Features F -> Geo-token Compressor -> latent features Z and latent positions P.
Draw scene center c and an explicit block PE(P-c).
Draw Query Features Q + Z + PE(P-c) -> Memory Fusion -> Fused Query Features -> Coordinate Head.

Hard constraints:
- Use the method name Mem-ACE.
- Use "Geo-token Compressor" as the paper-facing module name; title the figure with Mem-ACE.
- Pooled Scene Memory is N points/features. Anchored Scene Tokens (Z, P) are K=64 tokens after compression.
- Draw `Z + PE(P-c)` as the default value-geometry fusion path selected by the long-run ablation; do not render it as optional or tentative.
- Do not use code-stage labels, downstream global-refinement internals, branch-specific head variants, DINO layer selection, global scale tokens, fusion refinement variants, or dataset-specific lanes.
- Do not draw P as a final coordinate prediction. P is a latent 3D anchor used through PE(P-c).
```
