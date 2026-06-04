# APR + LMC Design Preparation

Status: preparation note, revised 2026-06-03.

## Positioning Correction

marepo / map-relative pose regression should **not** be used as the main "typical APR" proof. It is closer to a map-conditioned SCR/RPR hybrid: it keeps an explicit scene map representation and replaces the final PnP/DSAC-style solver with a learned transformer pose regressor. That is interesting, but it does not cleanly answer whether LMC helps typical absolute pose regression.

For the paper's APR generality proof, use canonical direct absolute pose regression (APR): image -> pose, without explicit dense scene coordinate prediction or PnP.

## Typical APR Baseline Candidates

Recommended order:

1. **PoseNet-style baseline**
   - Backbone image encoder + direct translation/quaternion pose head.
   - Easiest to implement and explain.
   - Weak baseline, but cleanest demonstration of LMC as scene-memory conditioning.

2. **MapNet-style temporal/geometric APR**
   - Adds geometric/relative constraints over image pairs or sequences.
   - More credible than PoseNet, but requires sequence/pair sampling.
   - Good second-stage APR baseline if PoseNet+LMC looks promising.

3. **MS-Transformer / Transformer APR**
   - Modern direct APR family with transformer aggregation.
   - Better reviewer-facing baseline than plain PoseNet.
   - Use if code is available or implementation effort is acceptable.

4. **TransPoseNet / attention-guided APR**
   - Useful reference and possible stronger APR backbone.
   - More complex than needed for the first LMC proof.

Not first priority:

- **marepo**: map-relative, not typical APR; keep as discussion or separate map-conditioned learned-solver experiment.
- **DFNet**: too heavy because it combines APR with matching/synthetic view ideas.
- **MultiLoc/RPR-style methods**: reference-conditioned RPR, not canonical APR.

## LMC Integration Designs for Typical APR

### Variant A: APR + global LMC scene descriptor

Minimal and clean:

- Query image -> APR backbone descriptor `q`.
- Scene memory -> GeoLMC latent tokens `Z`.
- Pool `Z` into scene descriptor `s` using mean pooling or learned attention.
- Pose head input: `[q, s]`.
- Train APR head and LMC adapter; optionally keep image backbone frozen for the first smoke.

This is the first implementation target.

### Variant B: APR + query-to-memory cross-attention

Stronger but still typical APR:

- Query descriptor tokens attend to LMC latent tokens.
- Output conditioned query descriptor goes into translation/rotation heads.
- No scene coordinate map and no PnP.
- Train adapter + pose heads first; optionally fine-tune compressor.

This is the best paper-quality APR+LMC variant if Variant A shows signal.

### Variant C: APR + memory-conditioned uncertainty/gate

Use LMC only to predict confidence/scale/gating for pose head features:

- Scene descriptor controls pose-head normalization, FiLM, or feature gates.
- This can reduce overfitting and is easy to ablate.
- Lower risk than full cross-attention, but weaker as a memory-use claim.

## Minimal Experiment Matrix

| Method | Trainable parts | Purpose |
|---|---|---|
| PoseNet-style APR | backbone + pose head | canonical direct APR baseline |
| APR + zero/random LMC descriptor | pose head/adapter | capacity/control baseline |
| APR + raw memory descriptor | pose head/adapter | checks whether uncompressed memory helps |
| APR + compressed LMC descriptor | pose head/adapter, frozen or lightly trained compressor | main APR+LMC result |
| APR + LMC cross-attention | adapter + pose heads | stronger variant if descriptor pooling is weak |
| MapNet-style APR + LMC | pair/sequence APR head + adapter | optional stronger APR evidence |

## Dataset Choice

Use a small, standard subset first:

- **7-Scenes** is the best first APR target because APR literature commonly reports it and data is compact.
- **Cambridge** is the second target because it is standard for APR and already part of the paper matrix.
- **Wayspots** can be used later, but direct APR may be weak on mixed outdoor scenes; use it only as a boundary result.

## Implementation Boundary

Do not modify the current SCR trainer first. Create a separate APR subproject that reuses:

- `memory_extraction/` memory files.
- `GeoLMC` compressor from `ace_compressor.py`.
- A lightweight APR dataset wrapper reading ACE-style `rgb/poses/calibration`.
- Metrics: median translation/rotation error and Acc5/Acc10.

Only merge APR code into the main trainer if the APR+LMC result is strong enough to maintain.

## Immediate Next Implementation Plan

1. Implement a minimal PoseNet-style APR baseline on 7-Scenes chess or Cambridge KingsCollege.
2. Add `APR + zero/random LMC descriptor` controls.
3. Add `APR + compressed LMC descriptor` using frozen LMC memory tokens.
4. If positive, replace descriptor pooling with query-to-memory cross-attention.
5. Only then consider MapNet/MS-Transformer strength upgrades.
