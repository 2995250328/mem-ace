# Title: A Reviewer-Proof Experiment Matrix for Latent Memory Compression in Visual Relocalization

## 1. Problem Formulation

We want the final paper to support a precise claim rather than an over-broad one. Let a relocalization method be decomposed into a backbone or local predictor \(B\), a scene memory/map representation \(M\), and a fusion or conditioning operator \(F\). The proposed LMC family learns a compact latent memory \(Z = C(M)\) and injects it into the predictor:

\[
\hat{X}, \hat{T} = RANSAC\left(F(B(I_q), Z, g_q)\right)
\]

for scene-coordinate-regression variants, or

\[
\hat{T} = P(B(I_q), Z, K_q)
\]

for APR/map-relative pose-regression variants. The experimental goal is to prove three claims:

1. LMC improves strong compatible scene-coordinate backbones under hard mapping/query shifts.
2. LMC is not tied to one backbone: it improves DINO-LMC on Indoor6 and GLACE/ACE-FCN-LMC on Wayspots.
3. LMC is a general scene-memory conditioning idea, with at least one lightweight APR/map-relative integration showing positive transfer.

The paper should explicitly distinguish primary claims from boundary results. If DINOACE is weak on a dataset, the paper should report DINO+LMC as a backbone-limited improvement, not as a universal SOTA claim.

## 2. Literature Landscape & Motivation

ACE and DSAC* show that scene coordinate regression can be accurate and efficient, but per-scene regressors can overfit to mapping images and fail under query distribution shifts. ACE-G addresses this by separating scene-specific map codes from a scene-agnostic coordinate regressor and by pre-training with mapping/query splits. It evaluates Indoor-6, RIO10, and Cambridge, and its discussion shows that DINO-style features can be strong indoors but less reliable under some outdoor/OOD conditions.

GLACE adds global-local accelerated coordinate encoding and is a natural high-performing SCR baseline for Wayspots-like scenes. The current project already shows that GLACE+LMC broadly improves Wayspots, which is the strongest evidence that LMC can improve a competitive non-DINO local/global SCR backbone.

Canonical absolute pose regression (APR) directly regresses camera pose from image features, without an explicit scene-coordinate map or PnP solver. This is the cleanest route for a lightweight LMC generality proof. Map-relative pose regression such as marepo is useful background, but it is closer to a map-conditioned learned solver/RPR hybrid than a typical APR baseline, so it should not be the primary APR experiment. APR integration should be positioned as generality evidence, not as the main SOTA claim unless results are unexpectedly strong.

The gap is that current results are strong but fragmented: Indoor6 success is mostly DINO+LMC, Wayspots success is GLACE+LMC, and there is not yet a clean paper-level experiment structure that explains when and why LMC helps.

## 3. Design Space & Selected Experiment Architecture

### 3.1 Dataset Strategy

Use a tiered dataset strategy.

**Tier A: primary claim datasets.**
- Indoor6: main hard indoor mapping/query shift benchmark. Keep DINO+LMC as the strongest result, and add GLACE+LMC to answer reviewer concerns that the method only works with DINO on Indoor6.
- Wayspots: main large outdoor/indoor mixed scene benchmark for GLACE+LMC. Keep full scene coverage and add reliability/negative-transfer ablations.

**Tier B: scale/generalization dataset.**
- Prefer Cambridge over the Naver large indoor dataset for the main paper unless the Naver dataset has a widely recognized public benchmark protocol and easy baseline comparability. Cambridge is not perfect for your method, but it is recognizable and already used by ACE-G. It is valuable as an honest medium-scale/OOD boundary test.
- Use the Naver large indoor dataset as supplementary or appendix if it is public, stable, and not too costly. It can demonstrate scale, but it will not convince reviewers as much as a standard benchmark unless SOTA numbers are available.

**Tier C: APR generality proof.**
- Add a small typical-APR adapter experiment on 7-Scenes or Cambridge. The goal is not to beat every APR method; it is to show that compressed scene memory can condition a direct pose regressor and improve over the same APR backbone without LMC.

### 3.2 Recommended Main Paper Tables

**Table 1: Indoor6 SOTA comparison.**
Rows: ACE, DINOACE, ACE-G, GLACE, DINO+LMC, GLACE+LMC, DINO+LMC+best memory extraction. Metrics: median cm/deg, Acc5, Acc10, mapping/training time, memory size. This is the strongest table.

**Table 2: Wayspots full-scene comparison.**
Rows: DSAC*, ACE, GLACE, DINOACE if available, GLACE+LMC, GLACE+LMC with reliability control. Scenes: all Wayspots scenes, with average. Metrics: 10cm/5deg and 5cm/5deg or the same thresholds used by marepo/GLACE. This is the second strongest table.

**Table 3: Cross-dataset summary.**
Datasets: Indoor6, Wayspots, Cambridge, optionally RIO10 if already runnable. Columns: base backbone, base score, +LMC score, relative gain, SOTA gap. This table should explicitly show that LMC consistently improves its own base, even when the base is not SOTA.

**Table 4: Resource and scalability.**
Rows: ACE/GLACE/DINOACE/LMC variants, plus ACE-G reported/reproduced map-code settings. Columns: memory file size, latent tokens/map codes, token dimension, training time, inference FPS, GPU memory, map build/extraction time. Reviewers will ask whether memory compression is worth it; this table answers that. It should also state that the current default LMC K=64 is much smaller than common ACE-G 1024/4096 map-embedding settings, so compactness vs accuracy is a central claim.

**Table 5: APR generality.**
Rows: PoseNet/MapNet-style APR baseline, APR + zero/random memory descriptor, APR + raw memory descriptor, APR + compressed LMC descriptor, and optionally APR + LMC cross-attention. Use a small number of scenes. Metrics: median cm/deg, Acc5/Acc10, training/fine-tuning time. Keep this as a generality table, not the main headline.

### 3.3 Required Ablations

**Ablation A: memory source and extraction.**
Compare no memory, random memory, DINO/ACE-FCN feature memory, BSE pooled memory, voxel/simple pooled memory, different view selection strategies, and GT-vs-predicted memory if applicable.

**Ablation B: compression capacity.**
This should be upgraded to a paper-critical ablation rather than a small-K sweep. The current default LMC uses K=64; with 512/1024-d features this is roughly 32K/65K latent scalars. Common ACE-G settings use 1024 or 4096 map embeddings, roughly 0.5M/2.1M scalars at 512 dimensions, which is more than an order of magnitude larger. Therefore the main capacity sweep should compare:

- compact LMC: K = 32, 64, 128, 256, 512;
- extended LMC: K = 1024 on 1-2 representative scenes;
- ACE-G-scale reference: K = 4096 on one scene or supplementary only, unless the gain is large and compute is acceptable.

Report accuracy vs memory size/training time/VRAM, and frame the result around whether LMC reaches or exceeds strong baselines while using far fewer scene tokens than ACE-G-style map codes.

**Ablation C: fusion location.**
Backbone-only, decoder/local-only, global concat, local+global, residual global, adaptive/reliability-gated global. This directly addresses SquareBench negative transfer.

**Ablation D: training stage contribution.**
Stage1 only, Stage2 only, S1+S2, no warmup, no query-style split, no final buffer refill, frozen vs trainable fusion/head. This shows the two-stage training is necessary.

**Ablation E: robustness.**
Mapping/query split difficulty, number of memory views, view selection quality, appearance shift, pose noise if available. Indoor6 is ideal for this because the main result is strong.

**Ablation F: failure analysis.**
Report scenes where DINO+LMC improves DINOACE but stays below SOTA. Include qualitative coordinate maps or error histograms. This is important: reviewers trust papers that explain limits.

## 4. Evaluation & Validation Plan

### 4.1 Immediate Priority Order

1. **Indoor6 GLACE+LMC.** This is mandatory. Since DINO+LMC is already outstanding on Indoor6 and GLACE+LMC is strong on Wayspots, the missing cross-check is GLACE+LMC on Indoor6. If positive, it strongly supports backbone-agnostic LMC. If negative, it still defines the method's compatibility boundary.

2. **Scene-token capacity ablation.** Immediately add an Indoor6 scene3/scene4a K sweep: 64, 128, 256, 512, with 1024 if needed. This answers whether current LMC is limited by an overly small scene-token budget and supplies the key evidence for the Resource/Scalability table. If K=256/512 is clearly better than 64, use the best K for all-scene GLACE+LMC. If larger K has little gain, emphasize compact memory.

3. **Complete Wayspots GLACE+LMC with reliability controls.** Wayspots already has improvements, but the paper needs full-scene averages, threshold tables, per-scene deltas, and SquareBench/Bears negative-transfer control. Use `stage2_global_reliability` as the design dependency.

4. **Cambridge as medium-scale/OOD benchmark.** Run DINOACE, DINO+LMC, GLACE, GLACE+LMC if feasible. The expected result may not be SOTA; the purpose is to show whether LMC improves the base and to align with ACE-G's Cambridge protocol.

5. **APR adapter proof.** Implement only after the main SCR tables are stable. Choose the smallest integration that compares a typical direct APR baseline vs APR+LMC on 7-Scenes or Cambridge. This should be in the supplementary unless strong.

6. **Naver large indoor dataset.** Use only if it is public, has clear splits, and can be described cleanly. Otherwise it is a supplement/demo, not a main paper pillar.

### 4.2 What to Run on Each Dataset

**Indoor6.**
- DINOACE baseline.
- DINO+LMC best.
- GLACE baseline.
- GLACE+LMC.
- ACE-G reported or reproduced if possible.
- K=64/128/256/512 capacity ablation on scene3 and scene4a, with optional single-scene K=1024/4096; view-count ablation only on representative scenes, not all scenes.
- Qualitative coordinate reconstruction for 2 hard scenes.

**Wayspots.**
- GLACE baseline.
- GLACE+LMC full all-scene run.
- GLACE+LMC reliability-gated variant.
- Zero/random memory controls on Bears and SquareBench.
- Full table across scenes; detailed ablations only on Bears, SquareBench, and one average/neutral scene.

**Cambridge.**
- DINOACE vs DINO+LMC.
- GLACE vs GLACE+LMC if implementation cost is manageable.
- One memory capacity setting only. Do not spend too much compute on exhaustive Cambridge ablations.
- Report honestly as OOD/scale boundary.

**Naver large indoor.**
- Only run DINOACE vs DINO+LMC and GLACE vs GLACE+LMC if data loading is stable.
- Use as scale demonstration: memory size, extraction time, FPS, qualitative maps.
- Do not make it a primary SOTA table unless it has public baselines.

**APR adapter.**
- Baseline PoseNet-style APR first; MapNet/MS-Transformer-style APR if code/compute allow.
- +zero/random memory descriptor controls.
- +raw scene memory descriptor.
- +compressed LMC memory descriptor.
- Optional LMC cross-attention adapter and fine-tuning.
- Use 7-Scenes or Cambridge first; use Wayspots only as a boundary result.

## 5. Expected Failure Modes & Engineering Risks

The main risk is over-claiming. If DINO+LMC is weak outside Indoor6, the paper must frame this as backbone-limited and use GLACE+LMC as the stronger general SCR evidence. Another risk is dataset overload: running Cambridge, Naver, APR, and all ablations can delay the paper without improving the central claim. The plan should prioritize Indoor6 GLACE+LMC and full Wayspots reliability first.

A second risk is negative transfer from global features. The SquareBench result already shows that raw or too-strong global injection can be harmful. Reliability controls, zero/random global controls, and Stage1 consistency are needed before claiming robust GLACE+LMC.

A third risk is unfair baselines. Every main table should separate reproduced numbers from reported numbers, list training time and memory size, and use the same thresholds as prior papers where possible.

## 6. Reuse Plan

Use `../trainer_dinov2_lmc.py`, `../options_dinov2_lmc.py`, and `../test_ace_dinov2_lmc.py` as the central implementation/evaluation path for DINO-LMC and GLACE/ACE-FCN-LMC. Use `../stage2_global_reliability/01_design/proposal.md` to guide the Wayspots reliability variants. Use `../scripts/run_squarebench_stage2_global_gate_matrix.sh` and `../scripts/summarize_wayspots_ace_fcn_lmc_suite.py` as templates for scene-level orchestration and reporting. Use `../../papers/Bruns 等 - 2025 - ACE-G Improving Generalization of Scene Coordinate Regression Through Query Pre-Training.pdf` for Indoor6/RIO10/Cambridge positioning. For APR, use typical direct APR baselines such as PoseNet/MapNet/MS-Transformer/TransPoseNet; keep marepo only as map-relative learned-solver related work.

## 7. Final Recommended Experiment Matrix

### Must-have before paper submission

| Block | Dataset | Methods | Purpose |
|---|---|---|---|
| Main SCR-1 | Indoor6 | DINOACE, DINO+LMC, GLACE, GLACE+LMC, ACE-G/GLACE reports | Prove strongest SOTA claim and cross-backbone Indoor6 validity. |
| Main SCR-2 | Wayspots | GLACE, GLACE+LMC, reliability-gated GLACE+LMC, zero/random controls | Prove broad gains and handle negative transfer. |
| Ablation-1 | Indoor6 scene3/scene4a | scene-token K=64/128/256/512, optional 1024/4096 ACE-G-scale reference | Prove LMC capacity/compactness trade-off and choose the all-scene main-experiment K. |
| Ablation-2 | Indoor6 subset | memory source, view count, S1/S2 | Explain why DINO+LMC works. |
| Ablation-3 | Bears/SquareBench | global gate, residual, Stage1 consistency, zero/random global | Explain why GLACE+LMC is robust. |
| Efficiency | Indoor6 + Wayspots | memory size, latent-token/map-code count, time, FPS, VRAM | Prove compact-memory practicality and align with ACE-G map-code capacity. |

### Strongly recommended if compute allows

| Block | Dataset | Methods | Purpose |
|---|---|---|---|
| Boundary | Cambridge | DINOACE vs DINO+LMC; GLACE vs GLACE+LMC if feasible | Standard benchmark and honest OOD boundary. |
| Generality | 7-Scenes or Wayspots subset | APR baseline vs APR+LMC | Show LMC is not SCR-only. |

### Optional supplementary

| Block | Dataset | Methods | Purpose |
|---|---|---|---|
| Scale demo | Naver large indoor | base vs +LMC, resource table, qualitative maps | Demonstrate large-scene feasibility if public protocol is weak. |
| RIO10 | RIO10 | DINOACE/DINO+LMC if easy | Align with ACE-G, but not necessary if compute is tight. |

## 8. Direct Answers to the User's Questions

**Should Indoor6 add GLACE+LMC?** Yes. This is the most important missing experiment. It closes the logical triangle: DINO+LMC works on Indoor6, GLACE+LMC works on Wayspots, and GLACE+LMC on Indoor6 tests whether LMC's Indoor6 success is not DINO-specific.

**Does Wayspots need more experiments?** Yes, but not endless sweeps. Complete full-scene GLACE+LMC tables, add zero/random controls, and add reliability/negative-transfer ablations on Bears and SquareBench. Do not keep sweeping scalar gate only.

**Cambridge or Naver large indoor?** Choose Cambridge for the main paper because it is standard and comparable. Use Naver as supplementary scale evidence unless it has public baselines and clean splits. If only one can be run, run Cambridge.

**Should LMC be connected to APR?** Yes, but as a lightweight generality proof after the main SCR story is stable. Use typical direct APR first, e.g. PoseNet/MapNet/MS-Transformer-style baselines, not marepo as the main APR evidence. The minimum useful result is APR baseline vs APR+zero/random memory vs APR+raw memory vs APR+compressed LMC memory on a small standard subset. Do not let APR integration delay the core paper.

**What is the final narrative?** LMC is a compact scene-memory conditioning mechanism. It gives SOTA-level gains when the base relocalizer is compatible and the dataset has strong mapping/query shifts, and it consistently improves base models even when the base backbone itself is not SOTA. The paper is strongest if it is honest about this boundary.