# Title: Confidence-Guided Buffer Sampling for Scene Coordinate Regression

## 1. Problem Formulation

**Task**: Given a set of training images $\{I_i\}$ with known poses $\{T_i\}$ and a frozen
ACE encoder $f_\theta$, learn a sampling policy $\pi: \mathbb{R}^{C \times H_f \times W_f} \to [0,1]^{H_f \times W_f}$
that selects which pixel-level features to store in the training buffer.

**Inputs**:
- Encoder features $F_i = f_\theta(I_i) \in \mathbb{R}^{512 \times H_f \times W_f}$
- Ground-truth pose $T_i \in SE(3)$, intrinsics $K_i \in \mathbb{R}^{3 \times 3}$
- Pre-trained ACE head $g_\phi$ (frozen during Phase 1)

**Objective**: Maximize localization accuracy of the ACE head trained on the guided buffer,
compared to a head trained on a uniformly-sampled buffer of the same size.

**Confidence target**:
$$c_{ij} = \exp\!\left(-\alpha \cdot e_{ij} - \beta \cdot \sigma^2_{ij}\right)$$
where $e_{ij}$ is the reprojection error (px) of pixel $j$ in image $i$ under the frozen head,
and $\sigma^2_{ij}$ is the MC Dropout reprojection variance (px²). $\beta = 0$ disables uncertainty.

## 2. Literature Landscape & Motivation

**ACE (Brachmann et al., CVPR 2023)**: Trains a scene-specific head on a GPU buffer of
randomly sampled encoder features. Random sampling is the baseline we improve.

**Active Learning / Curriculum Learning**: Bengio et al. (2009) show that ordering training
samples by difficulty improves convergence. Our method is a spatial analog: prefer pixels
where the current model is uncertain or wrong.

**MC Dropout (Gal & Ghahramani, ICML 2016)**: Enables uncertainty estimation via stochastic
forward passes. We use it as a secondary signal to identify structurally ambiguous regions.

**Research gap**: ACE's buffer filling is the only component that remains purely random.
Replacing it with a learned policy is a natural extension that has not been explored.

## 3. Design Space & Selected Architecture

**Alternative A — Online hard example mining**: Re-weight buffer samples by loss during
training. Requires modifying the training loop and introduces instability.

**Alternative B — Offline confidence map (selected)**: Train a separate lightweight network
to predict confidence from encoder features. Plug into buffer filling as a pre-processing step.
Clean separation of concerns; no changes to the ACE training loop.

**Alternative C — Attention-based sampling**: Use self-attention over the feature map.
Too expensive for per-image inference during buffer filling.

**Selected**: Alternative B. SamplerNet is a depthwise-separable CNN (~80K params) that
predicts a confidence map from encoder features. Trained via MSE against the confidence target.

### SamplerNet Architecture
```
Input:  (B, 512, Hf, Wf)
Block1: DWConv(512) → PWConv(128) → ReLU
Block2: DWConv(128) → PWConv(64)  → ReLU
Head:   Conv1x1(64 → 1) → Sigmoid
Output: (B, 1, Hf, Wf)  ∈ [0, 1]
```

### UncertaintyHead (optional)
```
Input:  (B, 512, Hf, Wf)
        Conv1x1(512 → 256) → Dropout2d(p) → ReLU → Conv1x1(256 → 3)
Output: (B, 3, Hf, Wf)  — scene coordinate prediction
```
Run T=10 stochastic passes → variance of reprojected 2D positions = uncertainty map.
Pre-trained via 2-epoch distillation from the frozen ACE head.

### Phase 2 Sampling Strategy
For each image, draw `samples_per_image` points:
- **Part A** (`ratio × spi`): top-k pixels by SamplerNet confidence (masked by valid region)
- **Part B** (`(1-ratio) × spi`): uniform random (cold-start robustness)

## 4. Evaluation & Validation Plan

**Datasets**: 7-Scenes (chess, fire, heads, office, pumpkin, redkitchen, stairs), Cambridge Landmarks

**Metrics**: Median translation error (cm), median rotation error (°), % frames within 5cm/5°

**Baselines**:
1. ACE baseline (random sampling, same buffer size)
2. ACE + SamplerNet (reprojection error only, β=0)
3. ACE + SamplerNet + MC Dropout (full method)

**Ablations**:
- `sampler_ratio` ∈ {0.5, 0.7, 0.9}
- `sampler_alpha` ∈ {0.05, 0.1, 0.2}
- `mc_samples` ∈ {5, 10, 20}

## 5. Expected Failure Modes & Engineering Risks

**Circular dependency**: SamplerNet is trained on a fixed head, but Phase 2 trains a new head.
If the new head diverges significantly, the confidence map may be stale.
*Mitigation*: Part B random sampling (ratio < 1.0) ensures coverage.

**Cold-start**: Early in Phase 2, the new head has high uniform error everywhere.
The confidence map from Phase 1 may not reflect the new head's error distribution.
*Mitigation*: The Phase 1 head should be well-trained (≥ a few epochs).

**Confirmation bias**: Sampling high-confidence regions may reinforce already-good predictions
while neglecting hard cases (e.g., textureless walls, repetitive patterns).
*Mitigation*: Part B random sampling; tune `sampler_ratio` down if needed.

## 6. Reuse Plan

| Module | Source |
|---|---|
| Frozen ACE regressor | `ace_network.py:Regressor` (unchanged) |
| Training data loader | `dataset_origin.py:CamLocDataset` (unchanged) |
| Pixel grid | `ace_util.py:get_pixel_grid` (unchanged) |
| Phase 2 buffer filling | Extends `ace_trainer.py:_fill_buffer` logic |
| SamplerNet, UncertaintyHead, SamplerTrainer | New code in `ace_sampler/` |
