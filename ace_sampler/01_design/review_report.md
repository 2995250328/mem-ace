# Critical Review Report (Red Team)

## 1. Claim Restatement

The proposal claims that replacing ACE's random buffer sampling with a confidence-guided
policy (SamplerNet trained on reprojection error + optional MC Dropout uncertainty) will
improve localization accuracy. The method is two-phase: train SamplerNet offline on a frozen
head, then use it during Phase 2 buffer filling.

## 2. Reviewer Stance

**Partially Agree** — The motivation is sound but the evaluation plan is insufficient to
establish the claim, and there are two structural weaknesses that could nullify the benefit.

## 3. Primary Critiques

### 3.1 Distribution Shift Between Phase 1 and Phase 2 (Optimization Dynamics)

SamplerNet is trained to predict confidence under head $g_\phi$ (Phase 1). In Phase 2, a
**new** head $g_{\phi'}$ is trained from scratch. The confidence map from Phase 1 reflects
$g_\phi$'s error distribution, not $g_{\phi'}$'s. If $g_\phi$ was trained on a different
random seed or subset, the confidence map may actively mislead Phase 2 sampling.

The proposal acknowledges this but dismisses it with "Part B random sampling." This is
insufficient: if `sampler_ratio=0.7`, 70% of samples are guided by a potentially stale map.
The authors must show empirically that the Phase 1 head's confidence map transfers to Phase 2.

### 3.2 Confirmation Bias Amplification

Top-k confidence sampling preferentially selects pixels the model already handles well
(low reprojection error → high confidence). This is the **opposite** of hard example mining.
The method may accelerate convergence on easy regions while starving the head of supervision
on hard regions (textureless surfaces, repetitive patterns, low-texture areas).

The proposal frames this as a feature ("prefer pixels the model localizes well") but provides
no theoretical justification for why this improves final accuracy vs. a curriculum that
focuses on hard examples.

### 3.3 MC Dropout Uncertainty is Poorly Calibrated at Initialization

The UncertaintyHead is pre-trained via 2-epoch distillation from the frozen ACE head.
Two epochs of MSE distillation on a lightweight head will produce a deterministic approximation
of the frozen head — the MC Dropout variance will be near-zero everywhere initially.
The variance only becomes meaningful after the head has seen diverse failure modes.
This undermines the utility of the uncertainty term in the confidence target.

### 3.4 Computational Overhead Not Quantified

Phase 1 requires a full pass over the training set with T=10 MC Dropout samples per image.
For a scene with 1000 training images, this is 10,000 forward passes through the encoder +
UncertaintyHead. The proposal does not report Phase 1 wall-clock time or compare it to the
time saved (if any) in Phase 2.

## 4. Mandatory Counterargument (Failure Mode)

**Scenario**: Scene with highly repetitive texture (e.g., 7-Scenes `stairs`).

The frozen ACE head $g_\phi$ achieves low reprojection error on the dominant texture pattern
but fails on the few distinctive keypoints. SamplerNet learns to assign high confidence to
the dominant pattern. Phase 2 buffer is 70% filled with features from the dominant pattern.
The new head $g_{\phi'}$ overfits to the dominant pattern and performs **worse** than the
random baseline on the distinctive keypoints that are critical for disambiguation.

This is not a corner case — `stairs` is the hardest 7-Scenes scene precisely because of
repetitive texture. The method may degrade on the scenes where improvement is most needed.

## 5. Mandatory Action Items

1. **Ablate `sampler_ratio`** across {0.3, 0.5, 0.7, 0.9} on at least 3 scenes including
   `stairs`. Report whether the benefit degrades as ratio increases. This directly tests
   the confirmation bias hypothesis.

2. **Report Phase 1 wall-clock time** and compare total training time (Phase 1 + Phase 2)
   vs. baseline ACE. If Phase 1 takes longer than the accuracy gain justifies, the method
   is not practically useful.

3. **Test with a randomly initialized SamplerNet** (no Phase 1 training) as an additional
   ablation. If random confidence maps perform similarly to trained ones, the Phase 1
   training signal is not meaningful.
