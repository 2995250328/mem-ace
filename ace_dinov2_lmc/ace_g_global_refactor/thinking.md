# PMRF / G-PMRF Design Evaluation and Handoff

Updated: 2026-06-12

Purpose: this document is a handoff note for evaluating the current PMRF / G-PMRF design. It is not a general final-paper roadmap and it does not cover other candidate directions such as SfM-track supervision or zero-gated residual memory refinement. The only question here is:

> Should we continue PMRF / G-PMRF, and if yes, what is the minimum scientifically valid next step?

## 1. Executive Conclusion

PMRF-v1 is **not yet an established contribution**. The current evidence does not support the claim that unconditional progressive memory re-reading is better than a single memory read.

The more interesting idea is **G-PMRF**: prediction-conditioned geometry-guided progressive memory re-reading. It has a clear task-specific novelty because the second memory read is conditioned on a coarse scene-coordinate hypothesis and the 3D anchors of compressed memory tokens.

However, G-PMRF has a long evidence chain and high engineering cost. It should be treated as a **promising but deferred architecture research direction**, not as the safest immediate paper-stage innovation.

The next valid step is not to implement a large G-PMRF stack. The next valid step is:

```text
E0: Single memory read baseline
E1: PMRF-v1 unconditional second memory read
E2: Parameter-matched FFN residual control
+ full memory/attention/geometry diagnostics
```

Only if PMRF-v1 beats or differs meaningfully from the parameter-matched FFN control should G-PMRF be implemented.

## 2. Current Experimental Evidence

### 2.1 Bears Stage1: Single vs PMRF-v1

The current Stage1 evidence is neutral to negative for PMRF-v1.

| Metric | Single | PMRF-v1 | PMRF - Single |
|---|---:|---:|---:|
| 25cm / 5deg | 94.48 | 93.97 | -0.51 |
| 10cm / 5deg | 90.17 | 88.10 | -2.07 |
| 5cm / 5deg | 79.48 | 78.97 | -0.51 |
| 2cm / 2deg | 10.34 | 10.52 | +0.18 |
| 1cm / 1deg | 0.86 | 1.21 | +0.35 |
| Median rotation | 1.024deg | 1.014deg | -0.010deg |
| Median translation | 3.028cm | 3.099cm | +0.071cm |

Interpretation:

- PMRF-v1 is worse at 25cm / 10cm / 5cm thresholds.
- It gives tiny gains only at very strict thresholds.
- Median rotation is slightly better, but median translation is worse.
- This does not prove useful memory re-reading.

The correct conclusion is:

> PMRF-v1 may change the representation and may help a small subset of strict cases, but current Stage1 results do not support unconditional second-read as a robust improvement.

### 2.2 Stage2 results are not proof of PMRF routing

The recorded Stage2 comparison is mixed.

| Metric | Single Stage2 | PMRF Stage2 | PMRF - Single |
|---|---:|---:|---:|
| 25cm / 5deg | 97.41 | 97.76 | +0.35 |
| 10cm / 5deg | 90.86 | 91.90 | +1.04 |
| 5cm / 5deg | 82.41 | 81.72 | -0.69 |
| Median rotation | 0.8838deg | 0.8624deg | -0.0214deg |
| Median translation | 3.066cm | 2.990cm | -0.076cm |

Interpretation:

- Stage2 improves wide thresholds and medians.
- Stage2 hurts 5cm / 5deg.
- Stage2 uses a frozen local representation plus a GLACE/global head adaptation.
- Therefore Stage2 improvement does **not** prove that PMRF learned better Stage1 memory routing.

Stage2 can be treated only as a weak clue that PMRF representations might contain information useful to a later head. It is not evidence that PMRF-v1 itself is a valid memory re-reading mechanism.

## 3. PMRF-v1 Structure and Ambiguity

The current PMRF-v1 structure is approximately:

```text
K, V = MemoryProjection(memory_feature, memory_point)

A1 = softmax(FirstQuery(Q) @ K^T / sqrt(d))
H1 = FirstReadAndFFN(Q, A1 @ V)

A2 = softmax(SecondQuery(LN(H1)) @ K^T / sqrt(d))
Delta2 = SecondOutput(A2 @ V)
H2 = LN(H1 + Delta2)
```

The second read reuses the same memory keys and values but changes the query through `LN(H1)`.

The unresolved questions are:

1. Does A2 actually change memory routing in a useful way?
2. Is A2 better than simply adding parameters and another nonlinear residual block?
3. Does A2 improve hard or ambiguous patches, or does it disturb already correct patches?
4. Are attention changes correlated with reprojection / coordinate / pose improvements?

The most important confound is parameter count and extra nonlinearity. PMRF-v1 adds a second query projection, output projection, residual update, and normalization. Without a parameter-matched non-memory control, any improvement cannot be attributed to memory re-reading.

## 4. Required Control: Parameter-Matched FFN

A mandatory control is:

```text
H2 = LN(H1 + Adapter(LN(H1)))
```

The adapter should roughly match the extra parameter count of the second-read branch, especially the second query/output projections.

Purpose:

- test whether PMRF gains come from memory re-reading;
- rule out ordinary residual capacity;
- rule out extra LayerNorm / FFN effects;
- provide a fair baseline before G-PMRF.

Decision logic:

| Result | Interpretation |
|---|---|
| PMRF-v1 > FFN and PMRF-v1 > Single | Memory re-reading may be useful. Proceed to G-PMRF. |
| PMRF-v1 ~= FFN > Single | Extra capacity helps, not memory re-reading. Do not claim PMRF. |
| FFN > PMRF-v1 | Second memory read is likely harmful or noisy. Stop PMRF. |
| PMRF-v1 ~= FFN ~= Single | No evidence. Stop or defer. |
| PMRF-v1 < Single | Unconditional second read is unsafe. Only consider G-PMRF if geometry diagnostics strongly motivate it. |

## 5. Diagnostics Required Before G-PMRF

The next experiment must include runtime diagnostics. Metrics alone are not enough.

### 5.1 Attention distribution diagnostics

Record for A1 and A2:

```text
entropy
effective token count
average max attention
top-1 agreement
top-k overlap
JS divergence between A1 and A2
```

Interpretation caution:

- Lower entropy is not automatically better.
- Higher JS divergence is not automatically better.
- Attention confidence is not a reliable proxy for geometric correctness.

### 5.2 Feature update diagnostics

Record:

```text
norm(Delta2) / norm(H1)
cosine(H1, H2)
RMS of Delta2 before/after output projection
LayerNorm effect size
```

These are needed because a residual branch can appear gated or small before projection but still produce a large effective feature change.

### 5.3 Memory geometry diagnostics

For memory token anchors `P_j`, compute:

```text
mu1_i = sum_j A1_ij * P_j
mu2_i = sum_j A2_ij * P_j
spread1_i = sum_j A1_ij * ||P_j - mu1_i||^2
spread2_i = sum_j A2_ij * ||P_j - mu2_i||^2
```

Record:

```text
spread(A1, P)
spread(A2, P)
||mu2 - mu1||
||mu1 - predicted coordinate||
||mu2 - predicted coordinate||
```

If ground-truth scene coordinates or SfM sparse points are available for diagnostic pixels, also record:

```text
||mu1 - X_gt||
||mu2 - X_gt||
selected-token distance to X_gt
top-k token distance distribution
```

### 5.4 Correlation with error improvement

The key evidence is not merely that A2 differs from A1. The key evidence is whether A2 changes correlate with better geometry.

Record correlations between:

```text
change in memory centroid distance
change in spread
change in reprojection error
change in coordinate error where available
change in pose success / inlier statistics
```

If centroid/spread changes do not correlate with reprojection or pose improvement, PMRF should not be claimed as useful memory re-reading.

## 6. G-PMRF Design: What Is Promising

G-PMRF means:

> The first compressed 3D memory read predicts a coarse scene coordinate and geometric uncertainty; the second memory read is biased toward memory tokens geometrically consistent with that prediction.

This is different from generic iterative attention because the memory tokens have explicit 3D anchors.

### 6.1 Core idea

Definitions:

```text
H1_i : first-read feature for query patch i
X1_i : coarse scene coordinate for patch i
P_j  : 3D anchor of compressed memory token j
K_j  : memory key
V_j  : memory value
A1_ij: first-read attention
```

Feature score:

```text
S_feat_ij = dot(Wq2(LN(H1_i)), K_j) / sqrt(d)
```

Geometry distance:

```text
D_ij = ||X1_i - P_j|| / scene_scale
```

Soft geometry bias:

```text
B_geo_ij = -D_ij^2 / (2 * sigma_i^2)
```

Second read:

```text
A2_i = softmax(S_feat_i + lambda_i * B_geo_i)
Delta2_i = Wo2(sum_j A2_ij * V_j)
H2_i = LN(H1_i + Delta2_i)
X2_i = StudentHead(H2_i)
```

The geometry bias must be soft. `P_j` is only a compressed-token anchor, not a guaranteed exact surface point.

### 6.2 Why this is novel enough if it works

The possible contribution is not “we add a second attention layer.” That is too generic.

The possible contribution is:

> Prediction-conditioned geometric re-reading of compressed 3D scene memory.

The task-specific ingredients are:

- memory token has a 3D anchor;
- first read predicts a coordinate hypothesis;
- second-read logits are biased by the distance between the hypothesis and memory anchors;
- the geometry bias is normalized by scene scale and uncertainty;
- routing geometry changes are measured and linked to pose/reprojection gains.

## 7. G-PMRF Main Risk: X1 Reliability

The biggest risk is that the coarse coordinate `X1` is unreliable, especially early in training.

Do not directly use a jointly trained `Head(H1)` as a trusted geometry source. The model can jointly change H1, the head, and the second-read branch to bypass the intended geometry constraint.

The clean first G-PMRF design is therefore a frozen teacher setup.

### 7.1 Frozen teacher setup

```text
Frozen teacher:
Q -> Frozen FirstRead -> H1_teacher -> Frozen Head -> X1

Student:
Q -> Student FirstRead -> H1
H1, X1, memory points P -> Geometry-Guided SecondRead -> H2
H2 -> Student Head -> X2
```

Recommended setup:

1. Train or load a converged Single checkpoint.
2. Initialize teacher from the Single checkpoint.
3. Freeze teacher first-read fusion and teacher coordinate head.
4. Initialize student from the same checkpoint.
5. First freeze student first-read and train only second-read branch plus student head.
6. Only if positive, unfreeze student first-read with lower learning rate.

This isolates the question:

> Given the same first-read representation and a stable coarse coordinate, can geometry-guided second-read improve predictions?

## 8. G-PMRF First Experiment

Do not start with adaptive gates, guard losses, hard top-k, routing losses, or STGS.

The first G-PMRF experiment should be:

```text
E3: Frozen single anchor + G-PMRF with constant lambda and sigma
```

Only one new mechanism:

```text
add geometry bias to A2 logits
```

Recommended minimal hyperparameters:

```text
lambda = 0.5
sigma  = 0.2  # normalized by scene scale
```

After one sanity run, a small matrix is acceptable:

```text
lambda in {0.25, 0.5, 1.0}
sigma  in {0.10, 0.20, 0.40}
```

But this is already 9 configurations. Do not run the full matrix unless the single sanity run is non-negative and diagnostics make sense.

## 9. Adaptive Geometry Should Come Later

Only after constant-bias G-PMRF is positive should adaptive variants be tried.

Possible adaptive sigma:

```text
sigma_i^2 = sigma_min^2 + beta * spread1_i
```

Possible adaptive lambda:

```text
lambda_i = lambda_max / (1 + spread1_i / tau)
```

where `tau` comes from warmup statistics such as median spread.

Do not use training-time ground-truth reprojection error as an input to gate/lambda because it is not available at inference. If adaptive routing uses any features, they must be inference-available:

```text
entropy(A1)
spread(A1, P)
sigma_i
norm(Delta2) / norm(H1)
```

## 10. Later Additions and Preconditions

### 10.1 Geometry-uncertainty gate

Only add after E3/E4 is positive.

Possible form:

```text
g_i = sigmoid(MLP(
    entropy(A1_i),
    spread(A1_i, P),
    sigma_i,
    lambda_i,
    norm(Delta2_i) / norm(H1_i)
))

H2_i = LN(H1_i + g_i * Delta2_i)
```

Caution:

- alpha/gate alone may not bound the effective update because output projection and LayerNorm can rescale changes.
- If needed, normalize Delta2 by relative RMS before applying gate.

### 10.2 Frozen teacher reprojection guard

Only a frozen teacher provides a stable guard.

```text
err_teacher = reprojection_error(X_teacher)
err_student = reprojection_error(X_student)
L_guard = relu(err_student - err_teacher - margin)
```

Do not use a jointly trained H1 prediction as teacher; it can move during training and turn the guard into a self-referential constraint.

### 10.3 Memory-anchor coordinate residual

Only after A2 has reliable geometry meaning:

```text
anchor_i = sum_j A2_ij * P_j
offset_i = OffsetHead(H2_i, anchor_i, X1_i)
X2_i = anchor_i + scene_scale * tanh(offset_i)
```

Use scene-scale-normalized residuals, not fixed meter-scale caps, to avoid scene-size sensitivity.

### 10.4 SfM-track / STGS consistency

If combining with track supervision later, do not force token distributions from two views to be identical. Different views may read different tokens for the same physical point.

Prefer centroid-level consistency:

```text
mu_i = sum_k A2_ik * P_k
mu_j = sum_k A2_jk * P_k
L_track_memory = ||mu_i - mu_j||
L_track_coord  = ||X_i - X_j||
```

This belongs after G-PMRF is independently validated.

## 11. Stop Conditions

Stop PMRF/G-PMRF if any of these hold:

1. PMRF-v1 and G-PMRF do not consistently beat parameter-matched FFN.
2. PMRF-v1 is worse than Single on main metrics and diagnostics do not show useful routing behavior.
3. A2 centroid/spread changes do not correlate with reprojection, coordinate, or pose improvement.
4. Positive signals appear only in Stage2, while Stage1 coordinate/pose metrics remain neutral or negative.
5. Gains are smaller than evaluation noise and not reproducible across repeated runs/seeds.
6. G-PMRF only works after adding gate/guard/top-k/STGS before the basic constant-bias version works.

If stopped, the correct interpretation is:

> Current memory rereading is not proven useful. Future work should revisit memory-token geometry, token-anchor construction, or single-read capacity rather than stacking more PMRF mechanisms.

## 12. Recommended Next Step

The next concrete step should be exactly:

```text
E0: Single
E1: PMRF-v1
E2: Parameter-matched FFN
```

with full diagnostics:

```text
A1/A2 entropy
effective token count
attention max
JS divergence
top-k overlap
Delta2/H1 norm ratio
cos(H1,H2)
3D attention centroid and spread
correlation with reprojection / pose improvements
```

Do not implement G-PMRF before this.

If E2 matches or beats E1, PMRF should be deferred.

If E1 clearly beats E2 and diagnostics show meaningful memory-rereading behavior, then implement:

```text
E3: Frozen single teacher + constant lambda/sigma G-PMRF
```

## 13. Final Assessment

PMRF/G-PMRF is scientifically interesting and better thought-out than a generic second attention layer. The G-PMRF version has real task-specific novelty because it conditions memory retrieval on a predicted 3D coordinate and explicit memory-token geometry.

However, it is not yet proven and should not be treated as a current paper contribution. Its evidence chain is long:

```text
PMRF-v1 > Single
PMRF-v1 > parameter-matched FFN
G-PMRF > PMRF-v1
A2 geometry improves
geometry diagnostics correlate with pose/reprojection gains
```

Until this chain is established, the right status is:

> promising deferred architecture track, not active mainline.

For handoff to another conversation, the key instruction is:

> Do not start by adding gates, guards, STGS, adaptive sigma/lambda, or coordinate-anchor residuals. First run the Single / PMRF-v1 / parameter-matched FFN control with full diagnostics. Only then decide whether G-PMRF deserves implementation.
