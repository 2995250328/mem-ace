# PMRF / G-PMRF-Lite Current Judgment and Handoff

Updated: 2026-06-16

Purpose: this document is the handoff note for the current PMRF line. It only evaluates what to do next for PMRF / G-PMRF. It does not cover SfM-track supervision, CIFA, or other paper directions.

## 1. Current Decision

Do **not** continue small alpha/gate sweeps on the current PMRF.

Current evidence already suggests two conclusions:

1. `post_norm=True` is a bad confound and should be excluded from the main PMRF line.
2. The second-read update `Delta2` has not yet been proven to be a stable or task-meaningful memory read.

Therefore, the next step is not to tune PMRF weaker. The next step is to test whether `Delta2` itself has value.

The active question is:

> Does the weak second-read residual help because it reads memory values, or would a parameter-matched query-only residual adapter do the same?

The answer determines whether PMRF has a valid memory-rereading story.

## 2. Stop Treating PMRF as Established

The older PMRF-v1 result on Bears was not a clear win.

Stage1 evidence:

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

- PMRF-v1 is neutral to negative on main Stage1 metrics.
- Tiny strict-threshold improvements do not prove useful memory rereading.
- Stage2 improvements are not proof, because Stage2 is a frozen-representation plus GLACE/global-head adaptation effect.

The correct status is:

> PMRF is an unproven hypothesis. The weak-residual variant may be useful, but the source of any gain is not yet identified.

## 3. First Priority: Weak-Residual Parameter-Matched FFN Control

The most important experiment is a fair control against a query-only residual adapter.

### 3.1 Compare these two updates

Weak-Residual PMRF:

```text
A2 = softmax(SecondQuery(LN(H1)) @ K^T / sqrt(d))
Delta2 = SecondOutput(A2 @ V)
H2 = H1 + alpha * g * Delta2
```

Parameter-matched FFN residual control:

```text
Delta_FFN = Adapter(LN(H1))
H2 = H1 + alpha * g * Delta_FFN
```

Both should use:

```text
post_norm = False
same alpha policy
same gate policy g
similar parameter count
same training schedule
same seeds/eval protocol
```

### 3.2 Why this is mandatory

Without this control, any PMRF gain can be explained by:

- extra residual capacity;
- extra nonlinear transformation;
- an adapter-like effect;
- weak residual regularization;
- training noise.

Only if Weak-Residual PMRF beats the FFN control can we argue that the second memory read contributes useful information.

### 3.3 Decision logic

| Result | Meaning | Next step |
|---|---|---|
| PMRF > FFN clearly | Delta2 memory value has evidence of utility | Try G-PMRF-lite |
| PMRF ~= FFN | Gain is likely residual adapter/capacity | Do not claim memory reread |
| FFN > PMRF | Current second memory read is harmful/noisy | Stop current PMRF |
| Both ~= Single | No useful signal | Stop or defer |

This experiment is the foundation. Do not skip it.

## 4. Second Priority: G-PMRF-Lite, Not Full Head(H1) Geometry

The older full G-PMRF idea used a coarse coordinate from `Head(H1)`:

```text
X1 = Head(H1)
B_geo_ij = - ||X1_i - P_j||^2 / (2 sigma_i^2)
```

This is conceptually interesting but too invasive for the immediate next step. It requires the fusion module to know about the coordinate head and introduces a teacher/student or frozen-head reliability problem.

The better first geometry-guided variant is **G-PMRF-lite**:

> Use the first-read memory attention itself to define a coarse 3D memory neighborhood, then bias the second read around that neighborhood.

### 4.1 First-read memory geometry

Given memory token anchors `P_j` and first-read attention `A1_ij`:

```text
P1_i = sum_j A1_ij * P_j
spread1_i = sum_j A1_ij * ||P_j - P1_i||^2
```

`P1_i` is the first-read memory centroid. It is not a predicted scene coordinate, but it is an inference-available geometric summary of the memory region used by the first read.

### 4.2 Geometry-biased second read

Feature score:

```text
S_feat_ij = dot(Wq2(LN(H1_i)), K_j) / sqrt(d)
```

Geometry bias around the first-read memory centroid:

```text
B_geo_ij = - ||P_j - P1_i||^2 / (2 sigma_i^2)
```

Second read:

```text
A2_i = softmax(S_feat_i + lambda * B_geo_i)
Delta2_i = SecondOutput(sum_j A2_ij * V_j)
H2_i = H1_i + alpha * g * Delta2_i
```

### 4.3 Why G-PMRF-lite is preferable now

G-PMRF-lite:

- does not require `Head(H1)`;
- does not require a frozen coordinate teacher;
- keeps the fusion/memory module self-contained;
- still makes the second read geometrically meaningful;
- turns PMRF from generic iterative attention into geometry-local residual correction;
- is easier to compare directly with Weak-Residual PMRF and FFN control.

It directly addresses the current issue:

> A2 is currently a pure feature reread. It may simply retrieve a different but not more geometrically meaningful set of tokens. G-PMRF-lite asks A2 to reread within the 3D neighborhood implied by A1.

### 4.4 First hyperparameters

Start simple:

```text
lambda = 0.5
sigma = c * sqrt(spread1_i + eps)
```

A more conservative first version can use constant sigma:

```text
sigma = 0.2 * scene_scale
```

Do not start with MLP gates, hard top-k, routing losses, or teacher guards.

## 5. Required Geometry Shift Diagnostics

Current runtime stats such as JS divergence, top-1 agreement, and reread norm are not enough.

Add geometry diagnostics:

```text
P1 = sum_j A1_ij * P_j
P2 = sum_j A2_ij * P_j
||P2 - P1||
spread(A1, P)
spread(A2, P)
Delta2 norm / H1 norm
cos(H1, H2)
```

Interpretation:

| Observation | Meaning |
|---|---|
| `||P2-P1||` large, metrics not better | second reread is likely noisy rerouting |
| `||P2-P1||` small, metrics not better | A2 adds little new information |
| `||P2-P1||` moderate, spread decreases, Acc5/Acc2 improve | G-PMRF-lite has evidence |
| spread collapses globally | geometry bias is too strong |
| spread increases while metrics degrade | A2 is diffusing memory attention |

The important question is not whether A2 is sharper. The important question is:

> Does A2 move each patch toward a more useful 3D memory neighborhood?

## 6. What Not To Do Now

Do not currently add:

- teacher guard;
- routing loss;
- hard top-k;
- multi-layer reread;
- complex patch-wise confidence gates;
- STGS + PMRF joint training;
- full `Head(H1)` G-PMRF;
- adaptive MLP lambda/sigma;
- coordinate-anchor residual head.

Reason:

> Delta2 itself has not yet been proven to have stable value. Adding control mechanisms around an unproven direction makes interpretation worse.

## 7. Recommended Execution Order

Use this order exactly:

```text
1. Implement weak_residual_ffn control.
2. Run Bears same-round comparison:
   - Single
   - Weak-Residual PMRF
   - Weak-Residual FFN
3. Add geometry shift diagnostics for A1/A2.
4. If PMRF does not beat FFN, stop current PMRF and move to G-PMRF-lite only if diagnostics motivate it.
5. If PMRF beats FFN, implement G-PMRF-lite.
6. Compare G-PMRF-lite against Weak-Residual PMRF and FFN.
7. Only if G-PMRF-lite has clear gains, consider confidence gate or other controls.
```

Do not proceed to steps 5–7 before step 2 is answered.

## 8. Final Research Framing

The route should shift from:

```text
Progressive Memory Re-reading
```

to:

```text
Identity-preserving geometry-guided memory residual correction
```

The intended contribution is not “repeat attention.” The intended contribution is:

> The first memory read identifies a coarse 3D memory neighborhood. The second read performs a weak, identity-preserving residual correction constrained to that neighborhood.

This framing is stronger because it is specific to scene-coordinate regression and compressed 3D memory.

## 9. Handoff Instruction

For future conversations or agents:

1. Do not tune alpha/gate further as the main activity.
2. Do not treat PMRF as established.
3. First implement and run the weak-residual FFN control.
4. Add geometry shift diagnostics.
5. Only then test G-PMRF-lite.
6. Do not implement full Head(H1) G-PMRF or teacher guard until G-PMRF-lite has evidence.

The minimum next code change should be either:

```text
A. weak_residual_ffn control
```

or

```text
B. A1/A2 geometry shift diagnostics
```

Nothing else should be added to PMRF before those are done.
