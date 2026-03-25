# Review Report: Bilateral Supervoxel Extraction for Scene-Agnostic Memory Compression

**Reviewer Role**: Adversarial CVPR/ICCV Area Chair

**Overall Assessment**: WEAK REJECT (score: 4/10)

The paper addresses a real problem (semantic boundary blurring in voxel pooling) and proposes a principled solution (bilateral clustering). However, the proposal suffers from insufficient empirical justification, internal inconsistencies, and a weak evaluation plan that relies heavily on qualitative visualization rather than quantitative metrics.

---

## Major Issues

### 1. Unjustified Hyperparameter Selection (Critical)

**Claim** (line 66): "The binary split (main/outlier) must be calibrated (τ) to avoid over-segmentation"

**Claim** (line 199): "τ=0.90 over-segments flat walls" (listed as Medium likelihood risk)

**Problem**: The proposal sets τ=0.90 as the default but provides ZERO empirical evidence for this choice. The authors claim DINOv2 features are "往往高度平滑" (highly smooth) in the TDD, suggesting τ=0.90 is sufficient to distinguish boundaries. But:

- What is the actual cosine similarity distribution of DINOv2 features within a 5cm voxel?
- Have the authors measured this on 7-Scenes or Indoor6?
- Why not τ=0.85 or τ=0.95?

The ablation plan (line 188) includes τ ∈ {0.70, 0.80, 0.90, 0.95}, but this is AFTER implementation. The default should be justified BEFORE claiming it's the right choice.

**Recommendation**: Add a preliminary analysis section showing DINOv2 feature similarity distributions on a pilot scene (e.g., 7-Scenes Chess). Plot histogram of cosine similarities within voxels. Show that τ=0.90 sits at a natural gap in the distribution.

---

### 2. Internal Contradiction: Normalization Timing (Critical)

**Claim** (line 67): "Global normalization must be computed **before pooling** to avoid data leakage"

**Implementation** (lines 137-142): Normalization is computed **AFTER BSE pooling** on P_bse, not P_raw.

**Problem**: These two statements directly contradict each other. The "Why Non-Trivial" section claims normalization before pooling is necessary to avoid data leakage, but the actual pipeline does the opposite.

**Analysis**: The authors likely changed their mind during TDD refinement (the user's detailed design document explicitly states normalization should be on P_bse to "avoid outlier contamination"). This is actually the CORRECT choice — normalizing after pooling prevents a single bad depth estimate from skewing μ_scene and σ_scene.

However, the proposal text was not updated to reflect this design change. Line 67 is now **factually incorrect** and should be removed or rewritten.

**Recommendation**: Delete line 67 or rewrite as: "Global normalization must be computed after pooling to avoid outlier contamination of scene statistics."

---

### 3. Weak Evaluation: No Quantitative Boundary Preservation Metrics (Major)

**Claim** (line 177): "Boundary preservation: qualitative Open3D visualization"

**Problem**: The core contribution of this paper is semantic boundary preservation, yet the evaluation plan relies entirely on QUALITATIVE visualization ("expect dense points at edges, sparse on flat walls", line 246). This is unacceptable for a top-tier venue.

**Missing Metrics**:
- **Edge density ratio**: Compute edge maps from RGB images (Canny/Sobel), measure point density within 10px of edges vs. flat regions. BSE should show higher ratio than vanilla pooling.
- **Boundary recall**: Given ground truth object masks (if available), measure what % of object boundary pixels have nearby points in P_bse.
- **Feature variance preservation**: Measure std(F_bse) in high-gradient regions vs. low-gradient regions. BSE should preserve higher variance at boundaries.

**Recommendation**: Add at least ONE quantitative boundary preservation metric. Edge density ratio is the easiest to implement (requires only Canny edge detection on RGB images).

---

### 4. Chunked Processing Artifacts Not Addressed (Major)

**Claim** (lines 146-161): Two-pass chunked processing strategy

**Problem**: The proposal describes a complex two-pass strategy:
- Pass 1: Per-chunk BSE (Steps 1-4) → append to global buffer
- Pass 2: Global BSE (Steps 2-4) on full buffer
- Pass 3: Normalization

This means BSE runs TWICE — once per chunk, then again globally. The proposal does not discuss:

1. **Chunk boundary artifacts**: Points near chunk boundaries may be split differently in Pass 1 vs. Pass 2. A point that was "main cluster" in its local chunk might become "outlier" in the global context.

2. **Feature drift**: After Pass 1, you have P_bse_chunk with pooled features F_bse_chunk. In Pass 2, you pool these AGAIN. This is double-pooling, which could over-smooth features.

3. **Computational cost**: Running BSE twice effectively doubles the cost. Why not just process all images in a single pass with larger GPU memory (e.g., A100 80GB)?

**Recommendation**: Either (a) prove that double-pooling doesn't degrade quality via ablation, or (b) simplify to single-pass processing with explicit memory budget analysis.

---

## Minor Issues

### 5. Missing Comparison to Modern Learned Pooling

**Problem**: The proposal dismisses FPS+kNN as O(N²) (line 80) but ignores modern learned pooling methods:
- PointNet++ with multi-scale grouping
- Sparse convolutions (MinkowskiEngine)
- Transformer-based pooling (Point Transformer)

These methods are widely used in 3D vision and could potentially preserve boundaries better than hand-crafted bilateral filtering.

**Counterargument**: The proposal explicitly states "no new trainable parameters" (TDD Section 1), so learned methods are out of scope. However, this constraint should be stated in the proposal itself.

**Recommendation**: Add to Section 3: "We constrain our design to non-parametric methods to avoid scene-specific overfitting and enable zero-shot transfer."

---

### 6. Binary Split May Be Insufficient

**Claim** (line 85): "binary split may be too coarse for complex boundaries (acceptable for v1)"

**Problem**: The proposal acknowledges this limitation but doesn't explore alternatives:
- Iterative k-means (k=3 or 4 sub-clusters per voxel)
- Hierarchical clustering with automatic k selection
- DBSCAN with density-based outlier detection

**Recommendation**: Add to future work section. For v1, binary split is acceptable if properly ablated.

---

### 7. Depth Quality Dependency Not Discussed

**Problem**: The unprojection step (line 98) requires depth Z, but the proposal doesn't discuss depth quality:
- What if Depth Anything V2 produces noisy depth estimates?
- How do depth errors propagate through BSE?
- Should depth confidence be used to weight points during pooling?

The proposal mentions "Depth Anything V2" in the context but never analyzes its error characteristics.

**Recommendation**: Add a robustness analysis section. At minimum, discuss expected depth error magnitude and whether it affects boundary preservation.

---

## Strengths

Despite the above issues, the proposal has several merits:

1. **Clear problem formulation**: Mathematical notation is precise and unambiguous.
2. **Practical implementation plan**: The 3-step roadmap (minimal invasion → vectorized hashing → BSE) is sensible and testable.
3. **Good risk analysis**: Table in Section 5 identifies real engineering risks with concrete mitigations.
4. **Ablation-friendly design**: τ exposed as CLI arg; --use_bse flag enables easy comparison.

---

## Counterarguments (Author's Likely Rebuttals)

### Response to Issue #1 (τ=0.90 unjustified)

**Author might argue**: "We chose τ=0.90 based on prior work with DINOv2 features in the dino_lmc_base implementation. The ablation study (line 188) will validate this choice empirically."

**Reviewer's counter**: This is circular reasoning. If τ=0.90 is based on prior work, cite it. If it's based on dino_lmc_base experiments, show the data. "We will validate it later" is not sufficient justification for a default hyperparameter.

### Response to Issue #2 (normalization timing contradiction)

**Author might argue**: "This was a deliberate design change during TDD refinement. We initially thought normalization before pooling was necessary, but realized normalization after pooling is more robust to outliers."

**Reviewer's counter**: This is a valid design evolution, but the proposal text must be updated to reflect the final design. Leaving contradictory statements in the document suggests carelessness.

### Response to Issue #3 (weak evaluation)

**Author might argue**: "Boundary preservation is inherently qualitative. The ultimate metric is downstream relocalization accuracy (line 178), which is quantitative."

**Reviewer's counter**: Downstream accuracy is a necessary but not sufficient metric. If BSE improves accuracy, we need to know WHY. Is it because of better boundary preservation, or some other factor? Quantitative boundary metrics provide this causal insight.

### Response to Issue #4 (chunked processing artifacts)

**Author might argue**: "The two-pass strategy is necessary for large scenes. We will verify that double-pooling doesn't degrade quality in the ablation study."

**Reviewer's counter**: This is a testable claim, but it should be tested BEFORE claiming the method works, not after. Add a pilot experiment on 7-Scenes Chess comparing single-pass vs. two-pass BSE.

---

## Recommendations for Acceptance

To raise this proposal from WEAK REJECT to ACCEPT, the authors must address:

1. **[Critical]** Add preliminary τ analysis: Plot DINOv2 feature cosine similarity distributions within voxels on a pilot scene. Show that τ=0.90 is empirically justified.

2. **[Critical]** Fix normalization timing contradiction: Update line 67 to match the actual implementation (normalization after pooling).

3. **[Major]** Add quantitative boundary preservation metric: Implement edge density ratio or feature variance preservation metric.

4. **[Major]** Analyze chunked processing: Either prove double-pooling is harmless via pilot experiment, or simplify to single-pass with memory budget analysis.

5. **[Minor]** Clarify design constraints: Explicitly state "no trainable parameters" constraint in Section 3.

---

## Final Verdict

**Score**: 4/10 (WEAK REJECT)

**Reasoning**: The core idea (bilateral clustering for boundary preservation) is sound and addresses a real problem. However, the proposal suffers from insufficient empirical justification (τ=0.90), internal inconsistencies (normalization timing), and weak evaluation (no quantitative boundary metrics). These are fixable issues, but they must be addressed before implementation begins.

**Recommendation**: Revise and resubmit after addressing Critical and Major issues above.
