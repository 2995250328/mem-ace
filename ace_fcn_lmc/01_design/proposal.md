# Title: ACE-FCN-LMC: Accelerated Coordinate Encoding with FCN Encoder and Geometric Latent Memory Compression

## 1. Problem Formulation

**Task**: 6DoF visual camera relocalization — given a query image $I \in \mathbb{R}^{H \times W \times 1}$ (grayscale), predict the camera pose $\mathbf{T} \in SE(3)$ relative to a known scene.

**Approach**: Scene coordinate regression. A network $f_\theta$ predicts a dense 3D scene coordinate map $\mathbf{S} \in \mathbb{R}^{(H/8) \times (W/8) \times 3}$, then DSAC* RANSAC solves the PnP problem to recover $\mathbf{T}$.

**Objective**:
$$\mathcal{L} = \mathcal{L}_\text{repro}(\mathbf{S}, \mathbf{T}^*, \mathbf{K}) + \lambda \cdot \mathcal{L}_\text{lmc}(\mathbf{S}, \mathbf{M})$$

where $\mathbf{T}^*$ is the ground-truth pose, $\mathbf{K}$ is the camera intrinsic matrix, $\mathbf{M}$ is the external 3D point-cloud memory, and $\mathcal{L}_\text{lmc}$ is the memory-guided coordinate alignment loss.

**Inputs**: Grayscale image (1-channel), camera intrinsics, optional pooled 3D memory $\mathbf{M}$

**Outputs**: Trained scene-specific head network (~4MB), per-frame 6DoF pose estimates

## 2. Literature Landscape & Motivation

### Representative Methods

1. **ACE (Brachmann et al., CVPR 2023)**: Two-stage training — pre-train scene-agnostic FCN encoder on ScanNet, then train scene-specific head via buffer-based coordinate regression. Fast (~5 min/scene), compact (~4MB), but limited by single-stage buffer training.

2. **ACE-G (Brachmann et al., CVPR 2024)**: Extends ACE with a global memory module. Iterative S1+S2 training: S1 aligns head predictions to memory-provided 3D coordinates; S2 refines via reprojection loss. Achieves state-of-the-art on 7-Scenes and Indoor6.

3. **DINOv2-LMC (this project, prior work)**: Replaces FCN encoder with DINOv2 ViT-L/14. Higher accuracy but slower inference (~10-15 FPS vs ~30 FPS) and higher memory requirements.

### Research Gap

ACE-G achieves strong accuracy but requires the DINOv2 backbone in some configurations. The FCN encoder is faster and lighter, but has not been combined with the GeoLMC iterative training framework in a clean, reusable way. The gap: **can we get ACE-G-level accuracy with ACE-vanilla-level speed by applying the LMC framework to the FCN encoder?**

### Why Non-Trivial

- The FCN encoder outputs grayscale-based 512-dim features at 8x downsampling, while DINOv2 outputs RGB-based 1024-dim features at 14x downsampling. The LMC pipeline must be adapted for these differences.
- The memory alignment loss (S1) must work with lower-dimensional features.
- The iterative training schedule (warmup steps, LR scaling) may need re-tuning for the FCN encoder.

## 3. Design Space & Selected Architecture

### Alternative Paradigms

| Paradigm | Description | Trade-offs |
|---|---|---|
| A: Full rewrite | Implement FCN-LMC from scratch | Clean but duplicates ~2000 lines of validated code |
| B: Thin subclass (selected) | Subclass DINOv2 trainers, override only `_create_regressor()` | Minimal code, inherits all validated logic |
| C: Config-based switch | Add `--encoder_type fcn/dinov2` flag to existing trainers | Simpler CLI but adds branching complexity to shared code |

**Selected: Paradigm B** — thin subclass pattern. Rationale:
- The DINOv2 LMC trainer is already validated across multiple scenes
- Only the encoder instantiation differs; all buffer management, loss computation, and iterative scheduling logic is identical
- Minimizes risk of introducing bugs into validated training logic

### Core Architecture

```
Input (grayscale, H×W×1)
    │
    ▼
ACEEncoder (FCN, frozen)
    │  512-dim features, H/8 × W/8
    ▼
RegressorACE (Head, scene-specific)
    │  3D scene coordinates, H/8 × W/8 × 3
    ▼
DSAC* RANSAC → 6DoF Pose T ∈ SE(3)
```

**LMC Two-Stage Training**:
- **Stage 1 (S1)**: Align head predictions to pooled memory $\mathbf{M}$ via coordinate matching loss. Uses online encoder forward or pre-filled buffer.
- **Stage 2 (S2)**: Refine via reprojection loss $\mathcal{L}_\text{repro}$ on full training set. Buffer-based, fast iteration.

### Key Modules

| Module | File | Role |
|---|---|---|
| ACEEncoder | `ace_network_ace.py` | Wraps FCN encoder, exposes DINOv2-compatible interface |
| RegressorACE | `ace_network_ace.py` | Head network for coordinate regression (512-dim input) |
| TrainerACEFCN | `trainer_ace_fcn.py` | Vanilla training (inherits TrainerACEDINOv2) |
| TrainerACEFCNLMC | `trainer_ace_fcn.py` | LMC training (inherits TrainerACEDINOv2LMC) |
| options_ace_lmc | `options_ace_lmc.py` | CLI args (encoder_path, no patch-size constraint) |
| train_ace_lmc | `train_ace_lmc.py` | Entry point (vanilla + LMC modes) |
| test_ace_lmc | `test_ace_lmc.py` | Evaluation script |

## 4. Evaluation & Validation Plan

### Datasets
- **7-Scenes** (indoor, small-scale): chess, fire, heads, office, pumpkin, redkitchen, stairs
- **Indoor6** (indoor, medium-scale): scene1–scene6
- **Cambridge Landmarks** (outdoor, large-scale): KingsCollege, OldHospital, ShopFacade, StMarysChurch

### Metrics
- `pct5`: % frames within 5cm/5deg (primary metric)
- `pct25_5`: % frames within 25cm/5deg
- `median_rErr`: median rotation error (degrees)
- `median_tErr`: median translation error (cm)
- `avg_time`: average inference time per frame (ms)

### Baselines
- ACE vanilla (single-stage, no memory)
- ACE-G (official, with memory)
- DINOv2-LMC (this project, prior work)

### Ablation Studies
- Vanilla vs. LMC (memory on/off)
- Number of LMC iterations (4, 8, 16, 28)
- S1 loss mode: `full_map` vs. `sample_per_image`
- S1 buffer mode: online encoder vs. pre-filled buffer
- Multi-round vanilla iterations (1, 2, 3)

## 5. Expected Failure Modes & Engineering Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| FCN 512-dim features insufficient for memory alignment | Medium | Use larger head (num_head_blocks=4+) |
| S1 LR schedule not tuned for FCN | Medium | Start with DINOv2 defaults, tune warmup_steps |
| Memory format mismatch (pooled_features dim=512 vs 1024) | Low | Memory is 3D points only; features not used in S1 loss |
| OOM with large buffer on 16GB GPU | Low | Use `--buffer_on_cpu True` |
| Grayscale→RGB conversion artifacts | Low | ACEEncoder handles RGB→grayscale internally |

## 6. Reuse Plan

See `00_ideas/reuse_map.md` for the complete mapping. Summary:

- **Reuse unchanged**: `ace_network.py` (Encoder), `ace_network_dinov2.py` (Head), `trainer_dinov2.py`, `trainer_dinov2_lmc.py`, `dataset.py`, `ace_loss.py`, `result_manager.py`, `utils_lmc.py`, `dsacstar/`
- **New (thin wrappers)**: `ace_network_ace.py`, `options_ace_lmc.py`, `trainer_ace_fcn.py`, `train_ace_lmc.py`, `test_ace_lmc.py`
- **Total new code**: ~1500 lines (vs ~6000 lines if written from scratch)
