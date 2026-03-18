# System Design: ACE-FCN-LMC

## Overview

The system is a thin extension of the existing DINOv2-LMC stack. The only structural change is swapping the backbone from DINOv2 to the ACE FCN encoder. All training orchestration, buffer management, loss computation, and result management are inherited unchanged.

## Module Dependency Graph

```
train_ace_lmc.py
    ├── options_ace_lmc.py          (CLI args)
    ├── utils_lmc.py                (shared utilities)
    ├── result_manager.py           (result management)
    └── trainer_ace_fcn.py
            ├── TrainerACEFCN       → inherits TrainerACEDINOv2 (trainer_dinov2.py)
            └── TrainerACEFCNLMC    → inherits TrainerACEDINOv2LMC (trainer_dinov2_lmc.py)
                    └── _create_regressor() → RegressorACE (ace_network_ace.py)
                                                └── ACEEncoder → Encoder (ace_network.py)
                                                └── Head (ace_network_dinov2.py)

test_ace_lmc.py
    ├── options_ace_lmc.py
    └── run_evaluation_lmc()
            └── RegressorACE (ace_network_ace.py)
            └── dsacstar (RANSAC pose estimation)
```

## Data Flow

### Training (Vanilla Mode)

```
CamLocDataset (grayscale, 480×480)
    → ACEEncoder.forward(x)          # (B,1,H,W) → (B,512,H/8,W/8)
    → TrainingBuffer.fill()          # sample features + coords
    → Head.forward(features)         # (B,512,H/8,W/8) → (B,3,H/8,W/8)
    → ReproLoss                      # reprojection loss
    → optimizer.step()
```

### Training (LMC Mode)

```
Stage 1 (S1): Memory Alignment
    PooledMemory.load()              # (N,3) 3D points
    CamLocDataset → ACEEncoder       # online or buffered
    Head.forward() → coords
    L_lmc = coord_alignment_loss(coords, memory_points)
    optimizer_s1.step()

Stage 2 (S2): Reprojection Refinement
    TrainingBuffer.fill()            # re-fill with current encoder
    Head.forward() → coords
    L_repro = ReproLoss(coords, pose_gt, K)
    optimizer_s2.step()

Repeat S1+S2 for lmc_iterations rounds
```

### Inference

```
Query image (grayscale, H×W)
    → ACEEncoder.forward()           # (1,512,H/8,W/8)
    → Head.forward()                 # (1,3,H/8,W/8) scene coords
    → dsacstar.forward()             # RANSAC PnP → T ∈ SE(3)
    → pose error vs ground truth
```

## Key Design Decisions

### 1. Thin Subclass Pattern

`TrainerACEFCN` and `TrainerACEFCNLMC` override only `_create_regressor()`:

```python
class TrainerACEFCN(TrainerACEDINOv2):
    def _create_regressor(self):
        return RegressorACE.create_from_encoder(
            encoder_path=self.options.encoder_path,
            mean=self.dataset.mean_cam_center,
            num_head_blocks=self.options.num_head_blocks,
            num_encoder_features=512,
            freeze_backbone=self.options.freeze_backbone,
        )
```

All buffer management, loss computation, LR scheduling, checkpoint saving, and result management are inherited unchanged from the DINOv2 trainers.

### 2. RGB→Grayscale Conversion in ACEEncoder

The original ACE FCN encoder expects grayscale input. The dataset pipeline provides RGB images (for compatibility with the shared DINOv2 dataset loader). ACEEncoder handles the conversion internally:

```python
# In ACEEncoder.forward():
gray = (x * self.rgb_weights).sum(dim=1, keepdim=True)  # (B,3,H,W) → (B,1,H,W)
return self.encoder(gray)
```

This avoids modifying the dataset loader and keeps the interface consistent.

### 3. Feature Dimension Adaptation

The Head network from `ace_network_dinov2.py` is parameterized by `num_encoder_features`. For ACE FCN, this is set to 512 (vs 1024 for DINOv2). The Head architecture adapts automatically.

### 4. No Patch-Size Constraint

Unlike DINOv2 (requires image dimensions to be multiples of 14), the FCN encoder supports arbitrary resolutions. The default is 480×480.

## Output Directory Structure

```
output/
└── {dataset}/
    └── {scene}/
        ├── ace_fcn_vanilla/
        │   └── {timestamp}_{config_tag}/
        │       ├── run_metadata.json
        │       ├── training_summary.json
        │       ├── best_*.pt
        │       └── eval_results/
        ├── ace_fcn_lmc_iter/          # lmc_flow=iterative
        │   └── {timestamp}_{config_tag}/
        └── ace_fcn_lmc_aceg/          # lmc_flow=ace_g
            └── {timestamp}_{config_tag}/
```

## Configuration Parameters

See `options_ace_lmc.py` for the full parameter list. Key differences from DINOv2 options:

| Parameter | ACE FCN | DINOv2 |
|---|---|---|
| `--encoder_path` | Required (FCN weights) | `--dinov2_path` |
| `--image_resolution` | Default 480, no constraint | Default 518, must be multiple of 14 |
| `--num_encoder_features` | 512 | 1024 |
| `--freeze_backbone` | Default True | Default True |
