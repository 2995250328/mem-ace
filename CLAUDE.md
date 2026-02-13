# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an extended implementation of ACE (Accelerated Coordinate Encoding) for visual camera relocalization. The codebase contains four variants:

1. **Basic ACE**: Original implementation from the CVPR 2023 paper
2. **ACE Depth**: Extended version integrating Depth Anything V2 for depth-aware coordinate regression
3. **ACE Full**: Complete version combining depth estimation, SuperPoint features, and camera intrinsic fusion
4. **ACE DINOv2**: Uses DINOv2 ViT-L/14 as backbone instead of FCN encoder (NEW)

The system learns to predict dense 3D scene coordinates from RGB images, then uses RANSAC-based pose estimation to determine 6DoF camera poses.

## Environment Setup

### Initial Setup

```bash
# Create and activate conda environment
conda env create -f environment_depth.yaml
conda activate ace

# Build DSAC* C++ bindings (required for RANSAC pose estimation)
cd dsacstar
python setup.py install
cd ..
```

Note: The project uses PyTorch 2.0.0 with CUDA 11.8 and Python 3.8.

### GPU Selection

All training and testing scripts support GPU selection via `--device` parameter:
```bash
./train_ace_depth.py <scene_path> <output_map> --device cuda:3
```

The scripts automatically set `CUDA_VISIBLE_DEVICES` to isolate GPU usage.

## Training Commands

### Basic ACE Training
```bash
./train_ace.py datasets/7scenes_chess output/7scenes_chess.pt
```

### ACE with Depth Training
```bash
./train_ace_depth.py datasets/7scenes_chess output/7scenes_chess_depth.pt --device cuda:3
```

Parameters:
- `--num_head_blocks`: Depth of regression head (default: 8 for depth variant)
- `--training_buffer_size`: Size of training buffer (default: 8M)
- `--samples_per_image`: Features sampled per image (default: 1024)
- `--epochs`: Training epochs (default: 16)

### ACE Full Training
```bash
./train_ace_full.py datasets/7scenes_chess output/7scenes_chess_full.pt --device cuda:3
```

This variant includes depth estimation, SuperPoint feature extraction, and camera intrinsic fusion.

### ACE DINOv2 Training
```bash
./train_ace_dinov2.py datasets/7scenes_chess output/7scenes_chess_dinov2.pt --device cuda:3
```

Parameters:
- `--dinov2_path`: Path to DINOv2 pretrained weights (default: `/data/xwh/checkpoints/dinov2_vitl14_pretrain.pth`)
- `--freeze_backbone`: Freeze DINOv2 backbone (default: True)
- `--image_resolution`: Image height, must be multiple of 14 (default: 518)
- `--num_head_blocks`: Depth of regression head (default: 1)
- `--batch_size`: Batch size (default: 512, reduce if OOM)

**Important**: Image resolution must be a multiple of 14 (DINOv2 patch size). Recommended: 518 (37×14), 532 (38×14), or 504 (36×14).

## Testing Commands

### Basic ACE Testing
```bash
./test_ace.py datasets/7scenes_chess output/7scenes_chess.pt
```

### ACE with Depth Testing
```bash
./test_ace_depth.py datasets/7scenes_chess output/7scenes_chess_depth.pt --device cuda:3
```

### ACE Full Testing
```bash
./test_ace_full.py datasets/7scenes_chess output/7scenes_chess_full.pt --device cuda:3
```

### ACE DINOv2 Testing
```bash
./test_ace_dinov2.py datasets/7scenes_chess output/7scenes_chess_dinov2.pt --device cuda:3
```

Parameters:
- `--dinov2_path`: Path to DINOv2 pretrained weights
- `--image_resolution`: Must match training resolution (default: 518)
- `--hypotheses`: Number of RANSAC iterations (default: 64)
- `--threshold`: Inlier threshold in pixels (default: 10)
- `--session`: Custom suffix for output pose files

Parameters:
- `--hypotheses`: Number of RANSAC iterations (default: 64)
- `--threshold`: Inlier threshold in pixels
- `--session`: Custom suffix for output pose files

Output: Creates `poses_<map_name>_<session>.txt` with per-frame pose estimates and errors.

## Batch Training Scripts

Located in `scripts/` directory:
- `train_7scenes.sh`: Train on all 7-Scenes scenes
- `train_12scenes.sh`: Train on all 12-Scenes scenes
- `train_cambridge.sh`: Train on Cambridge Landmarks
- `train_wayspots.sh`: Train on Niantic Wayspots
- `train_cambridge_ensemble.sh`: Train ensemble (ACE Poker) variant

## Architecture Overview

### Core Components

1. **Encoder**: Pre-trained scene-agnostic feature extractor. Two options:
   - `ace_encoder_pretrained.pt`: FCN encoder trained on 100 ScanNet scenes (512-dim features, 8x downsampling)
   - DINOv2 ViT-L/14: Transformer-based encoder (1024-dim features, 14x downsampling)

2. **Head Network**: Scene-specific coordinate regression network (~4MB). Four variants:
   - `ace_network.py`: Basic coordinate regression head
   - `ace_network_depth.py`: Adds Depth Anything V2 integration and relative depth loss
   - `ace_network_full.py`: Adds SuperPoint features and camera intrinsic fusion
   - `ace_network_dinov2.py`: Adapted for DINOv2 encoder (1024-dim input, 14x downsampling)

3. **Trainer Modules**:
   - `ace_trainer.py`: Basic training loop
   - `ace_trainer_depth.py`: Training with depth supervision
   - `ace_trainer_full.py`: Full training with all modalities
   - `trainer_dinov2.py`: Training with DINOv2 encoder

4. **Dataset Modules**:
   - `dataset.py`: Standard dataset loader (grayscale input, 8x downsampling)
   - `dataset_dinov2.py`: DINOv2-adapted dataset (RGB input, 14x downsampling, ensures image dimensions are multiples of 14)

5. **DSAC* Module** (`dsacstar/`): C++ implementation of differentiable RANSAC for pose estimation from 2D-3D correspondences.

### Data Flow

1. **Training Phase**:
   - Load pre-trained encoder weights
   - Generate training buffer by sampling features from training images
   - Train scene-specific head network on coordinate regression task
   - For depth/full variants: integrate depth maps and/or SuperPoint features

2. **Testing Phase**:
   - Load encoder + trained head network
   - For each query image: predict dense 3D scene coordinates
   - Run DSAC* RANSAC to estimate 6DoF pose from 2D-3D correspondences
   - Compare with ground truth and compute metrics

### Dataset Structure

Datasets follow DSAC* format:
```
scene_name/
├── train/
│   ├── rgb/           # Training images
│   ├── calibration/   # Camera intrinsics
│   └── poses/         # Ground truth poses
└── test/              # Same structure for test split
```

### Key Differences Between Variants

| Feature | Basic ACE | ACE Depth | ACE Full | ACE DINOv2 |
|---------|-----------|-----------|----------|------------|
| Encoder | FCN | FCN | FCN | DINOv2 ViT-L/14 |
| Input | Grayscale | Grayscale | Grayscale | RGB |
| Feature Dim | 512 | 512 | 512 | 1024 |
| Downsampling | 8x | 8x | 8x | 14x |
| Depth Branch | ✗ | ✓ | ✓ | ✗ |
| SuperPoint | ✗ | ✗ | ✓ | ✗ |
| Intrinsic Fusion | ✗ | ✗ | ✓ | ✗ |

**Variant Descriptions:**
- **Basic ACE**: RGB-only coordinate regression with FCN encoder
- **ACE Depth**: Adds depth prediction branch with relative depth loss (scale-shift invariant)
- **ACE Full**: Combines RGB, depth, SuperPoint keypoint features, and camera intrinsic information
- **ACE DINOv2**: Uses powerful Transformer-based DINOv2 encoder for potentially better generalization

## Ensemble Evaluation (ACE Poker)

For large scenes, train multiple head networks on spatial clusters:

```bash
# Train 4 cluster heads
./train_ace.py datasets/Cambridge_GreatCourt output/0_4.pt --num_clusters 4 --cluster_idx 0
./train_ace.py datasets/Cambridge_GreatCourt output/1_4.pt --num_clusters 4 --cluster_idx 1
./train_ace.py datasets/Cambridge_GreatCourt output/2_4.pt --num_clusters 4 --cluster_idx 2
./train_ace.py datasets/Cambridge_GreatCourt output/3_4.pt --num_clusters 4 --cluster_idx 3

# Test each cluster
./test_ace.py datasets/Cambridge_GreatCourt output/0_4.pt --session 0_4
./test_ace.py datasets/Cambridge_GreatCourt output/1_4.pt --session 1_4
./test_ace.py datasets/Cambridge_GreatCourt output/2_4.pt --session 2_4
./test_ace.py datasets/Cambridge_GreatCourt output/3_4.pt --session 3_4

# Merge results (select best inlier count per frame)
./merge_ensemble_results.py output/ output/merged_poses_4.txt --poses_suffix "_4.txt"

# Evaluate merged results
./eval_poses.py datasets/Cambridge_GreatCourt output/merged_poses_4.txt
```

## Visualization

Training and testing scripts support visualization via:
```bash
./train_ace_depth.py <scene> <output> --render_visualization True --render_target_path renderings/
```

This generates per-frame images showing the training/testing process (significantly slows down execution).

## DINOv2 Variant - Special Considerations

### Requirements

1. **DINOv2 Weights**: Must be available at `/data/xwh/checkpoints/dinov2_vitl14_pretrain.pth` (or specify custom path with `--dinov2_path`)
2. **Image Resolution**: Must be a multiple of 14 (patch size). Recommended values:
   - 518 (37×14) - default, closest to original 480
   - 532 (38×14)
   - 504 (36×14)
   - 560 (40×14)
3. **Memory**: DINOv2 ViT-L/14 is larger than FCN encoder:
   - Frozen backbone: ~6GB VRAM (batch_size=512)
   - Fine-tuning: ~12GB VRAM (batch_size=256)

### Key Differences from Original ACE

1. **Input Format**: RGB (3 channels) instead of grayscale (1 channel)
2. **Normalization**: ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
3. **Output Resolution**: 14x downsampling instead of 8x
4. **Feature Dimension**: 1024 instead of 512
5. **Camera Intrinsics**: Automatically adjusted for 14x downsampling

### Training Strategy

**Recommended (Default):**
- Freeze DINOv2 backbone (`--freeze_backbone True`)
- Only train the scene-specific head
- Faster training, less memory, prevents overfitting

**Alternative:**
- Fine-tune entire network (`--freeze_backbone False`)
- May achieve better performance on specific scenes
- Requires more memory and training time

### Performance Characteristics

**Advantages:**
- Stronger feature representations from large-scale pre-training
- Better generalization to new scenes
- More robust to lighting/appearance changes

**Trade-offs:**
- Slower inference (~10-15 FPS vs ~30 FPS)
- Higher memory requirements
- Requires RGB input (no grayscale)

### Troubleshooting

**Out of Memory:**
```bash
# Reduce batch size
--batch_size 256  # or 128

# Ensure backbone is frozen
--freeze_backbone True

# Use smaller image resolution
--image_resolution 504
```

**Image Size Error:**
```
AssertionError: Input size must be multiple of patch_size (14)
```
Solution: Use `--image_resolution 518` or another multiple of 14.

**DINOv2 Loading Issues:**
- Verify weights path is correct
- Check that torch.hub can access facebookresearch/dinov2
- Ensure weights file is complete (not corrupted)

For detailed DINOv2 usage instructions, see `DINOV2_USAGE.md`.

## Important Notes

- The pre-trained encoder (`ace_encoder_pretrained.pt`) must be present in the repository root
- SuperPoint weights (`superpoint_v1.pth`) required for ACE Full variant
- DSAC* C++ bindings must be compiled before running any pose estimation
- Default training time: ~5 minutes per scene on a single GPU
- Output head networks are stored as half-precision (FP16) for ~4MB file size
