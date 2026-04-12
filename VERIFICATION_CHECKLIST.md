# Migration Verification Checklist

## ✅ Files Created/Modified

### Core Modules (3 files)
- [x] `ace_compressor.py` - GeoLMC with 4 modes (global, local, hierarchical, learned)
- [x] `ace_fusion.py` - LMCFeatureFusion with cross-attention
- [x] `ace_head_lmc.py` - ACEHead_Homogeneous_Mean with optional scale token

### Training/Testing (3 files)
- [x] `trainer_dinov2_lmc.py` - Two-stage iterative trainer (extends TrainerACEDINOv2)
- [x] `train_ace_dinov2_lmc.py` - CLI with LMC parameters
- [x] `test_ace_dinov2_lmc.py` - Auto-detects LMC vs vanilla checkpoint

## ✅ Import Tests Passed

All modules import successfully and forward passes work correctly.

## ✅ Degradation Switch Verified

When `--use_lmc False` (or `--memory_path` not provided):
- TrainerACEDINOv2LMC delegates to parent TrainerACEDINOv2
- Behavior identical to vanilla DINO ACE

## 📋 Dataset Verified

- Path: `/mnt/storage/xwh/7Scenes/pgt_7scenes_chess/`
- Structure: train/ and test/ with rgb/, poses/, calibration/
- Ready for training

## 🚀 Usage Examples

### Vanilla DINO ACE (no LMC)
```bash
source ~/miniforge3/bin/activate mapanything
cd /home/xwh/project/ace_depth

./train_ace_dinov2_lmc.py \
    /mnt/storage/xwh/7Scenes/pgt_7scenes_chess \
    output/chess_vanilla.pt \
    --device cuda:0 \
    --use_lmc False
```

### With LMC (requires pre-saved memory)
```bash
./train_ace_dinov2_lmc.py \
    /mnt/storage/xwh/7Scenes/pgt_7scenes_chess \
    output/chess_lmc.pt \
    --device cuda:0 \
    --use_lmc True \
    --memory_path /path/to/memory.pt \
    --lmc_mode global \
    --num_latent_tokens 64
```

### Testing
```bash
./test_ace_dinov2_lmc.py \
    /mnt/storage/xwh/7Scenes/pgt_7scenes_chess \
    output/chess_lmc.pt \
    --device cuda:0
```

## ✅ Migration Complete

All core functionality migrated successfully.
