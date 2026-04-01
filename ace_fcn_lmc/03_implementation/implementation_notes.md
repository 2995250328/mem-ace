# Implementation Notes: ACE-FCN-LMC

## Status: Complete (validated across 7-Scenes and Indoor6)

## Implemented Files

### `ace_network_ace.py`
- `ACEEncoder`: wraps `ace_network.Encoder`, handles RGB→grayscale internally, exposes `feature_dim=512`, `patch_size=8`
- `RegressorACE`: combines ACEEncoder + Head (from `ace_network_dinov2.py`), provides `create_from_encoder()` factory method
- Key: `register_buffer("rgb_weights", ...)` for efficient grayscale conversion without modifying dataset loader

### `options_ace_lmc.py`
- Extends DINOv2 LMC option parser with ACE FCN-specific args
- Adds `--encoder_path` (required), removes DINOv2-specific `--dinov2_path`
- Removes patch-size constraint on `--image_resolution` (default 480)
- Adds `--data_backend` (ace/wai) for WAI dataset support

### `trainer_ace_fcn.py`
- `TrainerACEFCN(TrainerACEDINOv2)`: overrides only `_create_regressor()`
- `TrainerACEFCNLMC(TrainerACEDINOv2LMC)`: overrides `_create_regressor()` and `_evaluate_checkpoint()`
- `_evaluate_checkpoint()` uses `test_ace_lmc.run_evaluation_lmc` instead of DINOv2 eval

### `train_ace_lmc.py`
- Entry point mirroring `train_ace_dinov2_lmc.py`
- Supports vanilla mode (`--use_lmc False`) and LMC mode (`--use_lmc True`)
- Integrates `ResultManager` for hierarchical output organization
- Post-train eval via `run_post_train_eval()` with GPU memory cleanup

### `test_ace_lmc.py`
- Evaluation script for trained ACE FCN models
- `run_evaluation_lmc()`: loads RegressorACE, runs DSAC* RANSAC, computes metrics
- Outputs: `median_rErr`, `median_tErr`, `pct5`, `pct25_5`, `avg_time`

## Validated Configurations

### 7-Scenes Chess (vanilla)
```bash
python train_ace_lmc.py /mnt/storage/xwh/7Scenes/pgt_7scenes_chess output/chess.pt \
    --encoder_path ace_encoder_pretrained.pt --device cuda:0 \
    --training_buffer_size 2560000 --epochs 24 --batch_size 5120
```

### Indoor6 Scene3 (LMC, surpasses ACE-G)
```bash
./train_ace_lmc.py /mnt/storage/xwh/indoor6_ace/scene3 scene3_lmc.pt \
    --encoder_path ace_encoder_pretrained.pt --device cuda:1 \
    --use_lmc True \
    --memory_path /path/to/Indoor6_scene3_train_pooled_GT.pt \
    --lmc_iterations 28 --num_latent_tokens 64 --num_attn_layers 2 \
    --epochs 24 --batch_size 5120 --training_buffer_size 2560000
```

## Known Issues / Notes

- `--use_half True` recommended for production runs (saves ~50% checkpoint size)
- `--buffer_on_cpu True` required for 12GB GPUs with large scenes
- WAI data backend (`--data_backend wai`) requires `map-anything` repo in parent directory
- Memory preflight check (`--lmc_memory_preflight True`) validates pooled memory format before training
