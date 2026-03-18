# Reuse Map

> All files listed below are already reused in the implementation and must NOT be reimplemented.

| Existing File | Relevant Class/Function | Used For |
|---|---|---|
| `../ace_network.py` | `Encoder` | ACE FCN backbone (wrapped by ACEEncoder) |
| `../ace_network_dinov2.py` | `Head` | Coordinate regression head (reused directly) |
| `../trainer_dinov2.py` | `TrainerACEDINOv2` | Base class for vanilla training logic |
| `../trainer_dinov2_lmc.py` | `TrainerACEDINOv2LMC` | Base class for LMC two-stage training logic |
| `../train_ace_dinov2_lmc.py` | `run_vanilla_iterative_baseline`, `run_post_train_eval` | Training entry-point template |
| `../options_dinov2_lmc.py` | `get_lmc_train_parser` | Argument parser base (extended by options_ace_lmc.py) |
| `../dataset.py` | `CamLocDataset` | Standard ACE dataset loader (grayscale) |
| `../dataset_wai_dinov2.py` | `WAIDINOv2Dataset` | WAI data backend |
| `../ace_loss.py` | `ReproLoss` | Reprojection loss function |
| `../ace_util.py` | `get_pixel_grid`, `to_homogeneous` | Geometry utility functions |
| `../result_manager.py` | `ResultManager` | Hierarchical result management |
| `../utils_lmc.py` | `build_lmc_run_folder_config_tag`, `setup_cuda_environment` | Shared utility functions |
| `../eval_poses.py` | `eval_poses` | Pose evaluation metric computation |
| `../dsacstar/` | C++ RANSAC bindings | DSAC* pose estimation |

## New Files Created for ace_fcn_lmc

| New File | Purpose |
|---|---|
| `../ace_network_ace.py` | ACEEncoder wrapper + RegressorACE (adapted for LMC interface) |
| `../options_ace_lmc.py` | ACE FCN-specific args (encoder_path, no patch-size constraint, etc.) |
| `../trainer_ace_fcn.py` | TrainerACEFCN / TrainerACEFCNLMC (only override _create_regressor) |
| `../train_ace_lmc.py` | Training entry point (mirrors train_ace_dinov2_lmc.py structure) |
| `../test_ace_lmc.py` | Evaluation script (mirrors test_ace_dinov2_lmc.py structure) |
