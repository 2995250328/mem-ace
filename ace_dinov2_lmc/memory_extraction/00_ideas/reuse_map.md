# Reuse Map

## Inherited from Parent Idea (dino_lmc_base)

| File | Class/Function | Can Be Used For |
|---|---|---|
| `../trainer_dinov2_lmc.py` | `TrainerACEDINOv2LMC` | DINOv2 model loading pattern, buffer schema, feature extraction loop |
| `../options_dinov2_lmc.py` | `get_lmc_train_parser()` | argparse pattern, `--data_backend`, `--wai_repo_root` args |
| `../train_ace_dinov2_lmc.py` | main entry | Script structure, sys.path setup, device selection |
| `../test_ace_dinov2_lmc.py` | test entry | Evaluation loop reference |

## Reusable Components from Project Root

| Existing File | Class/Function | Can Be Used For |
|---|---|---|
| `../../dataset_dinov2.py` | `CamLocDatasetDINOv2` | RGB image loading, intrinsics, pose loading for 7-Scenes/Indoor6 |
| `../../ace_util.py` | `to_homogeneous` | Coordinate transform utilities |
| `../../ace_compressor.py` | `GeoLMC` | Reference for feature compression (not reused directly) |

## Key Reference Files

Files to consult during implementation (Stage 5) but not reimplemented:

- `../../dataset.py` — grayscale dataset loader (reference for pose/intrinsics loading pattern)
- `../../ace_trainer.py` — base training loop (reference for buffer fill pattern)
- `map-anything/mapanything/tasks/run_memory_extraction.py` — source to migrate (read-only)
- `map-anything/bash_scripts/ace/fps_memory.sh` — bash script to adapt

## Unverified (filename-only, not read)

- `../../dataset_wai_dinov2.py` — likely WAI-format dataset loader (may be needed for Indoor6)
- `../../ace_vis_util.py` — likely visualization utilities (may be useful for point cloud debug)
