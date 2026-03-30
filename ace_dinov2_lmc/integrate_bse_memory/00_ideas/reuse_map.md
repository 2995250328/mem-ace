# Reuse Map

## Inherited from Parent Ideas

### from: dino_lmc_base
| File | Class/Function | Can Be Used For |
|---|---|---|
| `trainer_dinov2_lmc.py` | `TrainerACEDINOv2LMC` | Main trainer — **no changes needed** downstream of memory loading |
| `train_ace_dinov2_lmc.py` | main() | Training entry point — may need minor flag addition |
| `options_dinov2_lmc.py` | `get_lmc_train_parser()` | CLI args — may need `--memory_format` flag |
| `test_ace_dinov2_lmc.py` | testing logic | No changes needed |

### from: memory_extraction
| File | Class/Function | Can Be Used For |
|---|---|---|
| `memory_extraction/run_memory_extraction.py` | `save_memory_pt()` | BSE save format (reference only, not called at runtime) |

## Reusable Components from Project Root

| Existing File | Class/Function | Can Be Used For |
|---|---|---|
| `utils_lmc.py` | `load_memory_features()` | **Primary modification target** — add BSE format branch |
| `utils_lmc.py` | `preflight_memory_features()` | Validation — should work unchanged if adapter produces correct dict |
| `ace_compressor.py` | `GeoLMC` | LMC compressor — **no changes** |
| `ace_fusion.py` | `LMCFeatureFusion` | Fusion module — **no changes** |
| `ace_loss.py` | `ReproLoss` | Loss computation — **no changes** |

## Key Reference Files
- `ace_network_dinov2.py` — DINOv2 Regressor (feature_dim=1024)
- `dataset_dinov2.py` — Dataset loader (generates mean_cam_center used in scene validation)

## Format Mapping (Critical for Implementation)

### BSE Memory → Pooled Memory Field Mapping

| BSE Field | Pooled Field | Notes |
|---|---|---|
| `points` [N,3] | `pooled_points` [N,3] | Direct rename |
| `features` [N,C] fp16 | `pooled_features` [N,C] | Rename + cast to float32 |
| `colors` [N,3] | `pooled_colors` [N,3] | Direct rename |
| `mu` [3] | `scene_center` [3] | Direct rename |
| — | `all_poses` | Construct from `view_camera_centers` + `view_camera_rotations` |
| — | `all_intrinsics` | Map from `view_camera_intrinsics` |
| `all_scale_tokens` [M,D] | `all_scale_tokens` [M,D] | Direct rename (now saved by BSE extraction) |
| — | `layers_idx` | **Missing** — default to [] → num_layers=4 |

### BSE Fields NOT Used
- `ray_dirs`, `ray_dirs_mean`, `ray_dirs_dominant`, `ray_dirs_first` — ray direction data
- `plucker_rays` — Plücker encoding
- `cluster_sizes` — BSE cluster statistics
- `sigma` — scene std (not consumed by trainer)
- `view_plucker_main_rays` — Plücker rays for views

### Key Numerical Compatibility
- BSE `features` is fp16 → must cast to float32 for `load_memory_features` validation
- BSE feature dim is 1024 (DINOv2 ViT-L/14) → matches `backbone_feature_dim`
- No `layers_idx` → trainer defaults to `num_layers=4`, `feature_dim=1024/4=256`
  - This means GeoLMC will treat features as 4-layer with 256-dim per layer
  - This is the same behavior as when layers_idx is absent in the old format
