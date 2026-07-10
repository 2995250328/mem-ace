# Source Map

Date: 2026-07-02

This map ties the cleaned paper-figure concepts to production sources. Use this map when the draw.io agent needs to verify a figure label against production code.

| Paper concept | Production source | Relevant implementation |
|---|---|---|
| Pooled scene memory | `/home/xwh/project/ace_depth/utils_lmc.py` | `load_memory_features`, pooled memory contract, validation, preflight checks |
| DINO/MapAnything memory extraction | `memory_extraction/run_memory_extraction.py` | RGB/depth feature extraction, point-feature alignment, pooling, serialization |
| ACE-FCN memory extraction | `memory_extraction/extract_memory_ace_fcn.py` | ACE-FCN feature extraction and COLMAP/sparse-depth-aligned memory export |
| SfM sparse support | `tools/project_colmap_sparse_depth.py` | Sparse point projection into training images |
| STGS sidecar construction | `tools/build_colmap_keyframe_channel.py` | track filtering, source/target patch alignment, sidecar writeout |
| STGS sidecar consumption | `trainer_dinov2_lmc.py` | sidecar metadata/load, buffer fields, cross-view track reprojection |
| 3D anchor sampling | `/home/xwh/project/ace_depth/ace_compressor.py` | `farthest_point_sampling`, `_sample_latent_coords` |
| Geo-token compression | `/home/xwh/project/ace_depth/ace_compressor.py` | `GeoLMC`, global forward path, latent `(Z, P)` output |
| Anchored token handoff | `trainer_dinov2_lmc.py` | memory compression and feature fusion calls |
| Memory fusion | `/home/xwh/project/ace_depth/ace_fusion.py` | `LMCFusionBlock`, `LMCFeatureFusion`, centered `P-c` geometry path |
| SCR coordinate heads | `/home/xwh/project/ace_depth/ace_network_dinov2.py`, `/home/xwh/project/ace_depth/ace_fcn_lmc/ace_network_ace.py` | coordinate head and regressor definitions |
| Single-view reprojection | `/home/xwh/project/ace_depth/ace_loss.py`, `trainer_dinov2_lmc.py` | reprojection and invalid-coordinate contract |
| Geometric Alignment phase | `trainer_dinov2_lmc.py` | joint training of compressor, fusion, and coordinate head |
| Cached Memory Adaptation phase | `trainer_dinov2_lmc.py` | cached/no-grad compressor output, fusion/head adaptation |

## Deliberately Excluded From Main Figure

Downstream global-refinement internals, branch-specific head variants, local/hierarchical/learned compressors, feature-layer ablations, global scale-token plumbing, fusion refinement variants, dataset-specific launch orchestration, optimizer details, logging, and checkpoint-format plumbing.
