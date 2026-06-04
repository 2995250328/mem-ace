# Reuse Map

## Inherited from Parent Idea / Current Results

| File | Class/Function | Can Be Used For |
|---|---|---|
| `../stage2_global_reliability/01_design/proposal.md` | proposal | Negative-transfer control for Stage2 GLACE global features; use as prerequisite for reviewer-proof Wayspots/SquareBench experiments. |
| `../options_dinov2_lmc.py` | CLI options | Existing and future flags for DINO-LMC, GLACE-LMC, ACE-FCN-LMC, global gates, memory paths, evaluation settings. |
| `../trainer_dinov2_lmc.py` | `TrainerACEDINOv2LMC` | Main implementation path for LMC training flows, memory compression, Stage1/Stage2, and GLACE/ACE-FCN variants. |
| `../test_ace_dinov2_lmc.py` | evaluation entry | Unified checkpoint testing for DINO-LMC / GLACE-LMC / ACE-FCN-LMC variants. |
| `../scripts/run_squarebench_stage2_global_gate_matrix.sh` | experiment runner | Template for scene-level matrix runs, especially Stage2 global-gate and reliability ablations. |
| `../scripts/summarize_wayspots_ace_fcn_lmc_suite.py` | result aggregation | Aggregates Wayspots thresholds and medians for paper tables. |

## Reusable Components from Project Root

| Existing File | Class/Function | Can Be Used For |
|---|---|---|
| `../../trainer_dinov2.py` | `TrainerACEDINOv2` | DINOACE baseline and vanilla trainer comparison. |
| `../../ace_network_dinov2.py` | `Regressor` | DINOv2 scene coordinate baseline head. |
| `../memory_extraction/extract_memory_ace_fcn.py` | ACE-FCN memory extraction | Stage1 ACE-FCN feature-space memory extraction for GLACE/ACE-FCN-LMC experiments. |
| `../ace_fcn_lmc/ace_network_ace.py` | ACE-FCN network | Local ACE-FCN feature stack for GLACE-LMC/ACE-FCN-LMC experiments. |

## Key Reference Files

- `../../papers/Bruns 等 - 2025 - ACE-G Improving Generalization of Scene Coordinate Regression Through Query Pre-Training.pdf` — ACE-G datasets, baselines, fine-threshold discussion, query-training/uncertainty analyses.
- `../../papers/Chen 等 - 2024 - Map-Relative Pose Regression for Visual Re-Localization.pdf` — marepo APR-style datasets, mapping time/throughput, optional fine-tuning, and APR comparison protocol.

## Unverified (filename-only, not read)

- `../scripts/*indoor6*` — likely Indoor6 launchers to adapt for GLACE+LMC.
- `../scripts/*cambridge*` — likely Cambridge launchers for medium-scale outdoor validation.
- APR/marepo integration files are not present in this repo snapshot; any APR adapter should first be treated as a lightweight proof-of-generality experiment, not a main implementation dependency.
