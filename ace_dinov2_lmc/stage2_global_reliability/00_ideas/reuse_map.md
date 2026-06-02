# Reuse Map

## Inherited from Parent Idea / Current Implementation

No single completed research idea exactly matches the current ACE-FCN-LMC Stage2 global gate implementation. This idea depends on the current in-repo Stage2 implementation and recent Wayspots results.

| File | Class/Function | Can Be Used For |
|---|---|---|
| `../options_dinov2_lmc.py` | `--ace_lmc_global_feature_mode`, `--ace_lmc_global_gate_init`, `--ace_lmc_global_gate_learnable`, `--ace_lmc_global_gate_max` | Existing CLI surface for Stage2 global feature mode and scalar/bounded gate controls. |
| `../trainer_dinov2_lmc.py` | `_prepare_ace_lmc_global_feature_bank`, `_init_ace_lmc_global_gate`, `_current_ace_lmc_global_gate_tensor`, `_apply_ace_lmc_global_gate`, `_append_ace_lmc_global_to_features`, `_append_ace_lmc_global_to_feature_maps` | Training-side Stage2 global feature preparation, gating, and concat path. Extend here for consistency/adaptive gate/residual controls. |
| `../test_ace_dinov2_lmc.py` | ACE-FCN-LMC checkpoint detection and eval global gate/mode loading | Eval-side consistency for checkpointed gate, mode, and global feature settings. Extend if Stage2 adds adaptive gate/residual/dual-head state. |
| `../scripts/run_squarebench_stage2_global_gate_matrix.sh` | `variant_config`, `run_variant` | Existing Bears/SquareBench Stage2-only matrix runner. Extend for new variants and quick two-scene validation. |
| `../scripts/summarize_wayspots_ace_fcn_lmc_suite.py` | threshold parsing and summary generation | Aggregate Acc50/Acc25/Acc10/Acc5 and median pose metrics across Stage1/Stage2 experiments. |

## Reusable Components from Project Root

| Existing File | Class/Function | Can Be Used For |
|---|---|---|
| `../memory_extraction/extract_memory_ace_fcn.py` | ACE-FCN memory extraction entry | Provides Stage1 local memory files; no Stage2 reliability change should break this input contract. |
| `../ace_fcn_lmc/ace_network_ace.py` | ACE-FCN encoder/head definitions | Defines local feature dimensionality and head input behavior for Stage1/Stage2 ACE-FCN-LMC. |

## Key Reference Files

Files to consult during implementation but not reimplemented:
- `../trainer_dinov2_lmc.py` — main training flow, Stage2 optimizer/checkpoint save logic, global concat path.
- `../test_ace_dinov2_lmc.py` — test-time feature assembly and checkpoint config loading.
- `../scripts/run_squarebench_stage2_global_gate_matrix.sh` — current command template for bounded gate experiments.
- `/data/xwh/ace_dinov2_lmc/04_evaluation/wayspots_ace_fcn_lmc_suite/20260530_181745/` — primary experiment root.
- `/data/xwh/ace_dinov2_lmc/04_evaluation/wayspots_ace_fcn_lmc_suite/20260530_181745/stage2_global_gate_matrix_squarebench_gatefix_bounded01/` — SquareBench bounded gate results.
- `/data/xwh/ace_dinov2_lmc/04_evaluation/wayspots_ace_fcn_lmc_suite/20260530_181745/stage2_global_gate_matrix_bears_gatefix_bounded01/` — Bears bounded gate results.

## Unverified (filename-only, not read)

- `../train_ace_dinov2_lmc.py` — likely training entry point that persists config; consult before adding CLI options.
- `../dataset_ace_fcn_lmc.py` — likely dataset/index mapping needed if per-image validation gates are introduced.
