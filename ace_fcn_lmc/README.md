# ACE-FCN-LMC

ACE (Accelerated Coordinate Encoding) with the original FCN encoder and GeoLMC (Geometric Latent Memory Compression) two-stage iterative training. Surpasses the ACE-G baseline on Indoor6 while maintaining ACE's original inference speed (~30 FPS).

## Overview

Two training modes:

- **Vanilla** (`--use_lmc False`): single-stage buffer training, fast baseline
- **LMC** (`--use_lmc True`): two-stage iterative training (S1 memory alignment + S2 reprojection refinement), requires a pre-built pooled memory file

## File Structure

All implementation files live in this directory:

| File | Role |
|---|---|
| `ace_network_ace.py` | `ACEEncoder` wrapper + `RegressorACE` (FCN → LMC interface) |
| `options_ace_lmc.py` | Full CLI argument parser |
| `trainer_ace_fcn.py` | `TrainerACEFCN` / `TrainerACEFCNLMC` (thin subclasses) |
| `train_ace_lmc.py` | Training entry point |
| `test_ace_lmc.py` | Evaluation script |

Root-level stubs (`../train_ace_lmc.py`, `../test_ace_lmc.py`) forward to these files for backward compatibility.

## Quick Reference

| Scenario | Key flags | Approx. time |
|---|---|---|
| Fast experiment | `--epochs 8 --training_buffer_size 1280000` | ~5 min |
| Standard vanilla | `--epochs 16 --training_buffer_size 10000000` | ~15 min |
| Iterative vanilla | `--vanilla_iterations 3 --eval_each_iteration True` | ~45 min |
| LMC quick | `--use_lmc True --lmc_iterations 4` | ~20 min |
| LMC full (ACE-G style) | `--use_lmc True --lmc_iterations 28` | ~60 min |
| Low VRAM | `--buffer_on_cpu True --batch_size 2560` | needs CPU RAM |

## Training Commands

All commands run from the **project root** (`ace_depth/`).

### Example 1 — Vanilla quick (fast validation)

```bash
python train_ace_lmc.py \
    /data/xwh/7Scenes/pgt_7scenes_chess \
    chess_quick.pt \
    --encoder_path ace_encoder_pretrained.pt \
    --device cuda:0 \
    --image_resolution 480 \
    --training_buffer_size 1280000 \
    --buffer_batch_size 5 \
    --samples_per_image 256 \
    --epochs 8 \
    --batch_size 5120 \
    --eval_after_train True
```

### Example 2 — Vanilla standard (production)

```bash
python train_ace_lmc.py \
    /data/xwh/7Scenes/pgt_7scenes_chess \
    chess_vanilla.pt \
    --encoder_path ace_encoder_pretrained.pt \
    --device cuda:0 \
    --run_name chess_vanilla \
    --image_resolution 480 \
    --training_buffer_size 10000000 \
    --buffer_batch_size 1 \
    --samples_per_image 512 \
    --epochs 16 \
    --batch_size 5120 \
    --eval_after_train True \
    --eval_session vanilla \
    --use_half True
```

> `--apply_baseline_contract True` (default) auto-applies vanilla defaults for `num_head_blocks`, `learning_rate_max`, `training_buffer_size`, `epochs`.

### Example 3 — Vanilla iterative (multi-round)

```bash
python train_ace_lmc.py \
    /data/xwh/7Scenes/pgt_7scenes_chess \
    chess_iter.pt \
    --encoder_path ace_encoder_pretrained.pt \
    --device cuda:0 \
    --run_name chess_iter3 \
    --vanilla_iterations 3 \
    --apply_baseline_contract False \
    --training_buffer_size 2560000 \
    --buffer_batch_size 1 \
    --samples_per_image 512 \
    --epochs 24 \
    --batch_size 5120 \
    --learning_rate_max 0.0001 \
    --eval_each_iteration True \
    --eval_after_train True \
    --keep_best_only True \
    --best_metric pct5 \
    --use_half True
```

### Example 4 — LMC quick (fast LMC experiment)

```bash
python train_ace_lmc.py \
    /data/xwh/7Scenes/pgt_7scenes_chess \
    chess_lmc_quick.pt \
    --encoder_path ace_encoder_pretrained.pt \
    --device cuda:0 \
    --run_name chess_lmc_quick \
    --use_lmc True \
    --memory_path /path/to/7Scenes_chess_train_pooled_GT.pt \
    --lmc_mode global \
    --lmc_flow iterative \
    --lmc_profile legacy \
    --num_latent_tokens 128 \
    --num_attn_layers 4 \
    --lmc_iterations 4 \
    --lmc_warmup_steps 1000 \
    --lmc_train_steps 500 \
    --s1_use_buffer False \
    --s1_loss_mode full_map \
    --s1_batch_size 16 \
    --s1_learning_rate_max 1e-4 \
    --training_buffer_size 2560000 \
    --buffer_batch_size 1 \
    --samples_per_image 768 \
    --epochs 16 \
    --batch_size 5120 \
    --s2_learning_rate_max 1e-3 \
    --image_resolution 480 \
    --eval_after_train True \
    --eval_session lmc_quick
```

### Example 5 — LMC full (ACE-G style, production)

```bash
python train_ace_lmc.py \
    /data/xwh/indoor6_ace/scene3 \
    scene3_lmc.pt \
    --encoder_path ace_encoder_pretrained.pt \
    --device cuda:0 \
    --run_name scene3_lmc_full \
    --use_lmc True \
    --memory_path /path/to/Indoor6_scene3_train_pooled_GT.pt \
    --lmc_mode global \
    --lmc_flow ace_g \
    --lmc_profile legacy \
    --lmc_scene_center_max_distance 4.0 \
    --num_latent_tokens 64 \
    --num_attn_layers 2 \
    --lmc_iterations 28 \
    --lmc_warmup_steps 2000 \
    --lmc_train_steps 600 \
    --s1_use_buffer True \
    --s1_loss_mode sample_per_image \
    --s1_buffer_refill_mode full \
    --s1_batch_size 16 \
    --s1_learning_rate_max 1e-4 \
    --s1_lr_scale_later 1.0 \
    --training_buffer_size 2560000 \
    --buffer_size_final 7680000 \
    --buffer_batch_size 1 \
    --buffer_on_cpu True \
    --samples_per_image 384 \
    --epochs 24 \
    --batch_size 5120 \
    --s2_learning_rate_max 1e-3 \
    --s2_repro_rewind_first_ratio 0.0 \
    --s2_repro_rewind_later_ratio 0.0 \
    --s2_lr_boost_first 1.0 \
    --s2_lr_boost_later 1.0 \
    --image_resolution 480 \
    --freeze_backbone True \
    --use_half True \
    --lmc_memory_preflight True \
    --lmc_memory_preflight_strict True \
    --lmc_lr_scheduler_type cosine \
    --eval_each_iteration True \
    --eval_after_train True \
    --keep_best_only True \
    --best_metric pct5
```

### Example 6 — Standalone evaluation

```bash
python test_ace_lmc.py \
    /data/xwh/7Scenes/pgt_7scenes_chess \
    output/7Scenes/pgt_7scenes_chess/ace_fcn_lmc_aceg/.../best_K64_it28_scene3_lmc.pt \
    --encoder_path ace_encoder_pretrained.pt \
    --device cuda:0 \
    --image_resolution 480 \
    --session lmc_eval \
    --hypotheses 64 \
    --threshold 10
```

## Parameter Reference

### Core (required)

| Parameter | Description |
|---|---|
| `scene` | Path to scene directory (must contain `train/`) |
| `output_map` | Output filename suffix (e.g. `chess.pt`); actual path is under `output/` |
| `--encoder_path` | Path to `ace_encoder_pretrained.pt` |
| `--device` | CUDA device, e.g. `cuda:0` |

### Buffer

| Parameter | Default | Description |
|---|---|---|
| `--training_buffer_size` | 2560000 | Number of feature samples in buffer |
| `--buffer_size_final` | `3×training_buffer_size` | Final buffer size for LMC last iteration |
| `--s1_last_iter_use_final_buffer` | False | Whether last S1 uses `buffer_size_final` (True) or keeps `training_buffer_size` (False) |
| `--buffer_batch_size` | 10 | Images per forward pass when filling buffer |
| `--samples_per_image` | 512 | Features sampled per image |
| `--buffer_on_cpu` | False | Store buffer on CPU (saves GPU VRAM) |
| `--buffer_sampling_replacement` | True | Allow duplicate samples |

VRAM recommendations:

| GPU | `training_buffer_size` | `buffer_batch_size` | `buffer_on_cpu` |
|---|---|---|---|
| 24 GB | 2560000 | 10 | False |
| 16 GB | 1280000 | 5 | False |
| 12 GB | 640000 | 3 | True |

### Training

| Parameter | Default | Description |
|---|---|---|
| `--epochs` | 24 | S2 training epochs per iteration |
| `--batch_size` | 5120 | S2 batch size |
| `--learning_rate_max` | 0.0001 | Vanilla max LR |
| `--s1_learning_rate_max` | 0.0001 | LMC Stage 1 max LR |
| `--s2_learning_rate_max` | 0.001 | LMC Stage 2 max LR |
| `--use_half` | False | FP16 mixed precision (saves ~50% checkpoint size) |
| `--num_head_blocks` | 4 | Depth of regression head |
| `--freeze_backbone` | True | Freeze FCN encoder |
| `--image_resolution` | 480 | Input image height (no patch-size constraint) |

### Vanilla iterative

| Parameter | Default | Description |
|---|---|---|
| `--vanilla_iterations` | 1 | Number of iterative rounds |
| `--apply_baseline_contract` | True | Auto-apply vanilla defaults (LR, buffer size, epochs) |
| `--reset_optimizer_each_iter` | False | Reset optimizer state between rounds |

### LMC

| Parameter | Default | Description |
|---|---|---|
| `--use_lmc` | False | Enable LMC mode |
| `--memory_path` | — | Path to POOLED `.pt` memory file (required when `use_lmc=True`) |
| `--lmc_mode` | `global` | `global` or `local` |
| `--lmc_flow` | `iterative` | `iterative` (classic S1+S2) or `ace_g` (ACE-G style) |
| `--lmc_profile` | `legacy` | `legacy` or `mapany_flow_v1` |
| `--lmc_iterations` | 4 | Number of S1+S2 rounds |
| `--num_latent_tokens` | 256 | Latent token count (K) |
| `--num_attn_layers` | 2 | Attention layers in compressor |
| `--lmc_lr_scheduler_type` | `onecycle` | `onecycle` or `cosine` |

### Stage 1 (S1)

| Parameter | Default | Description |
|---|---|---|
| `--lmc_warmup_steps` | 1000 | S1 steps for first iteration |
| `--lmc_train_steps` | 500 | S1 steps for subsequent iterations |
| `--s1_batch_size` | 16 | S1 batch size |
| `--s1_use_buffer` | True | Pre-fill raw buffer; `False` = online encoder forward |
| `--s1_loss_mode` | `sample_per_image` | `full_map` / `sample_per_image` / `sample_pooled` |
| `--s1_buffer_refill_mode` | `full` | `full` or `partial` (only when `s1_use_buffer=True`) |
| `--s1_lr_scale_later` | 0.2 | LR scale factor for iterations > 1 |
| `--s1_early_stop` | False | Enable early stopping in S1 |

### Stage 2 (S2)

| Parameter | Default | Description |
|---|---|---|
| `--s2_repro_rewind_first_ratio` | 0.20 | Rewind ratio for first iteration |
| `--s2_repro_rewind_later_ratio` | 0.08 | Rewind ratio for later iterations |
| `--s2_lr_boost_first` | 1.2 | LR boost multiplier for first iteration |
| `--s2_lr_boost_later` | 1.0 | LR boost multiplier for later iterations |
| `--s2_lr_warmup_steps` | 100 | S2 LR warmup steps |

### ACE-G specific (`--lmc_flow ace_g`)

| Parameter | Default | Description |
|---|---|---|
| `--ace_g_fusion_in_s2` | False | Train fusion in S2 (R2 path) |
| `--ace_g_fusion_lr_ratio` | 0.01 | Fusion LR ratio relative to head LR |
| `--ace_g_cross_iter_eval` | False | Evaluate after each S1 within an iteration |

### Memory validation

| Parameter | Default | Description |
|---|---|---|
| `--lmc_memory_preflight` | True | Validate memory file before training |
| `--lmc_memory_preflight_strict` | False | Strict validation (abort on any mismatch) |
| `--lmc_scene_center_max_distance` | 10.0 | Max allowed distance (m) between memory and scene centers |
| `--lmc_strict_scene_check` | False | Strict scene name check |
| `--lmc_strict_center_check` | False | Strict center distance check |

### Evaluation & output

| Parameter | Default | Description |
|---|---|---|
| `--eval_after_train` | True | Run evaluation after training completes |
| `--eval_each_iteration` | False | Evaluate after each LMC/vanilla iteration |
| `--keep_best_only` | False | Delete non-best iteration checkpoints |
| `--best_metric` | `pct5` | Metric for best checkpoint: `pct5`, `pct25_5`, `pct10_5`, `pct2`, `median` |
| `--eval_session` | `post_train` | Session name suffix for output files |
| `--post_train_eval_device` | `cuda:0` | Device for post-training evaluation |
| `--run_name` | `auto` | Run directory name (`auto` = timestamp + config tag) |
| `--experiment_root` | `output` | Root directory for all outputs |
| `--output_layout` | `hierarchical` | `hierarchical` (default) or `legacy` |

## Output Structure

```
output/
└── {dataset}/
    └── {scene}/
        ├── ace_fcn_vanilla/
        │   └── {timestamp}_{config_tag}/
        │       ├── run_config.json          # full config snapshot
        │       ├── run_command.txt          # exact command used
        │       ├── run_metadata.json        # ResultManager metadata
        │       ├── training_full_log.txt    # full training log
        │       ├── training_summary.json    # training stats
        │       ├── best_*.pt               # best checkpoint
        │       ├── post_train_eval.txt      # tab-separated eval metrics
        │       └── eval_results/
        │           ├── {ts}_post_train_eval.json
        │           └── {ts}_post_train_eval.txt
        ├── ace_fcn_lmc_iter/               # lmc_flow=iterative
        │   └── {timestamp}_{config_tag}/
        └── ace_fcn_lmc_aceg/              # lmc_flow=ace_g
            └── {timestamp}_{config_tag}/
                └── iteration_results/     # per-iteration eval
```

`config_tag` examples:
- vanilla: `vanilla_buf2.6M_bs5120_ep16`
- iterative: `iter_global_res480_buf2.6M_F7.7M_K128_it4_ep16_bs5120_fm_s1enc_sp768_onecycle`
- ace_g: `aceg_fS2cie_global_res480_buf2.6M_F7.7M_K64_it28_ep24_bs5120_spi_s1buf_sp384_cosine`

## Memory File Format

The pooled memory file must be a `.pt` dict with:

```python
{
    'pooled_points':   torch.Tensor,  # (N, 3)  3D scene points
    'pooled_features': torch.Tensor,  # (N, 512) FCN features
    'scene_center':    torch.Tensor,  # (3,)     scene centroid
}
```

## Performance

Typical results on 7-Scenes Chess:

| Mode | Train time | VRAM | 5cm/5deg |
|---|---|---|---|
| Vanilla (1×) | ~5 min | 6 GB | ~45% |
| Vanilla (3×) | ~15 min | 6 GB | ~50% |
| LMC (4 iter) | ~20 min | 8 GB | ~52% |
| LMC (28 iter) | ~60 min | 8 GB | ~55%+ |

## FAQ

**Q: OOM during training?**
1. Reduce `--batch_size` (e.g. 2560)
2. Reduce `--training_buffer_size`
3. Set `--buffer_on_cpu True`
4. Reduce `--buffer_batch_size`

**Q: OOM during post-train eval?**
Add `--post_train_eval_device cpu`.

**Q: When to use vanilla vs LMC?**
- Vanilla: fast baseline, no memory file needed
- LMC: higher accuracy, requires pre-built pooled memory

**Q: When does iterative training stop?**
- Fixed rounds: `--vanilla_iterations N`
- Best-checkpoint selection: `--eval_each_iteration True --keep_best_only True`

**Q: Difference from DINOv2-LMC?**

| | ACE FCN | DINOv2 |
|---|---|---|
| Encoder | FCN (512-dim, 8×) | ViT-L/14 (1024-dim, 14×) |
| Input | Grayscale (RGB→gray internally) | RGB |
| Resolution constraint | None | Must be multiple of 14 |
| Inference speed | ~30 FPS | ~10–15 FPS |
| VRAM | Low | High |

## Research Workflow Status

| Stage | Status |
|---|---|
| 1. Proposal | complete |
| 2. Review | complete |
| 3. Architecture | complete |
| 4. Skeleton | complete |
| 5. Implementation | complete |
| 6. Evaluation | pending |

To trigger Stage 6: copy eval logs to `04_evaluation/logs/` (use `04_evaluation/collect_logs.py`), then run `/research-eval ace_fcn_lmc`.
