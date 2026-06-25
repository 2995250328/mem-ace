---
name: ace-lmc-gpu-concurrency-scheduler
description: Use when launching, modifying, or reviewing ACE-DINOv2-LMC quick fusion training with multiple processes per GPU, especially when scheduling quick fusion 1iter matrices, avoiding CUDA_VISIBLE_DEVICES/device mapping mistakes, separating train and post-train eval, tuning num_data_loader_workers/eval_num_workers, or deciding safe per-GPU concurrency.
---

# ACE LMC GPU Concurrency Scheduler

## Overview

Use the scheduled launcher for safe quick fusion placement and optional multi-process GPU scheduling. By default it preserves legacy-safe one-job-per-GPU behavior; 2x per-GPU train concurrency is explicit opt-in. The launcher encodes the 2026-06-23 concurrency probe finding: single RTX 4090 2x train concurrency is reasonable for quick fusion, but 3x is not validated and train+eval mixing should be avoided.

## Hard Rules

- Run training/eval commands from `/home/xwh/project/ace_depth`.
- Use `conda run --no-capture-output -n mapanything python` for train/eval Python commands.
- Long training matrices must run inside tmux.
- Do not wrap training with external `CUDA_VISIBLE_DEVICES=<gpu>` and then pass `--device cuda:0`.
- Always pass the physical GPU to the script: `--device cuda:<physical_gpu>` and `--post_train_eval_device cuda:<physical_gpu>`.
- Immediately verify placement after launch with `nvidia-smi`, process command lines, and the log line `Info: CUDA_VISIBLE_DEVICES = <physical_gpu>`.
- Treat low VRAM as insufficient evidence for high concurrency; check CPU load, buffer build throughput, dataloader worker count, IO, eval, non-finite loss, and CUDA errors.

## Default Launcher

Use:

```bash
cd /home/xwh/project/ace_depth
tmux new-session -d -s <session> \
  "RUN_ROOT=/data/xwh/ace_dinov2_lmc/04_evaluation/<run_name> \
   GPU_LIST='0 1' \
   MAX_JOBS_PER_GPU=1 \
   TRAIN_WORKERS=12 \
   EVAL_WORKERS=6 \
   EVAL_MODE=inline \
   bash ace_dinov2_lmc/scripts/launch_quick_fusion_v3_adapter_matrix_scheduled.sh"
```

The script:

- unsets inherited `CUDA_VISIBLE_DEVICES` before every Python command
- passes physical `--device cuda:${gpu}`
- schedules train jobs in waves, with `MAX_JOBS_PER_GPU` slots per GPU
- staggers same-GPU jobs by `STAGGER_SECONDS`
- defaults to legacy-compatible `EVAL_MODE=inline`; use `EVAL_MODE=deferred` only when opting into train concurrency
- writes `status.tsv`, per-job logs, `placement_checks.log`, and `best_metric_summary.log`

## Safe Settings

Recommended production defaults after the GPU2 probe:

- Default and rollback baseline: `MAX_JOBS_PER_GPU=1`, `TRAIN_WORKERS=12`, `EVAL_WORKERS=6`, `EVAL_MODE=inline`.
- Conservative throughput mode: `MAX_JOBS_PER_GPU=2`, `TRAIN_WORKERS=6`, `EVAL_WORKERS=2`, `EVAL_MODE=deferred`, `MAX_EVAL_JOBS=1`, `STAGGER_SECONDS=90`.
- Probe-only mode: use idle GPUs with `GPU_LIST='2 3'`, shorter buffers or fewer scenes, and a separate `RUN_ROOT`.
- Do not use `MAX_JOBS_PER_GPU=3` for real runs until a clean 3x probe succeeds on an idle GPU.
- Do not use `EVAL_MODE=inline` with `MAX_JOBS_PER_GPU>1` unless the user explicitly accepts train+eval mixing risk and sets `ALLOW_INLINE_EVAL_WITH_CONCURRENCY=true`.

## Usage Examples

Dry run one small probe on GPU2:

```bash
cd /home/xwh/project/ace_depth
RUN_ROOT=/data/xwh/ace_dinov2_lmc/04_evaluation/concurrency_dryrun \
GPU_LIST='2' \
SCENES='wayspots_tendrils wayspots_map' \
VARIANTS='single' \
MAX_JOBS_PER_GPU=2 \
TRAINING_BUFFER_SIZE=1000000 \
BUFFER_SIZE_FINAL=1000000 \
EPOCHS=1 \
DRY_RUN=true \
bash ace_dinov2_lmc/scripts/launch_quick_fusion_v3_adapter_matrix_scheduled.sh
```

Legacy-safe quick fusion scheduled matrix on GPU0/1:

```bash
cd /home/xwh/project/ace_depth
tmux new-session -d -s quick_fusion_scheduled_$(date +%Y%m%d_%H%M) \
  "RUN_ROOT=/data/xwh/ace_dinov2_lmc/04_evaluation/quick_fusion_scheduled_$(date +%Y%m%d_%H%M) \
   GPU_LIST='0 1' \
   SCENES='wayspots_bears wayspots_squarebench wayspots_cubes' \
   VARIANTS='single pmrf_base cl_pmrf_v3 v3_adapter_control' \
   MAX_JOBS_PER_GPU=1 \
   TRAIN_WORKERS=12 \
   EVAL_WORKERS=6 \
   EVAL_MODE=inline \
   MAX_EVAL_JOBS=1 \
   STAGGER_SECONDS=90 \
   bash ace_dinov2_lmc/scripts/launch_quick_fusion_v3_adapter_matrix_scheduled.sh"
```

After launch, verify:

```bash
tmux list-windows -t <session>
nvidia-smi
ps -eo pid,ppid,pgid,etime,pcpu,pmem,rss,cmd | rg 'train_ace_dinov2_lmc|eval_best_post_train|conda run'
tail -n 120 <run_root>/logs/placement_checks.log
```

## Opt-In 2x Train Concurrency

Enable 2x per-GPU train concurrency only by passing all of these explicitly:

```bash
MAX_JOBS_PER_GPU=2 \
TRAIN_WORKERS=6 \
EVAL_WORKERS=2 \
EVAL_MODE=deferred \
MAX_EVAL_JOBS=1 \
STAGGER_SECONDS=90
```

This mode runs train jobs concurrently and defers post-train eval to a limited eval phase.

## Interpreting Health

During training, watch:

- each process reports the intended `Info: CUDA_VISIBLE_DEVICES = <gpu>`
- each process command includes `--device cuda:<same gpu>`
- same-GPU buffer build times do not inflate by more than roughly 35-40 percent
- logs do not contain `Traceback`, `RuntimeError`, `CUDA error`, `out of memory`, `naninf` spikes, or `nonFinite` spikes
- CPU load and IO wait do not climb into a sustained bottleneck
- post-train eval runs in the separate eval phase when `EVAL_MODE=deferred`

## Known Probe Result

On 2026-06-23, GPU2 quick fusion short probe with 1M/1M buffers showed:

- single tendrils: 121.7s total
- paired tendrils+map on one RTX 4090: tendrils 135.4s, map 140.6s
- tendrils same-scene slowdown about 11 percent
- no OOM, CUDA error, DataLoader deadlock, or non-finite loss

Use that as support for `MAX_JOBS_PER_GPU=2`, not for 3x concurrency.
