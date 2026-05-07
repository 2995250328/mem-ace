---
name: dl-vision-gpu-performance
description: Use when a deep learning or vision project needs GPU tuning, CUDA debugging, memory reduction, throughput improvement, profiling, or batch-size scaling. Focus on measurement first, then device placement, data loading, precision, and kernel/runtime bottlenecks.
---

# DL / Vision GPU Performance

Use this skill for CUDA OOM, low utilization, slow throughput, batch-size tuning, dataloader stalls, mixed precision issues, and profiling-driven optimization.

## Workflow

1. Baseline first.
   Record device, precision, batch size, steps/sec, memory usage, utilization, dataloader workers, and whether the bottleneck is train, eval, or preprocessing.
2. Classify the bottleneck.
   Separate compute-bound, memory-bound, input-pipeline-bound, and synchronization-bound cases.
3. Check device placement.
   Verify model, buffer, generator, tensors, and sampled indices are on compatible devices.
4. Check throughput knobs.
   Review batch size, gradient accumulation, AMP, compile mode, pinned memory, worker count, prefetching, persistent workers, and checkpoint frequency.
5. Profile before editing architecture.
   Use lightweight profiling first, then focused traces around the slow stage.
6. Change one thing at a time.
   Re-measure after every material change.

## Rules

- Never recommend larger batch size without checking optimizer semantics and memory headroom.
- Treat `buffer_on_cpu`, sampling generators, and random-index devices as contract points.
- Separate throughput optimizations from training-dynamics changes.
