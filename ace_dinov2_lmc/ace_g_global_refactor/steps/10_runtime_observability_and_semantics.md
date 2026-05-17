# 10 - Runtime Observability And Experiment Semantics

## Purpose

Finish the remaining non-structural cleanup before compressor/fusion
architecture ablations.

This step does not change:

- compressor layer selection
- fusion key/value formula
- PE or distance-bias geometry
- ACE head architecture
- reprojection loss math
- requested/effective LMC mode behavior

## Changes

### Optional runtime diagnostics

New CLI options:

```text
--lmc_log_runtime_stats False
--lmc_runtime_stats_interval 100
--lmc_runtime_stats_max_pixels 4096
```

Default behavior is unchanged.

When enabled, fusion can return scalar diagnostics through an explicit
`return_stats=True` path:

- attention entropy mean / p10 / p50 / p90
- effective token count
- average max attention
- token usage min / max / top5
- raw feature norm
- attention output norm
- fused feature norm

The normal fusion API still returns only the fused tensor unless stats are
explicitly requested.

### Compressor runtime stats

When runtime diagnostics are enabled, memory compression logs:

- effective latent token count
- selected key slice and key layer label
- latent coordinate radius mean/std
- centered coordinate std
- global distance-log min/max/std

These are scalar logs only. Large attention tensors are not saved.

### Experiment semantics in summaries

New checkpoints record the runtime diagnostics config plus:

- `lmc_flow`
- `lmc_auto_mode_by_visibility`
- `ace_g_fusion_in_s2`

Best-checkpoint metadata, post-train evaluation summaries, and test evaluation
summaries now include the main experiment semantics:

- requested/effective LMC mode
- auto visibility fallback flag
- LMC flow
- key slice and layer label
- memory layer list
- FPS start policy
- S1 sampled loss step mode
- ACE-G S2 fusion mode

This makes requested-global/effective-local runs easier to distinguish from
true-global runs without reading the full training log.

### Stage contract guards

S1 logs trainable parameter counts for compressor, fusion, head, and encoder.

S2-G verifies:

- compressor parameters have `requires_grad=False`
- the S2 optimizer does not contain compressor parameters

Violations raise immediately because they invalidate ACE-G experiment semantics.

## Compatibility

Valid default runs should be unchanged.

Diagnostics are opt-in. With `--lmc_log_runtime_stats False`, the fusion and
compressor math follow the existing path.

Old checkpoints still load. Missing semantic fields are written as `None` in
new eval summaries when the checkpoint does not contain them.

## Verification

Required lightweight checks:

```bash
cd /home/xwh/project/ace_depth/ace_dinov2_lmc
conda run -n mapanything python -m py_compile options_dinov2_lmc.py trainer_dinov2_lmc.py train_ace_dinov2_lmc.py test_ace_dinov2_lmc.py /home/xwh/project/ace_depth/ace_compressor.py /home/xwh/project/ace_depth/ace_fusion.py
```

Additional smoke checks:

- parser accepts the new runtime stats options
- fusion default output matches `return_stats=True` fused output in eval mode
- fusion stats are finite scalar values
- S2-G optimizer guard detects compressor-parameter leakage

## Non-Goals

- No GeoMatch Fusion v1.
- No key-layer ablation.
- No scene-scale PE or distance-bias normalization.
- No usage regularization.
- No normalized-coordinate target training.
- No multi-scene or sub-memory changes.
