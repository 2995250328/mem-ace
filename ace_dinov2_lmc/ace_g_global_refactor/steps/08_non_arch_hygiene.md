# 08 - Non-Architecture Hygiene And Fail-Fast Checks

## Purpose

Clean up small engineering hazards before changing ACE-G fusion, compressor,
or head architecture.

This step is intentionally non-architectural:

- no new loss
- no fusion or compressor behavior change
- no head architecture change
- no change to the normal ACE-G training math

## Changes

### Train preset contract

`memory_compare_ace_g_v2` already existed in `TRAIN_PRESET_DEFAULTS`, but
`--train_preset` did not accept it.

The parser choices now include `memory_compare_ace_g_v2`.

Default behavior:

- unchanged for `none`
- unchanged for `memory_compare_ace_g_v1`
- unchanged for `ace_g_indoor6_4090_global_fixedzero_v1`

Compatibility:

- old commands keep their previous meaning
- the already-defined v2 preset is now explicitly requestable

### LMC memory feature-dim fail-fast

If `pooled_features_dim` is not divisible by `num_layers`, training now raises
`ValueError` immediately.

The check is centralized in `_resolve_lmc_feature_dim(...)` so it can be
smoke-tested without constructing a full trainer.

The error reports:

- `pooled_features_dim`
- `num_layers`
- `layers_idx`
- `memory_path`

Behavioral effect:

- valid memory files are unchanged
- invalid memory/config combinations fail earlier instead of reaching a later
  shape error

### Head-grid packing visibility

Sampled feature rows are still packed into a fake `H x W` grid for the existing
1x1 ACE head. The fixed height remains `16`.

The helper is now named `_pack_feature_rows_for_head(...)`, with the previous
`_trim_batch_for_head_grid(...)` kept as a compatibility wrapper.

Tail trimming behavior is preserved, but each stage now accumulates:

- `trim_events`
- `trimmed_rows`

The first warning for a stage includes the cumulative values.

## Documentation Constraint

Every completed code, configuration, training-semantics, or experiment-flow
change must update `ace_g_global_refactor/PLAN.md` or the corresponding
`ace_g_global_refactor/steps/*.md` file.

A change without documentation is not considered complete.

## Verification

Required lightweight checks:

```bash
cd /home/xwh/project/ace_depth/ace_dinov2_lmc
conda run -n mapanything python -m py_compile options_dinov2_lmc.py trainer_dinov2_lmc.py train_ace_dinov2_lmc.py test_ace_dinov2_lmc.py /home/xwh/project/ace_depth/ace_compressor.py
```

Additional checks:

- parser accepts `--train_preset memory_compare_ace_g_v2`
- synthetic feature-dim mismatch raises `ValueError` with full context
- `git diff --check`

## Non-Goals

- No compressor/fusion attention diagnostics in this step.
- No GeoMatch Fusion v1.
- No usage loss.
- No coordinate normalization contract change.
- No sub-memory or multi-scene routing change.
