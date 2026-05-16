# 09 - Module Train/Eval Mode Contract

## Purpose

Tighten helper-level train/eval state handling before adding new ACE-G
architecture ablations.

This step is intentionally non-architectural:

- no loss change
- no fusion/compressor/head architecture change
- no optimizer membership change
- no change to requested/effective LMC mode semantics
- no change to normal ACE-G forward math

## Problem

Several helpers temporarily switch modules to eval mode for deterministic
feature extraction or memory compression.

The previous restoration pattern was too broad in a few places:

- `_compress_memory()` restored compressor mode only after a successful forward.
- `create_training_buffer(...)` restored compressor/fusion modes only through
  explicit conditional `.train()` calls.
- `create_training_buffer_ace_g(...)` called `regressor.train()` after raw
  buffer creation, which can turn child modules such as the DINO encoder back
  to train mode even if the caller had intentionally kept them in eval mode.
- `_create_training_buffer_with_scene_coords(...)` also restored with broad
  `regressor.train()` semantics in the normal path.

This is a state-contract issue, not a model-design issue.

## Changes

### Exact module mode snapshot

Added two small helpers:

- `_capture_module_training_modes(...)`
- `_restore_module_training_modes(...)`

They capture exact `.training` flags for each module and restore them in the
same order. Parent modules are listed before child modules where needed, so
child-specific states such as `encoder.eval()` are restored after the parent
state is restored.

### Compressor memory compression

`_compress_memory()` now restores compressor train/eval mode in a `finally`
block.

Behavioral effect:

- successful compression is unchanged
- failed compression no longer leaves the compressor stuck in eval mode

### Fused-buffer construction

`create_training_buffer(...)` now captures and restores:

- compressor
- fusion
- regressor
- regressor.encoder
- regressor.heads

The feature path and buffer contents are unchanged.

### ACE-G raw-buffer construction

`create_training_buffer_ace_g(...)` now captures and restores:

- regressor
- regressor.encoder
- regressor.heads

This preserves the ACE-G S2 expectation that the DINO encoder can remain eval
while the heads are trainable.

### Shared scene-coordinate buffer helper

The normal completion path of `_create_training_buffer_with_scene_coords(...)`
now restores the exact regressor/encoder/heads mode snapshot rather than
calling broad `regressor.train()`.

## Compatibility

Normal training behavior should be unchanged for valid runs.

The only intended difference is defensive:

- helper calls no longer silently broaden train mode after temporary eval-mode
  feature extraction
- compressor mode is restored even if memory compression raises

This does not change:

- S1 sampled loss behavior
- S1 full-map loss behavior
- S2 or S2-G loss behavior
- compressor trainability boundaries
- fusion mode selected by R1/R2
- checkpoint metadata format

## Verification

Required lightweight checks:

```bash
cd /home/xwh/project/ace_depth/ace_dinov2_lmc
conda run -n mapanything python -m py_compile trainer_dinov2_lmc.py
```

Additional smoke checks:

- synthetic parent/child module snapshot restores mixed train/eval state
- `_compress_memory()` restores compressor mode after an injected exception

## Non-Goals

- No GeoMatch Fusion v1.
- No fusion/token observability.
- No usage regularization.
- No coordinate normalization contract change.
- No sub-memory or multi-scene routing change.
