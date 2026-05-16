# Step 04 - LMC Mode Authority

Status: done

## Decision

The mode requested on the command line is authoritative by default.

`lmc_auto_mode_by_visibility` now defaults to `False`, so `--lmc_mode global`
trains an effective global model unless the user explicitly opts into
visibility-based fallback.

## Rationale

The fallback statistic only estimates whether sampled memory points lie in
front of each training camera (`z > 0` in camera coordinates). It does not test
image-frame projection validity, occlusion, depth consistency, or mask
validity. For a multi-view point cloud that represents the whole scene, this
coarse global front-ratio can be low even when global compression is the
intended experimental condition.

Automatically changing `global` into `local` makes experiment names, commands,
and checkpoints easy to misread. This is especially risky for ACE-G global
baselines.

## Implementation

- `options_dinov2_lmc.py`: `--lmc_auto_mode_by_visibility` default changed from
  `True` to `False`.
- `trainer_dinov2_lmc.py`: missing-option compatibility default changed from
  `True` to `False`.
- Existing fallback code remains available when explicitly requested with
  `--lmc_auto_mode_by_visibility True`.
- Existing metadata keeps recording requested and effective LMC modes.

## Expected Log Contract

Default global run:

```text
[LMC] requested_mode=global effective_mode=global auto_by_visibility=False
```

Explicit opt-in fallback run may still show:

```text
[LMC] Global mode auto-fallback triggered: ...
[LMC] requested_mode=global effective_mode=local auto_by_visibility=True
```
