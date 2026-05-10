# ACE-G Global Refactor Plan

Scope: `--lmc_flow ace_g` with effective `lmc_mode=global`.

This file is the high-level tracker only. Keep detailed decisions and concrete
implementation notes in one file per step under `steps/`.

## Status Legend

- `[todo]`: not started.
- `[designing]`: goal accepted, exact implementation still being discussed.
- `[ready]`: concrete implementation measure has been agreed.
- `[done]`: code/docs verified for this step.
- `[deferred]`: moved out of the near-term ACE-G global track.

## Plan

1. `[done]` Stabilize baseline reproducibility.
   - Deterministic global FPS start policy.
   - Detail: `steps/01_deterministic_fps.md`

2. `[todo]` Make global compressor key-layer selection explicit.
   - Replace implicit `_get_layer_slice(..., 1)` semantics with an explicit
     config field and checkpoint metadata.
   - Keep current actual behavior as the compatibility default until ablation
     says otherwise.

3. `[todo]` Save and report all structure-affecting LMC metadata.
   - Record requested/effective LMC mode, memory layer metadata, selected key
     slice, and future geometry/fusion mode fields.
   - Keep test-time reconstruction driven by checkpoint config.

4. `[todo]` Make S2 compressor freezing explicit.
   - Add a clear trainability boundary for S1/S2.
   - Prevent future live-compressor S2 edits from silently creating gradients.

5. `[todo]` Unify reprojection and invalid-loss behavior.
   - Consolidate S1 full-map, S1 sampled, S2, and S2-G loss paths around one
     shared contract.
   - Preserve existing behavior unless a difference is intentionally changed.

6. `[todo]` Make S1 sampled loss time-axis explicit.
   - Expose/log whether sampled S1 uses fixed-zero, per-iteration, or monotonic
     ReproLoss scheduling.

7. `[todo]` Add compressor and fusion observability.
   - Log key layer metadata, latent token count, PE coordinate scale, distance
     bias scale, and fusion attention diagnostics.

8. `[todo]` Add GeoMatch Fusion v1 as a controlled ablation.
   - Keep current value-only fusion as baseline.
   - Add gated geometry injection into both key and value paths.

9. `[todo]` Consider usage regularization only after diagnostics.
   - Add token usage loss only if attention metrics show collapse.
   - Start with S1-only ablations.

10. `[deferred]` Larger research changes.
    - Anchor residual branch as primary path.
    - Full normalized-coordinate target training.
    - DSD/local/deformable compressor.
    - Sub-memory routing.
    - ACE head replacement / MoE architectures.
