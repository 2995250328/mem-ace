# Handoff Prompt: ACE-G Promising Architecture Directions

You are discussing future ACE-G architecture changes for `/home/xwh/project/ace_depth/ace_dinov2_lmc`.

The user wants research/architecture feedback, not immediate training. Continue in Chinese unless asked otherwise. Be concrete: tie every idea to exact code locations, current results, and risks. Avoid proposing already-negative reruns unchanged.

## Working directory and project context

- Run commands from: `/home/xwh/project/ace_depth`
- Main subproject: `/home/xwh/project/ace_depth/ace_dinov2_lmc`
- Main trainer: `/home/xwh/project/ace_depth/ace_dinov2_lmc/trainer_dinov2_lmc.py`
- Main compressor: `/home/xwh/project/ace_depth/ace_compressor.py`
- Main fusion module: `/home/xwh/project/ace_depth/ace_fusion.py`
- Main CLI parser: `/home/xwh/project/ace_depth/ace_dinov2_lmc/options_dinov2_lmc.py`
- Main eval loader: `/home/xwh/project/ace_depth/ace_dinov2_lmc/test_ace_dinov2_lmc.py`

## Current accepted reference

Current clean reference is FGPI-4090 / `4090_forceglobal_s1_periter`:

```text
true global
lmc_profile=legacy
s1_loss_step_mode=per_iter
lmc_key_slice_idx=2  # selected DINO/MapAnything memory layer, layer12-like
lmc_feature_hierarchy_mode=selected_key_concat_value
lmc_fusion_geometry_mode=value_only_raw
pct5=83.66
pct2=34.24
med_t=2.6233 cm
med_r=0.2831 deg
```

When proposing new experiments, keep this contract unless the intended variable is explicitly changed.

## Already negative / do not rerun unchanged

- A0/A1/A2 and `forceglobal_fixedzero` default candidate.
- CPE-A1 / `geokey_norm` / `value_only_norm` as standalone default directions.
- B1 `scalar_mix` current form: collapsed to layer12 and underperformed.
- Key-layer ablations: key1/layer6 and key3/layer18 were already negative; layer12 remains the selected-key reference.
- TP / `mapany_flow_v1` profile was negative; do not combine it with new architecture tests by default.
- B3-lite current `levelwise_latent_merge` was negative; do not rerun unchanged.
- GBR-main and CRPB-main current forms were negative; do not rerun unchanged.

## Recent geometry result snapshot

Post-train aggregated median over seeds `1305,2026,4242,7777,9001`, hypotheses=256, scene2a/c0_p4:

| Run | Main change | pct5 | pct2 | med_t | med_r | Judgment |
|---|---|---:|---:|---:|---:|---|
| FGPI-4090 ref | current clean reference | 83.66 | 34.24 | 2.623 cm | 0.283° | baseline |
| GBR-main | residual multi-scale RBF distance bias | 78.99 | 30.35 | 2.90 cm | 0.30° | negative |
| FPE-main | Fourier PE v2 residual | 80.93 | 30.35 | 2.82 cm | 0.29° | below baseline |
| PointRoPE-main | axis-wise 3D RoPE on q/k | 81.71 | 33.46 | 2.73 cm | 0.28° | best geometry candidate, still below baseline |
| CRPB-main | MLP continuous relative position bias | 79.38 | 28.79 | 3.02 cm | 0.29° | negative |

Interpretation: PointRoPE is the only geometry direction worth further discussion. FPE-v2 is second but not strong enough. GBR/CRPB in their current forms should be rejected unless the discussion identifies a concrete bug or substantially different formulation.

## Direction 1: Multi-level feature compression

Goal: use multiple DINO/MapAnything feature levels inside the memory compressor more effectively than selected layer12 only.

Current code locations:

- CLI flags: `/home/xwh/project/ace_depth/ace_dinov2_lmc/options_dinov2_lmc.py`
  - `--lmc_key_slice_idx`: lines around `628-635`
  - `--lmc_key_feature_mode`: lines around `637-644`
  - `--lmc_feature_hierarchy_mode`: lines around `646-653`
  - `--lmc_level_merge_mode`, `--lmc_level_proj_shared`, `--lmc_level_cross_attn_shared`: lines around `655-683`
- Constructor wiring: `/home/xwh/project/ace_depth/ace_dinov2_lmc/trainer_dinov2_lmc.py`
  - builds `GeoLMC(...)`: lines around `1461-1508`
  - passes `feature_hierarchy_mode`, level merge flags, geo bias, and PE flags.
- Compressor implementation: `/home/xwh/project/ace_depth/ace_compressor.py`
  - `GeoLMC.__init__`: starts around line `548`
  - feature hierarchy validation: lines around `616-638`
  - levelwise modules: lines around `693-710`, `785-831`
  - per-layer feature helpers: lines around `882-896`
  - `GeoLMC._forward_levelwise_latent_merge`: lines around `999-1082`
  - default forward path: lines around `1086-1117+`

Current result status:

- Current B3-lite (`--lmc_feature_hierarchy_mode levelwise_latent_merge`) failed:
  - B3L-main: `pct5=77.43`, `pct2=24.90`
  - B3L-diag: `pct5=80.16`, `pct2=28.02`
- This does not prove multi-level features are useless. It suggests the current compression-only merge is the wrong place/form: it creates multiple level-specific compressed latents and then globally gates them before dense prediction.

Promising discussion questions:

1. Should multi-level memory compression preserve a strong layer12 anchor instead of replacing it?
   - Current B3-lite merges all levels into one latent tensor via global softmax gate at `ace_compressor.py:1063-1066`.
   - Alternative: keep baseline layer12 compressed tokens as the main path and add a zero-gated residual from multi-level compressed tokens.
   - This would make old behavior exactly recoverable at initialization.

2. Should compression be teacher/auxiliary rather than direct replacement?
   - Use multi-level memory features to supervise/regularize selected-layer compressor outputs, rather than changing the runtime compressed memory format.
   - This avoids destabilizing S2/fusion/head distribution.

3. Should level gating be token-wise or spatially conditioned?
   - Current code explicitly reserves `level_token_gate` but disallows it for B3-lite: `ace_compressor.py:635-638`.
   - A token-wise or geometry-aware gate may be more appropriate than one global level gate.

Suggested safer prototype:

```text
baseline_z = selected_key_concat_value(...)
ml_z = levelwise_compress(...)
out_z = baseline_z + gamma * adapter(ml_z)
# gamma zero-initialized
```

This should be treated as a new mechanism, not as a rerun of B3-lite.

## Direction 2: Multi-level feature fusion

Goal: use multi-level features closer to dense query prediction, where DPT/FPN-style architectures usually benefit dense tasks.

Current code locations:

- Fusion module: `/home/xwh/project/ace_depth/ace_fusion.py`
  - `LMCFusionBlock`: starts around line `11`
  - current attention path: lines around `109-157`
  - memory position encoding injection: lines around `127-142`
  - `LMCFeatureFusion`: starts around line `187`
  - hierarchical coarse/fine fusion: lines around `202-257`
  - normal single fusion: lines around `258-266`
- Trainer usage: `/home/xwh/project/ace_depth/ace_dinov2_lmc/trainer_dinov2_lmc.py`
  - builds `LMCFeatureFusion(...)`: lines around `1511-1520`
  - `_compress_memory`: around line `2404`
  - `_fuse_features`: around line `2658`
  - ACE-G S2 fused feature use: lines around `4756-4763`
- Eval usage: `/home/xwh/project/ace_depth/ace_dinov2_lmc/test_ace_dinov2_lmc.py`
  - rebuilds compressor/fusion from checkpoint config: lines around `307-390`
  - compresses memory once and caches it: lines around `426-442`
  - applies fusion to each test batch: lines around `490-525`

Current fusion behavior:

- Query features are dense image/backbone tokens, shape `(B, N_q, C)`.
- Memory is compressed to `K=64` tokens plus 3D latent coordinates.
- `LMCFusionBlock.forward` cross-attends `query_feats -> memory_z/memory_p`, then residual + FFN.
- Geometry currently enters memory value via Fourier PE, and optionally key under `geokey_norm`.

Promising discussion questions:

1. Query-side DPT/FPN adapter before ACE head
   - Instead of only changing memory compression, fuse DINOv2 intermediate image-side layers into a dense feature map before ACE regression.
   - This is closer to DPT/Depth Anything-style dense prediction.
   - Need to locate image feature extraction path in trainer/backbone code before implementation; likely outside `ace_fusion.py`, in DINOv2 feature extraction / buffer-fill path in `trainer_dinov2_lmc.py` and base `trainer_dinov2.py`.

2. Perceiver-IO style memory-to-query decoder
   - Current `LMCFeatureFusion` is a simple cross-attention block from dense query tokens to compressed memory tokens.
   - A more explicit decoder could use query image tokens as output queries over memory latents, possibly with multiple lightweight layers and zero-gated residual.
   - Best insertion point: replace or extend `LMCFeatureFusion.forward` in `ace_fusion.py:223-266`.

3. Aligned late multi-level residual
   - Keep current fusion output unchanged as the anchor.
   - Add a DPT/FPN-style residual branch after alignment:

```text
fused = baseline_fusion(query, memory)
residual = AlignFuse([query_L6, query_L12, query_L18, query_L24])
out = fused + gamma * residual
```

   - `gamma` should be zero-initialized.
   - This is safer than replacing the whole query feature distribution.

Caution: do not treat B3-lite failure as evidence against DPT-Q / dense fusion. B3-lite altered memory compression; it did not test image-side dense multi-level fusion.

## Direction 3: Fourier positional encoding replacement / modification

Goal: improve coordinate representation used by the compressor, replacing or modifying the crude Fourier PE.

Current code locations:

- PE implementation: `/home/xwh/project/ace_depth/ace_compressor.py`
  - `FourierPositionEncoding`: lines around `36-139`
  - legacy random Gaussian Fourier branch: lines around `86-121`
  - v2 residual branch: lines around `94-139`
  - `GeoLMC.pe_encoder` construction: lines around `678-692`
- PointRoPE implementation:
  - `DecoupledCrossAttention._apply_point_rope`: lines around `341-369`
  - q/k application in attention: lines around `380-389`
- CLI flags: `/home/xwh/project/ace_depth/ace_dinov2_lmc/options_dinov2_lmc.py`
  - `--lmc_pos_encoding_mode`: lines around `723-728`
  - `--lmc_pos_fourier_v2_scales`: lines around `730-735`
  - `--lmc_pos_fourier_radius`: lines around `743-748`
  - `--lmc_point_rope_*`: lines around `762-794`

Current tested variants:

- `fourier_legacy`: accepted reference path.
- `fourier_v2`: below baseline (`pct5=80.93`, `pct2=30.35`).
- `point_rope`: best geometry candidate (`pct5=81.71`, `pct2=33.46`) but still below FGPI.

Promising discussion questions:

1. PointRoPE sensitivity
   - Current PointRoPE uses axis-wise xyz channel split and radius 4.0.
   - Candidate sensitivity: radius 2.0 or learned/scene-derived radius.
   - Keep `--lmc_geo_bias_mode legacy` for isolated tests.

2. PointRoPE + residual Fourier, cautiously
   - Current code sets `GeoLMC.pe_encoder` to legacy Fourier when `pos_encoding_mode == point_rope`: `ace_compressor.py:686`.
   - PointRoPE only modulates attention q/k; it does not remove the legacy PE used to seed positional content.
   - Discussion question: should PointRoPE be combined with a normalized Fourier residual or should it fully replace legacy PE?

3. Deterministic / non-random Fourier basis
   - Legacy branch uses random `B_gauss`: `ace_compressor.py:86-87`.
   - v2 uses fixed scales but as a residual MLP branch.
   - Alternative: deterministic sinusoidal coordinate features, SIREN-like mapping, or small MLP over normalized coordinates with zero-gated residual.

Recommended near-term priority:

- PointRoPE sensitivity is more promising than more FPE-v2 changes, because PointRoPE nearly matched `pct2` and median rotation.
- Any change should preserve legacy initialization or include a legacy-control run.

## Direction 4: Distance bias replacement / modification

Goal: improve geometric attention bias in compressor attention logits.

Current code locations:

- Bias implementation: `/home/xwh/project/ace_depth/ace_compressor.py`
  - `LocalGeometricBias`: lines around `180-192`
  - `AdaptiveGeometricBias`: lines around `195-208`
  - `ResidualRBFDistanceBias`: lines around `211-272`
  - `DecoupledCrossAttention.__init__`: lines around `282-338`
  - CRPB MLP construction: lines around `323-333`
  - `_crpb_bias`: lines around `371-378`
  - adding bias to logits: lines around `390-398`
  - geometry info assembly in `GeoLMC`: lines around `951-963`, `1025-1031`
- CLI flags: `/home/xwh/project/ace_depth/ace_dinov2_lmc/options_dinov2_lmc.py`
  - `--lmc_geo_bias_mode`: lines around `691-696`
  - `--lmc_geo_bias_rbf_*`: lines around `697-720`
  - `--lmc_geo_bias_crpb_*`: lines around `795-824`

Current tested variants:

- `legacy`: accepted reference.
- `rbf_residual`: negative in current GBR-main (`pct5=78.99`, `pct2=30.35`).
- `crpb`: negative in current CRPB-main (`pct5=79.38`, `pct2=28.79`).

Promising discussion questions:

1. Was zero-initialization too conservative or too weak?
   - RBF residual starts with `alpha_init=0.0`: `ResidualRBFDistanceBias` around `ace_compressor.py:235-260`.
   - CRPB output can be zero-initialized: `ace_compressor.py:331-333`.
   - If gradients are weak, the bias may not meaningfully move. Need diagnostics before rerunning.

2. Should distance bias be signed/vector-aware rather than mostly radial?
   - RBF is radial only.
   - CRPB uses `[dx, dy, dz, dist, logdist]`, but current small MLP underperformed.
   - Better alternatives may borrow from Point Transformer: relative positional encoding added to keys/values, not only scalar logit bias.

3. Should distance bias be query-token/fusion-side rather than compressor-side?
   - Current bias affects latent-to-memory compression attention.
   - It may be more useful in dense query-to-memory fusion, where query pixels need geometry-conditioned memory selection.
   - Candidate insertion point: `LMCFusionBlock.forward` attention logits at `ace_fusion.py:149-150`; currently no distance/coordinate bias is used there beyond memory PE in values.

4. Should bias be learned with constraints?
   - Current CRPB MLP may learn arbitrary bias and hurt calibration.
   - Alternative: monotonic radial basis with bounded residual gate, or per-head temperature on existing legacy distance bias.

Recommended near-term priority:

- Do not rerun GBR/CRPB unchanged.
- If exploring distance bias, first add diagnostics: final alpha, bias norm relative to qk logits, attention entropy shift, per-head stats, and whether bias changes across training.
- Consider moving distance-aware logic into fusion rather than only compressor.

## Suggested ranking for discussion

1. Multi-level feature fusion / query-side DPT-style adapter
   - Most conceptually aligned with dense prediction literature.
   - Not invalidated by B3-lite failure.
   - Needs careful code-path discovery for DINOv2 intermediate image features.

2. PointRoPE refinement
   - Best empirical geometry candidate so far.
   - Concrete code already exists; sensitivity/combination is easy.

3. Safer multi-level compression with baseline anchor
   - Current B3-lite failed, but a residual anchored version may still be worth discussing.
   - Do not replace baseline layer12 path outright.

4. Distance bias redesign
   - Current GBR/CRPB failed.
   - Worth discussing only if reformulated with diagnostics or moved to fusion-side attention.

## Files to read first for a web-model discussion

1. `/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor/EXPERIMENT_REFERENCE.md`
2. `/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor/steps/12_geobias_rbf_and_fourier_pe_v2_plan.md`
3. `/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor/steps/13_query_side_dpt_adapter_multilevel_plan.md`
4. `/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor/steps/14_point_rope_and_continuous_relative_bias_plan.md`
5. `/home/xwh/project/ace_depth/ace_compressor.py`
6. `/home/xwh/project/ace_depth/ace_fusion.py`
7. `/home/xwh/project/ace_depth/ace_dinov2_lmc/trainer_dinov2_lmc.py`
8. `/home/xwh/project/ace_depth/ace_dinov2_lmc/options_dinov2_lmc.py`

## Discussion constraints

- Do not recommend `mapany_flow_v1` as default; it was negative.
- Do not recommend key-layer ablations; key1/key3 are already negative, key2/layer12 remains accepted.
- New experiments should change one primary mechanism when possible.
- Prefer zero-gated residual branches when adding multi-level or geometry modules.
- Preserve checkpoint config compatibility: any new CLI flag affecting architecture must be stored in `lmc_config` and used in `test_ace_dinov2_lmc.py` reconstruction.
- Preserve old default behavior byte-for-byte where possible; if a change touches existing code paths, expose it behind a new flag with legacy default.

## Good questions to ask the web models

1. Given that B3-lite multi-level compression failed, what architecture would preserve the layer12 baseline while extracting useful low/high-level information?
2. Should multi-level DINOv2 features enter the memory compressor, the dense query feature path, or the query-to-memory fusion block?
3. How would you design a DPT/FPN-style query-side adapter with zero-gated residual output for ACE scene coordinate regression?
4. PointRoPE nearly matched pct2 but not pct5. Is this likely a radius/normalization issue, a q/k application issue, or a conflict with the existing Fourier PE seed?
5. Should distance bias be a scalar attention-logit bias, a relative key/value encoding, or a fusion-side geometric prior?
6. What diagnostics would distinguish “bias module ineffective” from “bias module harmful” for GBR/CRPB?
