# FGPI Danchor Scene2a Comparison - 2026-05-23

Purpose: record the three completed `scene2a/c0_p4` Danchor runs reported on
2026-05-23, compare their post-train aggregated metrics, and keep the RIO10
adaptation discussion grounded in the fact that Indoor6 remains strong.

## Protocol

- Scene / variant: `scene2a/c0_p4`
- Output root:
  `04_evaluation/train_compare/indoor6_full_baselines_FGPI_level_anchor_residual_d1`
- Shared run suffix:
  `aceg_fS2cie_global_res518_buf2.6M_F7.7M_K64_it28_ep24_bs10240_spi_s1buf_sp384_onecycle_improved`
- Eval: post-train median over seeds `1305,2026,4242,7777,9001`
- DSAC: `hypotheses=256`, aggregation `median`
- Eval frames: `257`

## Results

| ID | Key change | pct5 | pct10_5 | pct2 | pct1 | med_t | med_r | avg ms | Decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Danchor keybias | anchor residual with key-bias weighting | 80.93 | 95.33 | 35.80 | 10.12 | 2.64 cm | 0.28 deg | 110.36 | Best of these three |
| Danchor mixP95h2 | key-bias plus high-percentile mix variant | 81.71 | 94.94 | 29.57 | 5.06 | 2.82 cm | 0.30 deg | 102.30 | Mixed: higher pct5, worse precision |
| Danchor uniform | uniform anchor residual weighting | 80.93 | 93.39 | 28.79 | 6.61 | 2.82 cm | 0.30 deg | 105.90 | Below keybias |

Metric names:

- `pct5`: `5cm/5deg`
- `pct10_5`: `10cm/5deg`
- `pct2`: `2cm/2deg`
- `pct1`: `1cm/1deg`

## Interpretation

- `Danchor keybias` is the clean winner for precision. It ties `uniform` on
  `pct5`, trails `mixP95h2` by only `0.78` pct5, but is clearly stronger on
  `10cm/5deg`, `2cm/2deg`, `1cm/1deg`, median translation, and median rotation.
- `mixP95h2` improves the loose `5cm/5deg` pass rate, but it loses too much at
  high precision: `pct2=29.57` versus `35.80` for `keybias`, and `pct1=5.06`
  versus `10.12`. Treat it as a broad-recall variant, not the default.
- `uniform` is not competitive with `keybias`. The gap at `pct2` is `7.01`
  points and the median pose is also worse.
- These Indoor6 results say the ACE-G path is still capable on the familiar
  scene2a setting. They do not explain the weak RIO10 result. For RIO10, the
  next work should stay on the Step20 diagnostic ladder: data/pose/calibration
  audit, train-set eval, same-split vanilla DINO ACE, DSAC sensitivity, and
  sparse-guided sampling/resize/memory ablations.

## RIO10 Implication

Do not use these strong Indoor6 numbers as a reason to add more architecture
before resolving RIO10 adaptation. The working hypothesis should be:

```text
Indoor6 scene2a remains healthy;
RIO10 failure is likely caused by dataset/eval contract, resize/memory geometry,
domain shift, or sparse-guided sampling behavior.
```

The immediate RIO10 adaptation branch should therefore prioritize:

1. Verify the high-resolution RIO10 memory and training pair:
   `DATASET_RESOLUTION=952` for memory extraction and `IMAGE_RESOLUTION=952`
   for training.
2. Compare RIO10 train-set eval versus public-val eval for the same checkpoint.
3. Train/evaluate same-split vanilla DINO ACE as the non-LMC anchor.
4. Run sparse guided sampling ratios `0.0`, `0.25`, `0.50`, `1.0` only after
   the resize/memory contract is confirmed.

## Provenance

| ID | Run directory | Summary files |
| --- | --- | --- |
| Danchor keybias | `04_evaluation/train_compare/indoor6_full_baselines_FGPI_level_anchor_residual_d1/indoor6_ace/scene2a/dino_ace_lmc_ace_g/scene2a_c0_p4_FGPI_Danchor_keybias_20260522_aceg_fS2cie_global_res518_buf2.6M_F7.7M_K64_it28_ep24_bs10240_spi_s1buf_sp384_onecycle_improved` | `post_train_eval.txt`, `post_train_eval_seed_runs.txt` |
| Danchor mixP95h2 | `04_evaluation/train_compare/indoor6_full_baselines_FGPI_level_anchor_residual_d1/indoor6_ace/scene2a/dino_ace_lmc_ace_g/scene2a_c0_p4_FGPI_Danchor_keybias_mixP95h2_20260522_aceg_fS2cie_global_res518_buf2.6M_F7.7M_K64_it28_ep24_bs10240_spi_s1buf_sp384_onecycle_improved` | `post_train_eval.txt`, `post_train_eval_seed_runs.txt` |
| Danchor uniform | `04_evaluation/train_compare/indoor6_full_baselines_FGPI_level_anchor_residual_d1/indoor6_ace/scene2a/dino_ace_lmc_ace_g/scene2a_c0_p4_FGPI_Danchor_uniform_20260522_aceg_fS2cie_global_res518_buf2.6M_F7.7M_K64_it28_ep24_bs10240_spi_s1buf_sp384_onecycle_improved` | `post_train_eval.txt`, `post_train_eval_seed_runs.txt` |

