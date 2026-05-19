# CPE / B1 Scene2a Comparison - 2026-05-18

Purpose: record the three completed `scene2a/c0_p4` runs launched after the
compressor PE scene-scale and B1 scalar-mix changes, compare them against the
current clean 4090 reference, and prevent duplicate reruns.

## Protocol

- Scene / variant: `scene2a/c0_p4`
- Memory: `memory_extraction/04_evaluation/memory_extract/scene2a/40v_v0.05_bilinear_bse_ut0.02_noaug_asb_adaptive_pool1.0_rgate4_prepair8_sor_gm_l2/20260508_103423/memory_bse.pt`
- Shared flags: `lmc_mode=global`, `lmc_auto_mode_by_visibility=False`,
  `lmc_key_slice_idx=2`, `s1_loss_step_mode=per_iter`, batch `10240`
- Eval: post-train median over seeds `1305,2026,4242,7777,9001`,
  `post_train_hypotheses=256`
- Reference: clean `FGPI-4090` true-global per-iter, 5-run median:
  `pct5=83.66`, `pct2=34.24`, `med_t=2.6233`, `med_r=0.2831`

## Results

| ID | Key change | pct5 | pct10_5 | pct2 | pct1 | med_t | med_r | avg ms | Decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| FGPI-4090 | clean true-global per-iter reference | 83.66 | 95.33 | 34.24 | 6.23 | 2.6233 | 0.2831 | 168.31 | Current reference |
| A1 | value-only normalized fusion reference | 80.54 | n/a | 35.41 | n/a | 2.6117 | 0.2803 | n/a | Diagnostic/mixed |
| CPE-A1 | A1 + compressor PE scene-scale | 81.32 | 93.77 | 30.74 | 5.84 | 2.6946 | 0.2781 | 104.76 | Reject as default |
| B1 | key learned scalar mix, value concat unchanged | 79.77 | 94.55 | 31.91 | 7.00 | 2.8101 | 0.2877 | 102.52 | Reject current form |
| B1-diag | same as B1 + runtime diagnostics | 76.65 | 94.55 | 32.68 | 6.23 | 2.7682 | 0.3028 | 106.29 | Diagnostic complete |

Runtime values are not used as the main decision signal because the old FGPI
reference was run under a different runtime context. Pose metrics are the
decision basis here.

## Seed-Wise Stability

| ID | pct5 range | pct2 range | med_t range | med_r range |
| --- | ---: | ---: | ---: | ---: |
| CPE-A1 | 79.38-82.49 | 29.57-33.07 | 2.6197-2.7862 | 0.2699-0.2885 |
| B1 | 77.43-80.93 | 28.79-33.07 | 2.6697-2.9164 | 0.2825-0.2949 |
| B1-diag | 76.26-78.99 | 30.35-35.02 | 2.7388-2.9632 | 0.2879-0.3059 |

## Interpretation

- `CPE-A1` is not worth promoting. It slightly improves `pct5` relative to A1
  (`81.32` vs `80.54`) but is still well below `FGPI-4090` (`83.66`) and hurts
  high-precision `pct2` relative to both A1 and FGPI.
- Current `B1 scalar_mix` is negative. It trails FGPI on `pct5`, `pct2`,
  `med_t`, and `med_r`; the diagnostic repeat is worse, so this is not a
  single unlucky eval seed effect.
- B1 did not actually explore a meaningful multi-layer mixture. The saved
  `lmc_key_mix_weights` remained almost entirely on layer 12:
  `0.00034,0.00034,0.99862,0.00034,0.00034`.
- Runtime diagnostics show compressor key mode was active, but the sharp
  initialization kept the effective key contract close to the old slice-2
  behavior while adding another trainable degree of freedom. This current B1
  form should not be expanded.

## Provenance

| ID | Run directory | Checkpoint | Summary |
| --- | --- | --- | --- |
| CPE-A1 | `04_evaluation/train_compare/indoor6_full_baselines_4090_CPE_A1_scene_scale/indoor6_ace/scene2a/dino_ace_lmc_ace_g/20260518_121339_aceg_fS2cie_global_res518_buf2.6M_F7.7M_K64_it28_ep24_bs10240_spi_s1buf_sp384_onecycle_improved` | `best_K64_it28_scene2a_c0_p4_CPE_A1_scene_scale_20260518.pt` | `post_train_eval.txt` |
| B1 | `04_evaluation/train_compare/indoor6_full_baselines_4090_B1_scalar_mix/indoor6_ace/scene2a/dino_ace_lmc_ace_g/20260518_121345_aceg_fS2cie_global_res518_buf2.6M_F7.7M_K64_it28_ep24_bs10240_spi_s1buf_sp384_onecycle_improved` | `best_K64_it28_scene2a_c0_p4_B1_scalar_mix_20260518.pt` | `post_train_eval.txt` |
| B1-diag | `04_evaluation/train_compare/indoor6_full_baselines_4090_B1_scalar_mix_diag/indoor6_ace/scene2a/dino_ace_lmc_ace_g/20260518_121352_aceg_fS2cie_global_res518_buf2.6M_F7.7M_K64_it28_ep24_bs10240_spi_s1buf_sp384_onecycle_improved` | `best_K64_it28_scene2a_c0_p4_B1_scalar_mix_diag_20260518.pt` | `post_train_eval.txt` |

## Next Decision

- Do not rerun `CPE-A1` unchanged.
- Do not rerun current sharp-init `B1 scalar_mix` unchanged.
- If key mixing is revisited, it must be a different mechanism, e.g. softer
  initialization, entropy regularization, or per-layer projected keys. Treat
  that as a new B1b experiment, not a repeat of B1.
- The next structural direction should probably skip unchanged scalar mix and
  move to a more explicit feature hierarchy test only after deciding whether
  the added capacity is acceptable.
