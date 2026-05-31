# ACE-FCN LMC Two-Stage Summary (2026-05-30)

## Scope

Scene: `wayspots_bears`

Goal:
- verify whether LMC works when the query and memory share the same ACE-FCN feature space
- then test whether GLACE image-level global features further help as a final-head conditioning signal
- compare that against the mismatched-memory ablation using the older MapAnything/BSE memory

## Reference Baseline

Baseline summary source:

- `/home/xwh/project/ace_depth/ace_dinov2_lmc/04_evaluation/wayspots_baselines/20260525_190055/summary.md`

Reference row:

| method | 25cm/5deg | 10cm/5deg | 5cm/5deg | 2cm/2deg | 1cm/1deg | median deg | median cm |
|---|---:|---:|---:|---:|---:|---:|---:|
| `ace` | 95.345 | 86.034 | 72.586 | 6.552 | 0.862 | 1.105 | 3.510 |

## ACE-FCN Memory Sanity

Memory file:

- `/data/xwh/ace_dinov2_lmc/04_evaluation/ace_fcn_lmc_two_stage/memory_ace_fcn_sparse_sp_r4.pt`
- sanity report: `/data/xwh/ace_dinov2_lmc/04_evaluation/ace_fcn_lmc_two_stage/memory_ace_fcn_sparse_sp_r4.pt.sanity.json`

Key sanity values:

| metric | value |
|---|---:|
| `hard_pass` | `True` |
| `nn_cosine median` | `0.8401` |
| `random_cosine median` | `0.3521` |
| `nn_cosine_margin_median` | `0.4880` |
| `nn_3d_error median (m)` | `0.0880` |
| `random_3d_error median (m)` | `1.1062` |
| `nn_3d_error_ratio_median` | `0.0795` |
| `pooled_points` | `7293` |

Interpretation:

- ACE-FCN query features can retrieve geometrically meaningful ACE-FCN memory points.
- The sanity check is strong enough to justify Stage 1 training.

## Results

| experiment | memory feature space | global head | 25cm/5deg | 10cm/5deg | 5cm/5deg | 2cm/2deg | 1cm/1deg | median deg | median cm |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `ACE baseline` | n/a | no | 95.345 | 86.034 | 72.586 | 6.552 | 0.862 | 1.105 | 3.510 |
| `Stage1 ACE-memory` | `ace_fcn` | no | 94.66 | 88.97 | 78.79 | 13.28 | 1.03 | 0.9612 | 3.0456 |
| `Stage2 ACE-memory + GLACE concat` | `ace_fcn` | `glace_concat` | 96.90 | 92.24 | 82.24 | 18.62 | 1.38 | 0.8722 | 3.0163 |
| `Stage1 MapAnything-memory` | `mapanything/bse` mismatch | no | 71.03 | 50.69 | 17.24 | 1.03 | 0.17 | 2.8148 | 9.5638 |
| `Stage2 MapAnything-memory + GLACE concat` | `mapanything/bse` mismatch | `glace_concat` | 73.79 | 50.52 | 17.41 | 0.52 | 0.00 | 2.6876 | 9.8238 |

Run sources:

- Stage1 ACE-memory:
  `/data/xwh/ace_dinov2_lmc/04_evaluation/ace_fcn_lmc_two_stage/stage1_local_ace_memory_it12/extracted/wayspots_bears/dino_ace_lmc_ace_g/20260529_174038_aceg_fS2cie_global_res512_buf2.8M_F7.6M_K64_it12_ep24_bs4096_spi_s1buf_sp512_onecycle_improved/post_train_eval.txt`
- Stage2 ACE-memory + GLACE concat:
  `/data/xwh/ace_dinov2_lmc/04_evaluation/ace_fcn_lmc_two_stage/stage2_glace_concat_it12/extracted/wayspots_bears/dino_ace_lmc_ace_g/20260529_195852_aceg_fS2cie_global_res512_buf2.8M_F7.6M_K64_it12_ep24_bs4096_spi_s1buf_sp512_onecycle_improved/post_train_eval.txt`
- Stage1 MapAnything-memory:
  `/data/xwh/ace_dinov2_lmc/04_evaluation/ace_fcn_lmc_two_stage/stage1_mapanything_memory_it12/extracted/wayspots_bears/dino_ace_lmc_ace_g/20260530_002538_aceg_fS2cie_global_res512_buf2.8M_F7.6M_K64_it12_ep24_bs4096_spi_s1buf_sp512_onecycle_improved/post_train_eval.txt`
- Stage2 MapAnything-memory + GLACE concat:
  `/data/xwh/ace_dinov2_lmc/04_evaluation/ace_fcn_lmc_two_stage/stage2_mapanything_memory_glace_concat_it12/extracted/wayspots_bears/dino_ace_lmc_ace_g/20260530_021351_aceg_fS2cie_global_res512_buf2.8M_F7.6M_K64_it12_ep24_bs4096_spi_s1buf_sp512_onecycle_improved/post_train_eval.txt`

## Main Conclusions

1. ACE-FCN memory is the critical ingredient.
   With matched ACE-FCN memory, Stage 1 already beats the original ACE baseline:
   `5cm/5deg: 72.586 -> 78.79`, `2cm/2deg: 6.552 -> 13.28`.

2. GLACE image-level global conditioning provides additional gains on top of the matched local stack.
   `5cm/5deg: 78.79 -> 82.24`, `2cm/2deg: 13.28 -> 18.62`.

3. Feature-space mismatch is strongly destructive.
   Reusing the older MapAnything/BSE memory collapses performance in both Stage 1 and Stage 2, despite keeping the rest of the ACE-FCN pipeline unchanged.

## One-Sentence Takeaway

LMC works in this setting only when the query and memory share the ACE-FCN feature space, and once that local stack is established, GLACE image-level global features further improve disambiguation through a frozen-local `glace_concat` final head.
