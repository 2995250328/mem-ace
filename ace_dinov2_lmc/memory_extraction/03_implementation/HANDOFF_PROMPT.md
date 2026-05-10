# Handoff Prompt: ACE-DINOv2-LMC C1 / Aux Depth / 4090 Migration

You are continuing work on `/home/xwh/project/ace_depth/ace_dinov2_lmc`. The user is moving most future experiments to a 4090 machine. Continue in Chinese unless the user asks otherwise. Be concise, but preserve exact paths and commands.

## Read First

Read these files before changing code or launching long runs:

1. `/home/xwh/project/ace_depth/ace_dinov2_lmc/memory_extraction/03_implementation/PROGRESS_reference_consistent_memory.md`
2. `/home/xwh/project/ace_depth/ace_dinov2_lmc/memory_extraction/03_implementation/KEY_RESULTS_reference_consistent_memory.md`
3. `/home/xwh/project/ace_depth/ace_dinov2_lmc/memory_extraction/03_implementation/EXECUTION_PLAN_reference_consistent_memory.md`
4. `/home/xwh/project/ace_depth/ace_dinov2_lmc/ACE_G_GLOBAL_CODE_AUDIT_TODO.md`
5. `/home/xwh/project/ace_depth/ace_dinov2_lmc/trainer_dinov2_lmc.py`
6. `/home/xwh/project/ace_depth/ace_dinov2_lmc/test_ace_dinov2_lmc.py`
7. `/home/xwh/project/ace_depth/ace_dinov2_lmc/options_dinov2_lmc.py`
8. `/home/xwh/project/ace_depth/ace_compressor.py`

Run commands from `/home/xwh/project/ace_depth`. Use `conda activate mapanything_new`; do not use `conda run`.

## Mainline Summary

The current workstream tests whether C1 reference-normalized coordinate learning can be improved enough to support larger multi-scene training. Historical baseline facts:

- Best validated scene2a single-memory C0 is repaired C0, not C1: best `acc5=76.65`, median translation about `3.015 cm`.
- Earlier scene2a C1 works end-to-end but trails repaired C0: best `acc5=71.98`.
- scene3 single-memory remains diagnostic/fallback; do not record it as successful single-forward policy.
- scene3 cluster C1 ensemble is validated but still slightly weaker than C0 cluster ensemble.

## Recent Code Changes That Must Exist on the 4090 Machine

1. C1 aux depth/reference supervision:
   - files: `trainer_dinov2_lmc.py`, `options_dinov2_lmc.py`, `memory_extraction/check_aux_depth_alignment.py`
   - flags: `--c1_aux_ref_loss_weight`, `--c1_aux_depth_root`, `--c1_aux_depth_kind`
   - off by default; old behavior is restored by omitting `--c1_aux_ref_loss_weight` or setting it to `0.0`.

2. Correct aux depth alignment:
   - Do not align ACE RGB and WAI depth by same filename.
   - ACE export can renumber frames.
   - Correct logic reads WAI `scene_meta.json`, matches ACE pose + calibration to WAI frame metadata, then uses that frame's `gt_depth`.
   - Verified for scene2a: filename-only match was `4377/4890`; scene-meta pose/calibration match is `4890/4890`, `bad_intrinsics=0`, `missing_depth=0`.

3. Deterministic GeoLMC FPS start:
   - files: `/home/xwh/project/ace_depth/ace_compressor.py`, `options_dinov2_lmc.py`, `trainer_dinov2_lmc.py`, `test_ace_dinov2_lmc.py`
   - flag: `--lmc_fps_start_policy`
   - default: `farthest_from_center`
   - rollback old behavior with `--lmc_fps_start_policy legacy_random`
   - fixes the audit issue where training and inference could compress different latent coordinates due to random FPS start.

4. LMC config persistence:
   - checkpoints now include structure-affecting fields such as `num_fine`, `num_coarse`, `geo_sigma`, `pe_normalize_input`, `backbone_feature_dim`, and `lmc_fps_start_policy`.
   - eval rebuilds `GeoLMC` from saved config.

Compile check that should pass:

```bash
cd /home/xwh/project/ace_depth/ace_dinov2_lmc
PYTHONPYCACHEPREFIX=/tmp /home/xwh/miniforge3/envs/mapanything_new/bin/python -m py_compile \
  /home/xwh/project/ace_depth/ace_compressor.py \
  options_dinov2_lmc.py trainer_dinov2_lmc.py test_ace_dinov2_lmc.py \
  memory_extraction/check_aux_depth_alignment.py
```

## Existing 4090 Paired Runs

Aux-ref exploratory run:

```text
/home/xwh/project/ace_depth/ace_dinov2_lmc/04_evaluation/train_compare/memory_pooled_vs_asb_c1/indoor6_ace/scene2a/dino_ace_lmc_ace_g/20260508_000246_aceg_fS2cie_global_res518_buf2.6M_F7.7M_K64_it28_ep24_bs10240_spi_s1buf_sp384_onecycle_improved
```

Control exploratory run:

```text
/home/xwh/project/ace_depth/ace_dinov2_lmc/04_evaluation/train_compare/memory_pooled_vs_asb_c1/indoor6_ace/scene2a/dino_ace_lmc_ace_g/20260508_000323_aceg_fS2cie_global_res518_buf2.6M_F7.7M_K64_it28_ep24_bs10240_spi_s1buf_sp384_onecycle_improved
```

Interpretation:

- These two 4090 runs can be compared to each other as exploratory paired ablation: `aux_ref=0.1` vs `aux_ref=0.0`.
- Do not compare them strictly against old 3090 runs without labeling as `4090 exploratory`.
- Both produced iter summaries and best checkpoints, but final `buffer_size_final=7.68M` hit OOM with `--buffer_on_cpu False`.
- Future full runs should add `--buffer_size_final 2560000`, or use `--buffer_on_cpu True`.

OOM cause:

```text
FINAL S2 buffer allocation on GPU
buffer_size_final = 7680000
buffer_on_cpu = False
24GB card ran out of memory
```

Recommended fix:

```bash
--buffer_on_cpu False \
--buffer_size_final 2560000 \
--batch_size 10240
```

## First 4090 Task: Re-evaluate Existing Pair with Same 5 Seeds

Run this before launching more training:

```bash
cd /home/xwh/project/ace_depth
conda activate mapanything_new

AUX_DIR="/home/xwh/project/ace_depth/ace_dinov2_lmc/04_evaluation/train_compare/memory_pooled_vs_asb_c1/indoor6_ace/scene2a/dino_ace_lmc_ace_g/20260508_000246_aceg_fS2cie_global_res518_buf2.6M_F7.7M_K64_it28_ep24_bs10240_spi_s1buf_sp384_onecycle_improved"
CTRL_DIR="/home/xwh/project/ace_depth/ace_dinov2_lmc/04_evaluation/train_compare/memory_pooled_vs_asb_c1/indoor6_ace/scene2a/dino_ace_lmc_ace_g/20260508_000323_aceg_fS2cie_global_res518_buf2.6M_F7.7M_K64_it28_ep24_bs10240_spi_s1buf_sp384_onecycle_improved"

AUX_CKPT="$AUX_DIR/best_K64_it28_scene2a_c1_auxref01_detfps_20260508_000244.pt"
CTRL_CKPT="$CTRL_DIR/best_K64_it28_scene2a_c1_detfps_ctrl_20260508_000321.pt"

for SEED in 1305 2026 4242 7777 9001; do
  ACE_DATA_ROOT=/home/xwh/data \
  python ace_dinov2_lmc/test_ace_dinov2_lmc.py \
    /home/xwh/data/indoor6_ace/scene2a \
    "$AUX_CKPT" \
    --device cuda:0 \
    --output_dir "$AUX_DIR" \
    --session "auxref01_seed${SEED}" \
    --hypotheses 256 \
    --eval_deterministic True \
    --dsacstar_seed "$SEED" &

  ACE_DATA_ROOT=/home/xwh/data \
  python ace_dinov2_lmc/test_ace_dinov2_lmc.py \
    /home/xwh/data/indoor6_ace/scene2a \
    "$CTRL_CKPT" \
    --device cuda:1 \
    --output_dir "$CTRL_DIR" \
    --session "ctrl_seed${SEED}" \
    --hypotheses 256 \
    --eval_deterministic True \
    --dsacstar_seed "$SEED" &

  wait
done
```

Summarize:

```bash
grep -H "accuracy_5cm5deg_pct\\|accuracy_10cm5deg_pct\\|accuracy_25cm5deg_pct\\|median" \
  "$AUX_DIR"/eval_summary_scene2a_auxref01_seed*.txt \
  "$CTRL_DIR"/eval_summary_scene2a_ctrl_seed*.txt
```

Acceptance check:

- If `aux_ref=0.1` clearly beats control across repeated evals, rerun the paired experiment fully with `--buffer_size_final 2560000`.
- If gain is small or negative, do not expand aux-ref to all scenes yet.

## Clean Paired 4090 Training Commands

Run a clean scene2a pair if strict results are needed.

Aux-ref:

```bash
cd /home/xwh/project/ace_depth
conda activate mapanything_new
TS=$(date +%Y%m%d_%H%M%S)

ACE_DATA_ROOT=/home/xwh/data \
python ace_dinov2_lmc/train_ace_dinov2_lmc.py \
  /home/xwh/data/indoor6_ace/scene2a \
  "scene2a_c1_auxref01_detfps_bf2p6_${TS}.pt" \
  --train_preset memory_compare_ace_g_v1 \
  --data_backend ace \
  --device cuda:0 \
  --post_train_eval_device cuda:0 \
  --use_lmc True \
  --memory_path /home/xwh/project/ace_depth/ace_dinov2_lmc/memory_extraction/04_evaluation/memory_extract/scene2a/40v_v0.05_bilinear_bse_ut0.02_noaug_asb_adaptive_pool1.0_rgate4_prepair8_sor_gm_l2/20260507_195231/memory_bse.pt \
  --lmc_mode global \
  --lmc_fps_start_policy farthest_from_center \
  --experiment_subdir memory_pooled_vs_asb_c1 \
  --c1_aux_ref_loss_weight 0.1 \
  --c1_aux_depth_root /home/xwh/data/mapanything-dataset/wai_data/indoor6/scene2a_train \
  --c1_aux_depth_kind gt_depth \
  --buffer_on_cpu False \
  --buffer_size_final 2560000 \
  --batch_size 10240 \
  --post_train_eval_seeds 1305 2026 4242 7777 9001 \
  --post_train_hypotheses 256
```

Control:

```bash
cd /home/xwh/project/ace_depth
conda activate mapanything_new
TS=$(date +%Y%m%d_%H%M%S)

ACE_DATA_ROOT=/home/xwh/data \
python ace_dinov2_lmc/train_ace_dinov2_lmc.py \
  /home/xwh/data/indoor6_ace/scene2a \
  "scene2a_c1_detfps_ctrl_bf2p6_${TS}.pt" \
  --train_preset memory_compare_ace_g_v1 \
  --data_backend ace \
  --device cuda:1 \
  --post_train_eval_device cuda:1 \
  --use_lmc True \
  --memory_path /home/xwh/project/ace_depth/ace_dinov2_lmc/memory_extraction/04_evaluation/memory_extract/scene2a/40v_v0.05_bilinear_bse_ut0.02_noaug_asb_adaptive_pool1.0_rgate4_prepair8_sor_gm_l2/20260507_195231/memory_bse.pt \
  --lmc_mode global \
  --lmc_fps_start_policy farthest_from_center \
  --experiment_subdir memory_pooled_vs_asb_c1 \
  --c1_aux_ref_loss_weight 0.0 \
  --buffer_on_cpu False \
  --buffer_size_final 2560000 \
  --batch_size 10240 \
  --post_train_eval_seeds 1305 2026 4242 7777 9001 \
  --post_train_hypotheses 256
```

## If Doing Full All-Scene Retraining on 4090

Do not immediately launch a large matrix. Recommended escalation:

1. scene2a paired control vs aux-ref.
2. If aux-ref helps, run one cross-scene validation pair, preferably scene4a because previous C1 was strong.
3. Only after both pass, run all available single-memory C1 scenes.
4. Keep cluster experiments separate from single-memory comparisons.

Suggested future full-matrix defaults:

- single-memory scenes: `scene1`, `scene2a`, `scene4a`, `scene5`, `scene6`
- keep scene3 single-memory diagnostic unless using cluster fallback
- compare `aux_ref=0.1` vs `aux_ref=0.0` only where clean C1 memory exists
- always use:
  - `--lmc_fps_start_policy farthest_from_center`
  - `--buffer_size_final 2560000`
  - `--post_train_eval_seeds 1305 2026 4242 7777 9001`
  - `--post_train_hypotheses 256`

## Required Aux-Depth Data Check Per Scene

Before any aux-ref run:

```bash
cd /home/xwh/project/ace_depth/ace_dinov2_lmc
conda activate mapanything_new

/home/xwh/miniforge3/envs/mapanything_new/bin/python \
  memory_extraction/check_aux_depth_alignment.py \
  --ace-train-root /home/xwh/data/indoor6_ace/scene2a/train \
  --wai-scene-root /home/xwh/data/mapanything-dataset/wai_data/indoor6/scene2a_train \
  --depth-kind gt_depth
```

Pass condition:

```text
scene_meta_alignment:
  status=complete
  matched=<all ACE RGB frames>
  bad_intrinsics=0
  missing_depth=0
```

Do not trust filename-only `matched`; ACE may renumber frames.

## Cautions

- 4090 and 3090 results can be compared only as exploratory cross-hardware context. Strict ablations must be same hardware, same code, same memory, same evaluation seeds.
- Do not overwrite older 3090 tables with 4090 values unless clearly labeled.
- If evaluation is cheap, use 5 repeated eval passes.
- If `--buffer_on_cpu False`, never leave default final buffer expansion on 24GB cards unless you know memory is sufficient.
- Old checkpoints do not store old random FPS latent coordinates. After deterministic FPS code changes, old checkpoint eval may not exactly reproduce original compression unless config explicitly carries the old policy.

## Update After New Runs

After new code or experiments, update:

1. `/home/xwh/project/ace_depth/ace_dinov2_lmc/memory_extraction/03_implementation/PROGRESS_reference_consistent_memory.md`
2. `/home/xwh/project/ace_depth/ace_dinov2_lmc/memory_extraction/03_implementation/KEY_RESULTS_reference_consistent_memory.md`
3. `/home/xwh/project/ace_depth/ace_dinov2_lmc/memory_extraction/03_implementation/EXECUTION_PLAN_reference_consistent_memory.md` if the matrix/plan changes.

Keep 4090 exploratory results labeled separately until a clean paired rerun is complete.
