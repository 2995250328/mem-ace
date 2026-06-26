#!/usr/bin/env bash
set -euo pipefail

cd /home/xwh/project/ace_depth

OUT_ROOT="/data/xwh/ace_dinov2_lmc/04_evaluation/cambridge_stage2_highhyp1024_eval_20260625_gpu3"
LOG_DIR="${OUT_ROOT}/logs"
mkdir -p "${LOG_DIR}"

HYPOTHESES=1024
SEEDS=(1305 2026 4242)

run_eval() {
  local scene_name="$1"
  local scene_path="$2"
  local checkpoint="$3"
  local out_subdir="$4"

  local out_dir="${OUT_ROOT}/${out_subdir}/${scene_name}"
  mkdir -p "${out_dir}"

  for seed in "${SEEDS[@]}"; do
    local session="highhyp${HYPOTHESES}_seed${seed}"
    local log_file="${LOG_DIR}/${scene_name}_${session}.log"
    echo "[highhyp] scene=${scene_name} seed=${seed} hypotheses=${HYPOTHESES}" | tee "${log_file}"
    conda run --no-capture-output -n mapanything \
      python ace_dinov2_lmc/test_ace_dinov2_lmc.py \
        "${scene_path}" \
        "${checkpoint}" \
        --data_backend ace \
        --device cuda:3 \
        --output_dir "${out_dir}" \
        --session "${session}" \
        --hypotheses "${HYPOTHESES}" \
        --eval_deterministic True \
        --dsacstar_seed "${seed}" \
        --dsacstar_seed_per_frame True \
        --eval_num_workers 6 \
        --image_resolution 518 \
        --ace_encoder_path /home/xwh/project/ace_depth/ace_encoder_pretrained.pt \
        --glace_root /home/xwh/project/glace \
        --glace_feat_name features.npy \
      2>&1 | tee -a "${log_file}"
  done
}

run_eval \
  "Cambridge_GreatCourt" \
  "/data/xwh/Cambridge/Cambridge_GreatCourt" \
  "/data/xwh/ace_dinov2_lmc/04_evaluation/cambridge_stage2_fused_backbone_glace_it10_buf10m_final12m_20260624_gpu23/stage2_fused_backbone_glace_it10_buf10m_final12m/Cambridge/Cambridge_GreatCourt/dino_ace_lmc_ace_g/20260624_235359_aceg_fS1_global_res512_buf10M_F12M_K64_it10_ep24_bs4096_spi_s1buf_sp512_onecycle_improved/best_K64_it10_ace_fcn_fused_backbone_glace_stage2.pt" \
  "stage2_fused_backbone_glace_it10"

run_eval \
  "Cambridge_ShopFacade" \
  "/data/xwh/Cambridge/Cambridge_ShopFacade" \
  "/data/xwh/ace_dinov2_lmc/04_evaluation/cambridge_stage2_fused_backbone_glace_it10_buf10m_final12m_20260624_gpu23/stage2_fused_backbone_glace_it10_buf10m_final12m/Cambridge/Cambridge_ShopFacade/dino_ace_lmc_ace_g/20260624_235359_aceg_fS1_global_res512_buf10M_F12M_K64_it10_ep24_bs4096_spi_s1buf_sp512_onecycle_improved/best_K64_it10_ace_fcn_fused_backbone_glace_stage2.pt" \
  "stage2_fused_backbone_glace_it10"

run_eval \
  "Cambridge_StMarysChurch" \
  "/data/xwh/Cambridge/Cambridge_StMarysChurch" \
  "/data/xwh/ace_dinov2_lmc/04_evaluation/cambridge_stmary_stage2_fused_from_global_it30_it10_buf10m_final12m_20260625_gpu0/stage2_fused_from_global_it30_it10_buf10m_final12m/Cambridge/Cambridge_StMarysChurch/dino_ace_lmc_ace_g/20260625_102008_aceg_fS1_global_res512_buf10M_F12M_K64_it10_ep24_bs4096_spi_s1buf_sp512_onecycle_improved/best_K64_it10_ace_fcn_fused_globalit30_glace_stage2.pt" \
  "stage2_fused_from_global_it30_glace_it10"

echo "[highhyp] done: ${OUT_ROOT}"
