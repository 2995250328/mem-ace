#!/usr/bin/env bash
set -euo pipefail

# Single-scene rerun for Cambridge_KingsCollege concat stage2.
# This keeps the same protocol as the completed Cambridge concat-stage2 runs:
#   stage1_fused backbone + GLACE global concat + reinitialized head
#   it10, buffer 10M, final buffer 12M, deterministic post-train seeds.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

CONDA_ENV="${CONDA_ENV:-mapanything}"
SCENE="${SCENE:-Cambridge_KingsCollege}"
GPU="${GPU:-0}"
CAMBRIDGE_ROOT="${CAMBRIDGE_ROOT:-/data/xwh/Cambridge}"
SOURCE_RUN_ROOT="${SOURCE_RUN_ROOT:-/home/xwh/project/ace_depth/ace_dinov2_lmc/04_evaluation/cambridge_single_stage12_checked_20260624_gpu23}"
RUN_ROOT="${RUN_ROOT:-/data/xwh/ace_dinov2_lmc/04_evaluation/cambridge_kings_concat_stage2_rerun_20260626_gpu${GPU}}"
STATUS_FILE="${RUN_ROOT}/status.tsv"
DRY_RUN="${DRY_RUN:-false}"

ACE_ENCODER_PATH="${ACE_ENCODER_PATH:-${ROOT_DIR}/ace_encoder_pretrained.pt}"
GLACE_ROOT="${GLACE_ROOT:-/home/xwh/project/glace}"
GLACE_FEAT_NAME="${GLACE_FEAT_NAME:-features.npy}"

POST_TRAIN_HYPOTHESES="${POST_TRAIN_HYPOTHESES:-256}"
POST_TRAIN_EVAL_SEEDS=(${POST_TRAIN_EVAL_SEEDS:-1305 2026 4242})

STAGE2_SUBDIR="${STAGE2_SUBDIR:-stage2_fused_backbone_glace_it10_buf10m_final12m}"
STAGE2_LMC_ITERATIONS="${STAGE2_LMC_ITERATIONS:-10}"
STAGE2_TRAINING_BUFFER_SIZE="${STAGE2_TRAINING_BUFFER_SIZE:-10000000}"
STAGE2_BUFFER_SIZE_FINAL="${STAGE2_BUFFER_SIZE_FINAL:-12000000}"
STAGE2_IMAGE_RESOLUTION="${STAGE2_IMAGE_RESOLUTION:-518}"
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-4096}"
STAGE2_SAMPLES_PER_IMAGE="${STAGE2_SAMPLES_PER_IMAGE:-512}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OMP_DYNAMIC="${OMP_DYNAMIC:-FALSE}"

mkdir -p "${RUN_ROOT}/logs"
if [[ ! -f "${STATUS_FILE}" ]]; then
  printf "timestamp\tscene\tmethod\tstage\tstatus\texit_code\tdevice\tlog\n" > "${STATUS_FILE}"
fi

log_status() {
  local stage="$1" status="$2" exit_code="$3" log_file="$4"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$(date +%Y-%m-%dT%H:%M:%S)" "${SCENE}" "stage2_fused_backbone_glace" \
    "${stage}" "${status}" "${exit_code}" "cuda:${GPU}" "${log_file}" >> "${STATUS_FILE}"
}

print_cmd() {
  printf "%q " "$@"
  printf "\n"
}

run_logged() {
  local stage="$1" log_file="$2"
  shift 2
  mkdir -p "$(dirname "${log_file}")"
  {
    printf "[%s] scene=%s stage=%s device=cuda:%s\n" "$(date)" "${SCENE}" "${stage}" "${GPU}"
    print_cmd "$@"
  } | tee -a "${log_file}"

  if [[ "${DRY_RUN}" == "true" ]]; then
    log_status "${stage}" "dry_run" 0 "${log_file}"
    return 0
  fi

  "$@" 2>&1 | tee -a "${log_file}"
  local exit_code=${PIPESTATUS[0]}
  if [[ "${exit_code}" -eq 0 ]]; then
    log_status "${stage}" "ok" 0 "${log_file}"
  else
    log_status "${stage}" "failed" "${exit_code}" "${log_file}"
  fi
  return "${exit_code}"
}

find_local_stage1_ckpt() {
  find "${SOURCE_RUN_ROOT}/stage1_local_ace_memory_it12" -type f \
    -name "best_K64_it12_ace_fcn_local_stage1.pt" -path "*${SCENE}*" 2>/dev/null | sort | tail -n 1 || true
}

preflight_scene() {
  local scene_root="${CAMBRIDGE_ROOT}/${SCENE}"
  for split in train test; do
    [[ -d "${scene_root}/${split}/rgb" ]] || { echo "missing rgb: ${scene_root}/${split}/rgb" >&2; return 2; }
    [[ -d "${scene_root}/${split}/poses" ]] || { echo "missing poses: ${scene_root}/${split}/poses" >&2; return 2; }
    [[ -d "${scene_root}/${split}/calibration" ]] || { echo "missing calibration: ${scene_root}/${split}/calibration" >&2; return 2; }
    [[ -s "${scene_root}/${split}/${GLACE_FEAT_NAME}" ]] || { echo "missing GLACE feat: ${scene_root}/${split}/${GLACE_FEAT_NAME}" >&2; return 2; }
  done
  [[ -d "${scene_root}/train/sparse_depth" ]] || { echo "missing sparse depth: ${scene_root}/train/sparse_depth" >&2; return 2; }
}

main() {
  local scene_root="${CAMBRIDGE_ROOT}/${SCENE}"
  local memory_path="${SOURCE_RUN_ROOT}/memory/${SCENE}/memory_ace_fcn_sparse_sp_r4.pt"
  local stage1_ckpt
  stage1_ckpt="$(find_local_stage1_ckpt)"
  local output_suffix="ace_fcn_fused_backbone_glace_stage2.pt"
  local log_file="${RUN_ROOT}/logs/stage2_concat_${SCENE}_gpu${GPU}.log"

  [[ -s "${stage1_ckpt}" ]] || { echo "missing stage1 checkpoint for ${SCENE}" | tee -a "${log_file}" >&2; return 2; }
  [[ -s "${memory_path}" ]] || { echo "missing memory for ${SCENE}: ${memory_path}" | tee -a "${log_file}" >&2; return 2; }
  preflight_scene 2>&1 | tee -a "${log_file}"

  printf "RUN_ROOT=%s\n" "${RUN_ROOT}"
  printf "SCENE=%s GPU=%s\n" "${SCENE}" "${GPU}"
  printf "stage1_ckpt=%s\n" "${stage1_ckpt}"
  printf "memory_path=%s\n" "${memory_path}"
  printf "DRY_RUN=%s\n" "${DRY_RUN}"

  run_logged "train" "${log_file}" \
    conda run --no-capture-output -n "${CONDA_ENV}" python ace_dinov2_lmc/train_ace_dinov2_lmc.py \
      "${scene_root}" "${output_suffix}" \
      --model_backend ace_fcn_lmc \
      --data_backend ace \
      --post_train_eval_scene "${scene_root}" \
      --use_lmc True \
      --lmc_flow ace_g \
      --memory_path "${memory_path}" \
      --use_scale_token False \
      --ace_encoder_path "${ACE_ENCODER_PATH}" \
      --ace_lmc_global_head_mode glace_concat \
      --ace_lmc_local_checkpoint_path "${stage1_ckpt}" \
      --ace_lmc_freeze_local_stack True \
      --ace_lmc_stage2_feature_source stage1_fused \
      --ace_lmc_global_feature_mode glace \
      --ace_lmc_global_gate_init 1.0 \
      --ace_lmc_global_gate_learnable False \
      --ace_lmc_global_normalize True \
      --ace_lmc_global_noise_std 0.1 \
      --glace_root "${GLACE_ROOT}" \
      --glace_feat_name "${GLACE_FEAT_NAME}" \
      --glace_head_channels 768 \
      --glace_mlp_ratio 1.0 \
      --num_head_blocks 2 \
      --best_metric median_error \
      --device "cuda:${GPU}" \
      --post_train_eval_device "cuda:${GPU}" \
      --experiment_root "${RUN_ROOT}" \
      --experiment_subdir "${STAGE2_SUBDIR}" \
      --lmc_iterations "${STAGE2_LMC_ITERATIONS}" \
      --num_latent_tokens 64 \
      --lmc_fusion_refinement_mode single \
      --lmc_fusion_cascade_layers 4 \
      --lmc_fusion_assembly_gamma_init 0.0 \
      --s1_loss_step_mode fixed_zero \
      --s1_early_stop False \
      --lmc_log_runtime_stats True \
      --lmc_runtime_stats_interval 100 \
      --lmc_runtime_stats_max_pixels 4096 \
      --num_data_loader_workers 12 \
      --eval_num_workers 6 \
      --eval_deterministic True \
      --eval_dsacstar_seed 1305 \
      --eval_dsacstar_seed_per_frame True \
      --iteration_eval_seed 1305 \
      --iteration_eval_hypotheses 256 \
      --ace_g_fusion_in_s2 False \
      --ace_g_cross_iter_eval False \
      --s1_use_buffer True \
      --s1_loss_mode sample_per_image \
      --s1_buffer_refill_mode full \
      --image_resolution "${STAGE2_IMAGE_RESOLUTION}" \
      --batch_size "${STAGE2_BATCH_SIZE}" \
      --training_buffer_size "${STAGE2_TRAINING_BUFFER_SIZE}" \
      --buffer_size_final "${STAGE2_BUFFER_SIZE_FINAL}" \
      --buffer_on_cpu True \
      --buffer_on_cpu_final True \
      --samples_per_image "${STAGE2_SAMPLES_PER_IMAGE}" \
      --buffer_sample_valid_coords True \
      --buffer_valid_coord_sample_ratio 1.0 \
      --buffer_valid_coord_neighbor_radius 1 \
      --buffer_valid_coord_neighbor_mode cross \
      --c1_aux_depth_root "${scene_root}/train/sparse_depth" \
      --c1_aux_depth_kind sparse_depth \
      --post_train_eval_seeds "${POST_TRAIN_EVAL_SEEDS[@]}" \
      --post_train_hypotheses "${POST_TRAIN_HYPOTHESES}"
}

main "$@"
