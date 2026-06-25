#!/usr/bin/env bash
set -uo pipefail

# Single-GPU training-efficiency matrix for ACE-G stage1 local memory.
# Goal: test whether fewer LMC iterations plus larger buffers can replace it12.
#
# Run from anywhere:
#   GPU_ID=0 bash /home/xwh/project/ace_depth/ace_dinov2_lmc/scripts/launch_single_gpu_train_efficiency_matrix.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

CONDA_ENV="${CONDA_ENV:-mapanything}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-0}"
SCENE="${SCENE:-wayspots_bears}"
WAYSPOTS_ROOT="${WAYSPOTS_ROOT:-/data/xwh/Wayspots}"
RUN_ROOT="${RUN_ROOT:-/data/xwh/ace_dinov2_lmc/04_evaluation/train_efficiency_single_gpu_20260623}"
MEMORY_PATH="${MEMORY_PATH:-/data/xwh/ace_dinov2_lmc/04_evaluation/wayspots_ace_fcn_lmc_suite/20260530_181745/memory/wayspots_bears/memory_ace_fcn_sparse_sp_r4.pt}"
ACE_ENCODER_PATH="${ACE_ENCODER_PATH:-${ROOT_DIR}/ace_encoder_pretrained.pt}"

IMAGE_RESOLUTION="${IMAGE_RESOLUTION:-512}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
SAMPLES_PER_IMAGE="${SAMPLES_PER_IMAGE:-512}"
NUM_LATENT_TOKENS="${NUM_LATENT_TOKENS:-64}"
EPOCHS_DEFAULT="${EPOCHS_DEFAULT:-24}"
POST_TRAIN_HYPOTHESES="${POST_TRAIN_HYPOTHESES:-256}"
ITERATION_EVAL_HYPOTHESES="${ITERATION_EVAL_HYPOTHESES:-256}"
POST_TRAIN_EVAL_SEEDS=(${POST_TRAIN_EVAL_SEEDS:-1305 2026 4242})

mkdir -p "${RUN_ROOT}"
STATUS_FILE="${RUN_ROOT}/status.tsv"
if [[ ! -f "${STATUS_FILE}" ]]; then
  printf "timestamp\tvariant\tscene\tstatus\texit_code\tlog\n" > "${STATUS_FILE}"
fi

log_status() {
  local variant="$1" status="$2" exit_code="$3" log_file="$4"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$(date +%Y-%m-%dT%H:%M:%S)" "${variant}" "${SCENE}" "${status}" "${exit_code}" "${log_file}" >> "${STATUS_FILE}"
}

print_cmd() {
  printf "%q " "$@"
  printf "\n"
}

run_variant() {
  local variant="$1"
  local lmc_iterations="$2"
  local training_buffer_size="$3"
  local buffer_size_final="$4"
  local epochs="${5:-${EPOCHS_DEFAULT}}"

  local scene_root="${WAYSPOTS_ROOT}/${SCENE}"
  local variant_root="${RUN_ROOT}/${variant}"
  local log_dir="${variant_root}/logs"
  local log_file="${log_dir}/train_${SCENE}.log"
  local stage_subdir="stage1_local_ace_memory_it${lmc_iterations}"
  local output_suffix="ace_fcn_local_stage1.pt"

  mkdir -p "${log_dir}"
  {
    printf "[%s] variant=%s scene=%s gpu=%s\n" "$(date)" "${variant}" "${SCENE}" "${GPU_ID}"
    printf "lmc_iterations=%s training_buffer_size=%s buffer_size_final=%s epochs=%s\n" \
      "${lmc_iterations}" "${training_buffer_size}" "${buffer_size_final}" "${epochs}"
  } | tee -a "${log_file}"

  local checkpoint_pattern="best_K${NUM_LATENT_TOKENS}_it${lmc_iterations}_${output_suffix}"
  local existing_ckpt
  existing_ckpt=$(find "${variant_root}/${stage_subdir}" -type f -name "${checkpoint_pattern}" -path "*${SCENE}*" 2>/dev/null | sort | tail -n 1 || true)
  if [[ -n "${existing_ckpt}" && "${SKIP_EXISTING:-true}" == "true" ]]; then
    printf "Skipping existing checkpoint: %s\n" "${existing_ckpt}" | tee -a "${log_file}"
    log_status "${variant}" "skipped_existing" 0 "${log_file}"
    return 0
  fi

  local cmd=(
    conda run --no-capture-output -n "${CONDA_ENV}" "${PYTHON_BIN}" ace_dinov2_lmc/train_ace_dinov2_lmc.py
    "${scene_root}" "${output_suffix}"
    --model_backend ace_fcn_lmc
    --data_backend ace
    --use_lmc True
    --lmc_flow ace_g
    --memory_path "${MEMORY_PATH}"
    --use_scale_token False
    --ace_encoder_path "${ACE_ENCODER_PATH}"
    --ace_lmc_global_head_mode none
    --device "cuda:${GPU_ID}"
    --post_train_eval_device "cuda:${GPU_ID}"
    --experiment_root "${variant_root}"
    --experiment_subdir "${stage_subdir}"
    --lmc_iterations "${lmc_iterations}"
    --num_latent_tokens "${NUM_LATENT_TOKENS}"
    --lmc_fusion_refinement_mode single
    --s1_loss_step_mode fixed_zero
    --s1_early_stop False
    --num_data_loader_workers 12
    --eval_num_workers 6
    --eval_deterministic True
    --eval_dsacstar_seed 1305
    --eval_dsacstar_seed_per_frame True
    --iteration_eval_seed 1305
    --iteration_eval_hypotheses "${ITERATION_EVAL_HYPOTHESES}"
    --ace_g_fusion_in_s2 True
    --ace_g_cross_iter_eval True
    --s1_use_buffer True
    --s1_loss_mode sample_per_image
    --s1_buffer_refill_mode full
    --s1_last_iter_use_final_buffer True
    --image_resolution "${IMAGE_RESOLUTION}"
    --batch_size "${BATCH_SIZE}"
    --training_buffer_size "${training_buffer_size}"
    --buffer_size_final "${buffer_size_final}"
    --buffer_on_cpu True
    --buffer_on_cpu_final True
    --samples_per_image "${SAMPLES_PER_IMAGE}"
    --buffer_sample_valid_coords True
    --buffer_valid_coord_sample_ratio 1.0
    --buffer_valid_coord_neighbor_radius 1
    --buffer_valid_coord_neighbor_mode cross
    --c1_aux_depth_root "${scene_root}/train/sparse_depth"
    --c1_aux_depth_kind sparse_depth
    --epochs "${epochs}"
    --post_train_eval_seeds "${POST_TRAIN_EVAL_SEEDS[@]}"
    --post_train_hypotheses "${POST_TRAIN_HYPOTHESES}"
  )

  print_cmd "${cmd[@]}" | tee -a "${log_file}"
  "${cmd[@]}" 2>&1 | tee -a "${log_file}"
  local exit_code=${PIPESTATUS[0]}
  if [[ ${exit_code} -eq 0 ]]; then
    log_status "${variant}" "ok" 0 "${log_file}"
  else
    log_status "${variant}" "failed" "${exit_code}" "${log_file}"
    if [[ "${CONTINUE_ON_ERROR:-true}" != "true" ]]; then
      exit "${exit_code}"
    fi
  fi
  return "${exit_code}"
}

printf "Run root: %s\n" "${RUN_ROOT}"
printf "Scene   : %s\n" "${SCENE}"
printf "GPU     : %s\n" "${GPU_ID}"
printf "Memory  : %s\n" "${MEMORY_PATH}"

run_variant "single_it01_buf7p6M" 1 7600000 7600000 "${EPOCHS_DEFAULT}"
run_variant "single_it01_buf10M" 1 10000000 10000000 "${EPOCHS_DEFAULT}"
run_variant "single_it02_buf5M_final7p6M" 2 5000000 7600000 "${EPOCHS_DEFAULT}"

conda run --no-capture-output -n "${CONDA_ENV}" "${PYTHON_BIN}" \
  ace_dinov2_lmc/.codex_skills/lmc-experiment-standard-workflow/scripts/best_metric_summary.py \
  "${RUN_ROOT}" | tee "${RUN_ROOT}/best_metric_summary.log" || true

printf "Done. Root: %s\n" "${RUN_ROOT}"
