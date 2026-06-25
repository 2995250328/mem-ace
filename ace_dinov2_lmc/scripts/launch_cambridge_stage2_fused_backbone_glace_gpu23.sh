#!/usr/bin/env bash
set -euo pipefail

# Cambridge ACE-FCN-LMC stage2: frozen Stage1 fused backbone + GLACE-style concat head.
# Reuses existing Stage1 checkpoints and ACE-FCN memory; does not rerun memory/stage1.
# Run from /home/xwh/project/ace_depth, preferably inside tmux.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

CONDA_ENV="${CONDA_ENV:-mapanything}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CAMBRIDGE_ROOT="${CAMBRIDGE_ROOT:-/data/xwh/Cambridge}"
SOURCE_RUN_ROOT="${SOURCE_RUN_ROOT:-/home/xwh/project/ace_depth/ace_dinov2_lmc/04_evaluation/cambridge_single_stage12_checked_20260624_gpu23}"
RUN_ROOT="${RUN_ROOT:-/data/xwh/ace_dinov2_lmc/04_evaluation/cambridge_stage2_fused_backbone_glace_20260624_gpu23}"
SCENES="${SCENES:-Cambridge_GreatCourt Cambridge_ShopFacade}"
GPU_0="${GPU_0:-2}"
GPU_1="${GPU_1:-3}"
DRY_RUN="${DRY_RUN:-false}"
SKIP_EXISTING="${SKIP_EXISTING:-true}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-false}"

ACE_ENCODER_PATH="${ACE_ENCODER_PATH:-${ROOT_DIR}/ace_encoder_pretrained.pt}"
GLACE_ROOT="${GLACE_ROOT:-/home/xwh/project/glace}"
GLACE_FEAT_NAME="${GLACE_FEAT_NAME:-features.npy}"
SOURCE_MEMORY_DIRNAME="${SOURCE_MEMORY_DIRNAME:-memory}"
SOURCE_STAGE1_SUBDIR="${SOURCE_STAGE1_SUBDIR:-stage1_local_ace_memory_it12}"
STAGE2_SUBDIR="${STAGE2_SUBDIR:-stage2_fused_backbone_glace_it1}"
STATUS_FILE="${RUN_ROOT}/status.tsv"

LMC_ITERATIONS="${LMC_ITERATIONS:-1}"
STAGE1_LMC_ITERATIONS="${STAGE1_LMC_ITERATIONS:-12}"
NUM_LATENT_TOKENS="${NUM_LATENT_TOKENS:-64}"
LMC_FUSION_REFINEMENT_MODE="${LMC_FUSION_REFINEMENT_MODE:-single}"
LMC_FUSION_CASCADE_LAYERS="${LMC_FUSION_CASCADE_LAYERS:-4}"
LMC_FUSION_ASSEMBLY_GAMMA_INIT="${LMC_FUSION_ASSEMBLY_GAMMA_INIT:-0.0}"
S1_LOSS_STEP_MODE="${S1_LOSS_STEP_MODE:-fixed_zero}"
S1_EARLY_STOP="${S1_EARLY_STOP:-False}"
LMC_LOG_RUNTIME_STATS="${LMC_LOG_RUNTIME_STATS:-True}"
LMC_RUNTIME_STATS_INTERVAL="${LMC_RUNTIME_STATS_INTERVAL:-100}"
LMC_RUNTIME_STATS_MAX_PIXELS="${LMC_RUNTIME_STATS_MAX_PIXELS:-4096}"
NUM_DATA_LOADER_WORKERS="${NUM_DATA_LOADER_WORKERS:-12}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-6}"
EVAL_DETERMINISTIC="${EVAL_DETERMINISTIC:-True}"
EVAL_DSACSTAR_SEED="${EVAL_DSACSTAR_SEED:-1305}"
EVAL_DSACSTAR_SEED_PER_FRAME="${EVAL_DSACSTAR_SEED_PER_FRAME:-True}"
ITERATION_EVAL_SEED="${ITERATION_EVAL_SEED:-1305}"
ITERATION_EVAL_HYPOTHESES="${ITERATION_EVAL_HYPOTHESES:-256}"
POST_TRAIN_HYPOTHESES="${POST_TRAIN_HYPOTHESES:-256}"
POST_TRAIN_EVAL_SEEDS=(${POST_TRAIN_EVAL_SEEDS:-1305 2026 4242})

IMAGE_RESOLUTION="${IMAGE_RESOLUTION:-518}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
TRAINING_BUFFER_SIZE="${TRAINING_BUFFER_SIZE:-2800000}"
BUFFER_SIZE_FINAL="${BUFFER_SIZE_FINAL:-7600000}"
SAMPLES_PER_IMAGE="${SAMPLES_PER_IMAGE:-512}"
NUM_HEAD_BLOCKS="${NUM_HEAD_BLOCKS:-2}"
GLACE_HEAD_CHANNELS="${GLACE_HEAD_CHANNELS:-768}"
GLACE_MLP_RATIO="${GLACE_MLP_RATIO:-1.0}"
ACE_LMC_GLOBAL_NOISE_STD="${ACE_LMC_GLOBAL_NOISE_STD:-0.1}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OMP_DYNAMIC="${OMP_DYNAMIC:-FALSE}"

mkdir -p "${RUN_ROOT}/logs"
if [[ ! -f "${STATUS_FILE}" ]]; then
  printf "timestamp\tscene\tmethod\tstage\tstatus\texit_code\tdevice\tlog\n" > "${STATUS_FILE}"
fi

print_cmd() {
  printf "%q " "$@"
  printf "\n"
}

log_status() {
  local scene="$1" method="$2" stage="$3" status="$4" exit_code="$5" device="$6" log_file="$7"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$(date +%Y-%m-%dT%H:%M:%S)" "$scene" "$method" "$stage" "$status" "$exit_code" "$device" "$log_file" >> "${STATUS_FILE}"
}

run_logged() {
  local scene="$1" method="$2" stage="$3" device="$4" log_file="$5"
  shift 5
  mkdir -p "$(dirname "${log_file}")"
  {
    printf "[%s] %s/%s/%s device=%s\n" "$(date)" "$scene" "$method" "$stage" "$device"
    print_cmd "$@"
  } | tee -a "${log_file}"

  if [[ "${DRY_RUN}" == "true" ]]; then
    log_status "$scene" "$method" "$stage" "dry_run" 0 "$device" "$log_file"
    return 0
  fi

  "$@" 2>&1 | tee -a "${log_file}"
  local exit_code=${PIPESTATUS[0]}
  if [[ ${exit_code} -eq 0 ]]; then
    log_status "$scene" "$method" "$stage" "ok" 0 "$device" "$log_file"
  else
    log_status "$scene" "$method" "$stage" "failed" "${exit_code}" "$device" "$log_file"
    if [[ "${CONTINUE_ON_ERROR}" != "true" ]]; then
      exit "${exit_code}"
    fi
  fi
  return "${exit_code}"
}

find_stage1_ckpt() {
  local scene="$1"
  local pattern="best_K${NUM_LATENT_TOKENS}_it${STAGE1_LMC_ITERATIONS}_ace_fcn_local_stage1.pt"
  find "${SOURCE_RUN_ROOT}/${SOURCE_STAGE1_SUBDIR}" -type f -name "${pattern}" -path "*${scene}*" 2>/dev/null | sort | tail -n 1 || true
}

run_stage2_scene() {
  local scene="$1"
  local gpu="$2"
  local scene_root="${CAMBRIDGE_ROOT}/${scene}"
  local train_root="${scene_root}/train"
  local memory_path="${SOURCE_RUN_ROOT}/${SOURCE_MEMORY_DIRNAME}/${scene}/memory_ace_fcn_sparse_sp_r4.pt"
  local stage1_ckpt
  stage1_ckpt="$(find_stage1_ckpt "${scene}")"
  local log_file="${RUN_ROOT}/logs/train_${scene}.log"

  if [[ -z "${stage1_ckpt}" || ! -s "${stage1_ckpt}" ]]; then
    echo "ERROR: missing stage1 checkpoint for ${scene} under ${SOURCE_RUN_ROOT}/${SOURCE_STAGE1_SUBDIR}" | tee -a "${log_file}" >&2
    log_status "${scene}" "stage2_fused_backbone_glace" "preflight" "failed_missing_stage1" 2 "cuda:${gpu}" "${log_file}"
    return 2
  fi
  if [[ ! -s "${memory_path}" ]]; then
    echo "ERROR: missing memory for ${scene}: ${memory_path}" | tee -a "${log_file}" >&2
    log_status "${scene}" "stage2_fused_backbone_glace" "preflight" "failed_missing_memory" 2 "cuda:${gpu}" "${log_file}"
    return 2
  fi
  for split in train test; do
    if [[ ! -s "${scene_root}/${split}/${GLACE_FEAT_NAME}" ]]; then
      echo "ERROR: missing GLACE features: ${scene_root}/${split}/${GLACE_FEAT_NAME}" | tee -a "${log_file}" >&2
      log_status "${scene}" "stage2_fused_backbone_glace" "preflight" "failed_missing_glace_features" 2 "cuda:${gpu}" "${log_file}"
      return 2
    fi
  done

  local output_suffix="ace_fcn_fused_backbone_glace_stage2.pt"
  local checkpoint_pattern="best_K${NUM_LATENT_TOKENS}_it${LMC_ITERATIONS}_${output_suffix}"
  local existing_ckpt
  existing_ckpt=$(find "${RUN_ROOT}/${STAGE2_SUBDIR}" -type f -name "${checkpoint_pattern}" -path "*${scene}*" 2>/dev/null | sort | tail -n 1 || true)
  if [[ -n "${existing_ckpt}" && "${SKIP_EXISTING}" == "true" ]]; then
    log_status "${scene}" "stage2_fused_backbone_glace" "train" "skipped_existing" 0 "cuda:${gpu}" "${log_file}"
    return 0
  fi

  run_logged "${scene}" "stage2_fused_backbone_glace" "train" "cuda:${gpu}" "${log_file}" \
    conda run --no-capture-output -n "${CONDA_ENV}" "${PYTHON_BIN}" ace_dinov2_lmc/train_ace_dinov2_lmc.py \
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
      --ace_lmc_global_noise_std "${ACE_LMC_GLOBAL_NOISE_STD}" \
      --glace_root "${GLACE_ROOT}" \
      --glace_feat_name "${GLACE_FEAT_NAME}" \
      --glace_head_channels "${GLACE_HEAD_CHANNELS}" \
      --glace_mlp_ratio "${GLACE_MLP_RATIO}" \
      --num_head_blocks "${NUM_HEAD_BLOCKS}" \
      --best_metric median_error \
      --device "cuda:${gpu}" \
      --post_train_eval_device "cuda:${gpu}" \
      --experiment_root "${RUN_ROOT}" \
      --experiment_subdir "${STAGE2_SUBDIR}" \
      --lmc_iterations "${LMC_ITERATIONS}" \
      --num_latent_tokens "${NUM_LATENT_TOKENS}" \
      --lmc_fusion_refinement_mode "${LMC_FUSION_REFINEMENT_MODE}" \
      --lmc_fusion_cascade_layers "${LMC_FUSION_CASCADE_LAYERS}" \
      --lmc_fusion_assembly_gamma_init "${LMC_FUSION_ASSEMBLY_GAMMA_INIT}" \
      --s1_loss_step_mode "${S1_LOSS_STEP_MODE}" \
      --s1_early_stop "${S1_EARLY_STOP}" \
      --lmc_log_runtime_stats "${LMC_LOG_RUNTIME_STATS}" \
      --lmc_runtime_stats_interval "${LMC_RUNTIME_STATS_INTERVAL}" \
      --lmc_runtime_stats_max_pixels "${LMC_RUNTIME_STATS_MAX_PIXELS}" \
      --num_data_loader_workers "${NUM_DATA_LOADER_WORKERS}" \
      --eval_num_workers "${EVAL_NUM_WORKERS}" \
      --eval_deterministic "${EVAL_DETERMINISTIC}" \
      --eval_dsacstar_seed "${EVAL_DSACSTAR_SEED}" \
      --eval_dsacstar_seed_per_frame "${EVAL_DSACSTAR_SEED_PER_FRAME}" \
      --iteration_eval_seed "${ITERATION_EVAL_SEED}" \
      --iteration_eval_hypotheses "${ITERATION_EVAL_HYPOTHESES}" \
      --ace_g_fusion_in_s2 False \
      --ace_g_cross_iter_eval False \
      --s1_use_buffer True \
      --s1_loss_mode sample_per_image \
      --s1_buffer_refill_mode full \
      --image_resolution "${IMAGE_RESOLUTION}" \
      --batch_size "${BATCH_SIZE}" \
      --training_buffer_size "${TRAINING_BUFFER_SIZE}" \
      --buffer_size_final "${BUFFER_SIZE_FINAL}" \
      --buffer_on_cpu True \
      --buffer_on_cpu_final True \
      --samples_per_image "${SAMPLES_PER_IMAGE}" \
      --buffer_sample_valid_coords True \
      --buffer_valid_coord_sample_ratio 1.0 \
      --buffer_valid_coord_neighbor_radius 1 \
      --buffer_valid_coord_neighbor_mode cross \
      --c1_aux_depth_root "${train_root}/sparse_depth" \
      --c1_aux_depth_kind sparse_depth \
      --post_train_eval_seeds "${POST_TRAIN_EVAL_SEEDS[@]}" \
      --post_train_hypotheses "${POST_TRAIN_HYPOTHESES}"
}

SCENES_ARR=(${SCENES})
if [[ ${#SCENES_ARR[@]} -ne 2 ]]; then
  echo "ERROR: this launch script expects exactly two scenes for GPU2/3, got: ${SCENES}" >&2
  exit 2
fi

printf "RUN_ROOT=%s\n" "${RUN_ROOT}"
printf "SOURCE_RUN_ROOT=%s\n" "${SOURCE_RUN_ROOT}"
printf "SCENES=%s\n" "${SCENES}"
printf "GPUS=%s,%s\n" "${GPU_0}" "${GPU_1}"
printf "LMC_ITERATIONS=%s BUFFER_SIZE_FINAL=%s head_channels=%s global_noise=%s\n" \
  "${LMC_ITERATIONS}" "${BUFFER_SIZE_FINAL}" "${GLACE_HEAD_CHANNELS}" "${ACE_LMC_GLOBAL_NOISE_STD}"
printf "DRY_RUN=%s\n" "${DRY_RUN}"

run_stage2_scene "${SCENES_ARR[0]}" "${GPU_0}" & pid_a=$!
run_stage2_scene "${SCENES_ARR[1]}" "${GPU_1}" & pid_b=$!
wait "${pid_a}" "${pid_b}"

printf "Done. Status: %s\n" "${STATUS_FILE}"
