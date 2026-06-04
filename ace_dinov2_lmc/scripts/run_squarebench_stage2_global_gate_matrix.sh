#!/usr/bin/env bash
set -uo pipefail

# Stage2-only matrix for the SquareBench negative-transfer case.
# It reuses an existing ACE-FCN memory and Stage1 local checkpoint, and runs only on GPUs 0/1.
# Run from /home/xwh/project/ace_depth:
#   bash ace_dinov2_lmc/scripts/run_squarebench_stage2_global_gate_matrix.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

CONDA_ENV="${CONDA_ENV:-mapanything}"
PYTHON_BIN="${PYTHON_BIN:-python}"
WAYSPOTS_ROOT="${WAYSPOTS_ROOT:-/data/xwh/Wayspots}"
SUITE_ROOT="${SUITE_ROOT:-/data/xwh/ace_dinov2_lmc/04_evaluation/wayspots_ace_fcn_lmc_suite/20260530_181745}"
SCENE="${SCENE:-wayspots_squarebench}"
MATRIX_SUBDIR="${MATRIX_SUBDIR:-stage2_global_gate_matrix_squarebench}"
MEMORY_DIRNAME="${MEMORY_DIRNAME:-memory}"
STAGE1_SUBDIR="${STAGE1_SUBDIR:-stage1_local_ace_memory_it12}"
GPUS_STR="${GPUS_STR:-0 1}"
VARIANTS_STR="${VARIANTS_STR:-glace_concat_gate_unified}"
CONSISTENCY_LOSS="${CONSISTENCY_LOSS:-smooth_l1}"
CONSISTENCY_SAMPLE_LIMIT="${CONSISTENCY_SAMPLE_LIMIT:-0}"
DRY_RUN="${DRY_RUN:-false}"
SKIP_EXISTING="${SKIP_EXISTING:-true}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-true}"

ACE_ENCODER_PATH="${ACE_ENCODER_PATH:-${ROOT_DIR}/ace_encoder_pretrained.pt}"
GLACE_FEAT_NAME="${GLACE_FEAT_NAME:-features.npy}"
LMC_ITERATIONS="${LMC_ITERATIONS:-12}"
NUM_LATENT_TOKENS="${NUM_LATENT_TOKENS:-64}"
IMAGE_RESOLUTION="${IMAGE_RESOLUTION:-512}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
TRAINING_BUFFER_SIZE="${TRAINING_BUFFER_SIZE:-2800000}"
BUFFER_SIZE_FINAL="${BUFFER_SIZE_FINAL:-7600000}"
SAMPLES_PER_IMAGE="${SAMPLES_PER_IMAGE:-512}"
POST_TRAIN_HYPOTHESES="${POST_TRAIN_HYPOTHESES:-256}"
POST_TRAIN_EVAL_SEEDS=(${POST_TRAIN_EVAL_SEEDS:-1305 2026 4242})
RANDOM_GLOBAL_SEED="${RANDOM_GLOBAL_SEED:-20260531}"

for gpu in ${GPUS_STR}; do
  if [[ "${gpu}" != "0" && "${gpu}" != "1" ]]; then
    echo "ERROR: this matrix is constrained to GPUs 0/1 only, got GPU '${gpu}'." >&2
    exit 2
  fi
done

SCENE_ROOT="${WAYSPOTS_ROOT}/${SCENE}"
MEMORY_PATH="${SUITE_ROOT}/${MEMORY_DIRNAME}/${SCENE}/memory_ace_fcn_sparse_sp_r4.pt"
AUX_DEPTH_DIR="${SCENE_ROOT}/train/sparse_depth"
RUN_ROOT="${SUITE_ROOT}/${MATRIX_SUBDIR}"
STATUS_FILE="${RUN_ROOT}/status.tsv"
mkdir -p "${RUN_ROOT}"
if [[ ! -f "${STATUS_FILE}" ]]; then
  printf "timestamp\tscene\tvariant\tstatus\texit_code\tgpu\tlog\n" > "${STATUS_FILE}"
fi

if [[ ! -s "${MEMORY_PATH}" ]]; then
  echo "ERROR: missing memory: ${MEMORY_PATH}" >&2
  exit 2
fi

if [[ -n "${STAGE1_CKPT:-}" ]]; then
  STAGE1_CHECKPOINT="${STAGE1_CKPT}"
else
  STAGE1_CHECKPOINT="$(find "${SUITE_ROOT}/${STAGE1_SUBDIR}" -type f -name "best_K${NUM_LATENT_TOKENS}_it${LMC_ITERATIONS}_ace_fcn_local_stage1.pt" -path "*${SCENE}*" 2>/dev/null | sort | tail -n 1 || true)"
fi
if [[ -z "${STAGE1_CHECKPOINT}" || ! -s "${STAGE1_CHECKPOINT}" ]]; then
  echo "ERROR: missing Stage1 checkpoint. Set STAGE1_CKPT or check ${SUITE_ROOT}/${STAGE1_SUBDIR}." >&2
  exit 2
fi

print_cmd() {
  printf "%q " "$@"
  printf "\n"
}

log_status() {
  local variant="$1" status="$2" exit_code="$3" gpu="$4" log_file="$5"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$(date +%Y-%m-%dT%H:%M:%S)" "${SCENE}" "${variant}" "${status}" "${exit_code}" "${gpu}" "${log_file}" >> "${STATUS_FILE}"
}

variant_config() {
  local variant="$1"
  case "${variant}" in
    zero_g1) echo "glace_concat zero 1.0 False 0.0 0.0 0.0 0 0.0 0.25 100.0 0 0.0 1.0 0.0" ;;
    random_g1) echo "glace_concat random 1.0 False 0.0 0.0 0.0 0 0.0 0.25 100.0 0 0.0 1.0 0.0" ;;
    glace_g0_learn) echo "glace_concat glace 0.0 True 0.0 0.0 0.0 0 0.0 0.25 100.0 0 0.0 1.0 0.0" ;;
    glace_g001_learn) echo "glace_concat glace 0.01 True 0.0 0.0 0.0 0 0.0 0.25 100.0 0 0.0 1.0 0.0" ;;
    glace_g001_max001_learn) echo "glace_concat glace 0.001 True 0.01 0.0 0.0 0 0.0 0.25 100.0 0 0.0 1.0 0.0" ;;
    glace_g001_max01_learn) echo "glace_concat glace 0.01 True 0.1 0.0 0.0 0 0.0 0.25 100.0 0 0.0 1.0 0.0" ;;
    glace_concat_gate_unified) echo "glace_concat glace 0.01 True 0.1 0.001 0.0 0 0.0 0.25 100.0 0 0.0 1.0 0.0" ;;
    glace_g001_max01_cons01) echo "glace_concat glace 0.01 True 0.1 0.0 0.1 100 0.0 0.25 100.0 0 0.0 1.0 0.0" ;;
    glace_g001_max01_cons1) echo "glace_concat glace 0.01 True 0.1 0.0 1.0 100 0.0 0.25 100.0 0 0.0 1.0 0.0" ;;
    glace_g001_max001_cons01) echo "glace_concat glace 0.001 True 0.01 0.0 0.1 100 0.0 0.25 100.0 0 0.0 1.0 0.0" ;;
    glace_g001_max01_guard01) echo "glace_concat glace 0.01 True 0.1 0.0 0.0 0 0.1 0.25 100.0 100 0.0 1.0 0.0" ;;
    glace_g001_max01_guard1) echo "glace_concat glace 0.01 True 0.1 0.0 0.0 0 1.0 0.25 100.0 100 0.0 1.0 0.0" ;;
    glace_g001_max001_guard01) echo "glace_concat glace 0.001 True 0.01 0.0 0.0 0 0.1 0.25 100.0 100 0.0 1.0 0.0" ;;
    glace_g01_learn) echo "glace_concat glace 0.1 True 0.0 0.0 0.0 0 0.0 0.25 100.0 0 0.0 1.0 0.0" ;;
    glace_g1_legacy) echo "glace_concat glace 1.0 False 0.0 0.0 0.0 0 0.0 0.25 100.0 0 0.0 1.0 0.0" ;;
    glace_residual_identity) echo "glace_residual glace 0.0 False 0.0 0.0 0.0 0 0.0 0.25 100.0 0 0.0 0.0 0.0" ;;
    glace_residual_unified) echo "glace_residual glace 0.001 False 0.1 0.0 0.0 0 1.0 0.25 100.0 100 0.01 0.5 10.0" ;;
    glace_film_unified) echo "glace_film glace 1.0 False 0.0 0.0 0.0 0 0.0 0.25 100.0 0 0.0 0.0 0.0" ;;
    zero_film) echo "glace_film zero 1.0 False 0.0 0.0 0.0 0 0.0 0.25 100.0 0 0.0 0.0 0.0" ;;
    random_film) echo "glace_film random 1.0 False 0.0 0.0 0.0 0 0.0 0.25 100.0 0 0.0 0.0 0.0" ;;
    *) echo "ERROR unknown variant ${variant}" >&2; return 2 ;;
  esac
}

run_variant() {
  local variant="$1" gpu="$2"
  local cfg head_mode mode gate learnable gate_max gate_l1 cons_weight cons_warmup guard_weight guard_margin guard_max guard_warmup residual_gate_l1 residual_delta_max_m residual_bad_gate_weight
  cfg="$(variant_config "${variant}")" || return 2
  read -r head_mode mode gate learnable gate_max gate_l1 cons_weight cons_warmup guard_weight guard_margin guard_max guard_warmup residual_gate_l1 residual_delta_max_m residual_bad_gate_weight <<< "${cfg}"
  local variant_root="${RUN_ROOT}/${variant}"
  local log_file="${variant_root}/train.log"
  local output_suffix="ace_fcn_glace_global_stage2_${variant}.pt"
  mkdir -p "${variant_root}"

  local existing
  existing="$(find "${variant_root}" -type f -name "best_K${NUM_LATENT_TOKENS}_it${LMC_ITERATIONS}_${output_suffix}" 2>/dev/null | sort | tail -n 1 || true)"
  if [[ -n "${existing}" && "${SKIP_EXISTING}" == "true" ]]; then
    log_status "${variant}" "skipped_existing" 0 "cuda:${gpu}" "${log_file}"
    return 0
  fi

  local cmd=(
    conda run --no-capture-output -n "${CONDA_ENV}" "${PYTHON_BIN}" ace_dinov2_lmc/train_ace_dinov2_lmc.py
    "${SCENE_ROOT}" "${output_suffix}"
    --model_backend ace_fcn_lmc
    --data_backend ace
    --use_lmc True
    --lmc_flow ace_g
    --memory_path "${MEMORY_PATH}"
    --use_scale_token False
    --ace_encoder_path "${ACE_ENCODER_PATH}"
    --ace_lmc_global_head_mode "${head_mode}"
    --ace_lmc_local_checkpoint_path "${STAGE1_CHECKPOINT}"
    --ace_lmc_freeze_local_stack True
    --ace_lmc_global_feature_mode "${mode}"
    --ace_lmc_global_gate_init "${gate}"
    --ace_lmc_global_gate_learnable "${learnable}"
    --ace_lmc_global_gate_max "${gate_max}"
    --ace_lmc_global_gate_l1_weight "${gate_l1}"
    --ace_lmc_random_global_seed "${RANDOM_GLOBAL_SEED}"
    --ace_lmc_global_residual_gate_l1_weight "${residual_gate_l1}"
    --ace_lmc_global_residual_delta_max_m "${residual_delta_max_m}"
    --ace_lmc_global_residual_bad_gate_weight "${residual_bad_gate_weight}"
    --ace_lmc_stage2_consistency_weight "${cons_weight}"
    --ace_lmc_stage2_consistency_loss "${CONSISTENCY_LOSS}"
    --ace_lmc_stage2_consistency_warmup_steps "${cons_warmup}"
    --ace_lmc_stage2_consistency_sample_limit "${CONSISTENCY_SAMPLE_LIMIT}"
    --ace_lmc_stage2_guard_weight "${guard_weight}"
    --ace_lmc_stage2_guard_margin_px "${guard_margin}"
    --ace_lmc_stage2_guard_max_px "${guard_max}"
    --ace_lmc_stage2_guard_warmup_steps "${guard_warmup}"
    --glace_feat_name "${GLACE_FEAT_NAME}"
    --device "cuda:${gpu}"
    --post_train_eval_device "cuda:${gpu}"
    --experiment_root "${SUITE_ROOT}"
    --experiment_subdir "${MATRIX_SUBDIR}/${variant}"
    --lmc_iterations "${LMC_ITERATIONS}"
    --num_latent_tokens "${NUM_LATENT_TOKENS}"
    --ace_g_fusion_in_s2 True
    --ace_g_cross_iter_eval True
    --s1_use_buffer True
    --s1_loss_mode sample_per_image
    --s1_buffer_refill_mode full
    --image_resolution "${IMAGE_RESOLUTION}"
    --batch_size "${BATCH_SIZE}"
    --training_buffer_size "${TRAINING_BUFFER_SIZE}"
    --buffer_size_final "${BUFFER_SIZE_FINAL}"
    --buffer_on_cpu True
    --buffer_on_cpu_final True
    --samples_per_image "${SAMPLES_PER_IMAGE}"
    --buffer_sample_valid_coords True
    --buffer_valid_coord_sample_ratio 1.0
    --buffer_valid_coord_neighbor_radius 1
    --buffer_valid_coord_neighbor_mode cross
    --c1_aux_depth_root "${AUX_DEPTH_DIR}"
    --c1_aux_depth_kind sparse_depth
    --post_train_eval_seeds "${POST_TRAIN_EVAL_SEEDS[@]}"
    --post_train_hypotheses "${POST_TRAIN_HYPOTHESES}"
  )

  {
    printf "[%s] %s variant=%s gpu=%s mode=%s gate=%s gate_max=%s gate_l1=%s learnable=%s cons=%s cons_warmup=%s guard=%s guard_margin=%s guard_warmup=%s\n" "$(date)" "${SCENE}" "${variant}" "${gpu}" "${mode}" "${gate}" "${gate_max}" "${gate_l1}" "${learnable}" "${cons_weight}" "${cons_warmup}" "${guard_weight}" "${guard_margin}" "${guard_warmup}"
    printf "Stage1: %s\n" "${STAGE1_CHECKPOINT}"
    print_cmd "${cmd[@]}"
  } | tee -a "${log_file}"

  if [[ "${DRY_RUN}" == "true" ]]; then
    log_status "${variant}" "dry_run" 0 "cuda:${gpu}" "${log_file}"
    return 0
  fi

  "${cmd[@]}" 2>&1 | tee -a "${log_file}"
  local exit_code=${PIPESTATUS[0]}
  if [[ ${exit_code} -eq 0 ]]; then
    log_status "${variant}" "ok" 0 "cuda:${gpu}" "${log_file}"
  else
    log_status "${variant}" "failed" "${exit_code}" "cuda:${gpu}" "${log_file}"
    if [[ "${CONTINUE_ON_ERROR}" != "true" ]]; then
      exit "${exit_code}"
    fi
  fi
  return "${exit_code}"
}

GPUS_ARR=(${GPUS_STR})
VARIANTS_ARR=(${VARIANTS_STR})
printf "Run root : %s\n" "${RUN_ROOT}"
printf "Scene    : %s\n" "${SCENE}"
printf "Memory   : %s\n" "${MEMORY_PATH}"
printf "Stage1   : %s\n" "${STAGE1_CHECKPOINT}"
printf "Variants : %s\n" "${VARIANTS_STR}"
printf "GPUs     : %s\n" "${GPUS_STR}"
printf "Dry run  : %s\n" "${DRY_RUN}"

pids=()
for i in "${!VARIANTS_ARR[@]}"; do
  gpu="${GPUS_ARR[$(( i % ${#GPUS_ARR[@]} ))]}"
  run_variant "${VARIANTS_ARR[$i]}" "${gpu}" &
  pids+=("$!")
  if (( ${#pids[@]} >= ${#GPUS_ARR[@]} )); then
    for pid in "${pids[@]}"; do wait "${pid}"; done
    pids=()
  fi
done
for pid in "${pids[@]}"; do wait "${pid}"; done

printf "Done. Matrix status: %s\n" "${STATUS_FILE}"
