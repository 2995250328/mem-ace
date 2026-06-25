#!/usr/bin/env bash
set -uo pipefail

# Cambridge ACE-FCN-LMC single-fusion two-stage baseline.
#
# Phase 1: extract ACE-FCN feature memory from train/sparse_depth.
# Phase 2: Stage1 local ACE-FCN-LMC, no global head.
# Phase 3: Stage2 GLACE global concat, frozen Stage1 local stack.
#
# Run from /home/xwh/project/ace_depth:
#   RUN_ROOT=/data/xwh/ace_dinov2_lmc/04_evaluation/cambridge_single_stage12_YYYYMMDD_gpu23 \
#     bash ace_dinov2_lmc/scripts/launch_cambridge_single_stage12_gpu23.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

CONDA_ENV="${CONDA_ENV:-mapanything}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CAMBRIDGE_ROOT="${CAMBRIDGE_ROOT:-/data/xwh/Cambridge}"
RUN_ROOT="${RUN_ROOT:-/data/xwh/ace_dinov2_lmc/04_evaluation/cambridge_single_stage12_$(date +%Y%m%d_%H%M%S)_gpu23}"
SCENES="${SCENES:-Cambridge_GreatCourt Cambridge_KingsCollege Cambridge_OldHospital Cambridge_ShopFacade Cambridge_StMarysChurch}"
METHODS="${METHODS:-memory stage1 stage2}"
DRY_RUN="${DRY_RUN:-false}"
SKIP_EXISTING="${SKIP_EXISTING:-true}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-true}"

GPU_0="${GPU_0:-2}"
GPU_1="${GPU_1:-3}"

ACE_ENCODER_PATH="${ACE_ENCODER_PATH:-${ROOT_DIR}/ace_encoder_pretrained.pt}"
GLACE_FEAT_NAME="${GLACE_FEAT_NAME:-features.npy}"

MEMORY_DIRNAME="${MEMORY_DIRNAME:-memory}"
STAGE1_SUBDIR="${STAGE1_SUBDIR:-stage1_local_ace_memory_it12}"
STAGE2_SUBDIR="${STAGE2_SUBDIR:-stage2_glace_concat_it12}"
STATUS_FILE="${RUN_ROOT}/status.tsv"

COORD_SOURCE="${COORD_SOURCE:-sparse_depth}"
DEPTH_REL_DIR="${DEPTH_REL_DIR:-train/sparse_depth}"
MEMORY_IMAGE_RESOLUTION="${MEMORY_IMAGE_RESOLUTION:-518}"
MEMORY_SAMPLES_PER_IMAGE="${MEMORY_SAMPLES_PER_IMAGE:-1024}"
MEMORY_VOXEL_SIZE="${MEMORY_VOXEL_SIZE:-0.05}"
MEMORY_MAX_POINTS="${MEMORY_MAX_POINTS:-300000}"
MEMORY_NUM_WORKERS="${MEMORY_NUM_WORKERS:-2}"

LMC_ITERATIONS="${LMC_ITERATIONS:-12}"
NUM_LATENT_TOKENS="${NUM_LATENT_TOKENS:-64}"
LMC_FUSION_REFINEMENT_MODE="${LMC_FUSION_REFINEMENT_MODE:-single}"
LMC_FUSION_CASCADE_LAYERS="${LMC_FUSION_CASCADE_LAYERS:-4}"
LMC_FUSION_ASSEMBLY_GAMMA_INIT="${LMC_FUSION_ASSEMBLY_GAMMA_INIT:-0.0}"
S1_LOSS_STEP_MODE="${S1_LOSS_STEP_MODE:-fixed_zero}"
S1_EARLY_STOP="${S1_EARLY_STOP:-False}"
LMC_LOG_RUNTIME_STATS="${LMC_LOG_RUNTIME_STATS:-False}"
LMC_RUNTIME_STATS_INTERVAL="${LMC_RUNTIME_STATS_INTERVAL:-100}"
LMC_RUNTIME_STATS_MAX_PIXELS="${LMC_RUNTIME_STATS_MAX_PIXELS:-4096}"
NUM_DATA_LOADER_WORKERS="${NUM_DATA_LOADER_WORKERS:-12}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-6}"
EVAL_DETERMINISTIC="${EVAL_DETERMINISTIC:-True}"
EVAL_DSACSTAR_SEED="${EVAL_DSACSTAR_SEED:-1305}"
EVAL_DSACSTAR_SEED_PER_FRAME="${EVAL_DSACSTAR_SEED_PER_FRAME:-True}"
ITERATION_EVAL_SEED="${ITERATION_EVAL_SEED:-1305}"
ITERATION_EVAL_HYPOTHESES="${ITERATION_EVAL_HYPOTHESES:-256}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
OMP_DYNAMIC="${OMP_DYNAMIC:-FALSE}"
export OMP_NUM_THREADS
export OMP_DYNAMIC

IMAGE_RESOLUTION="${IMAGE_RESOLUTION:-518}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
TRAINING_BUFFER_SIZE="${TRAINING_BUFFER_SIZE:-2800000}"
BUFFER_SIZE_FINAL="${BUFFER_SIZE_FINAL:-7600000}"
SAMPLES_PER_IMAGE="${SAMPLES_PER_IMAGE:-512}"
POST_TRAIN_HYPOTHESES="${POST_TRAIN_HYPOTHESES:-256}"
POST_TRAIN_EVAL_SEEDS=(${POST_TRAIN_EVAL_SEEDS:-1305 2026 4242})

mkdir -p "${RUN_ROOT}"
if [[ ! -f "${STATUS_FILE}" ]]; then
  printf "timestamp\tscene\tmethod\tstage\tstatus\texit_code\tdevice\tlog\n" > "${STATUS_FILE}"
fi

has_method() {
  local needle="$1"
  local method
  for method in ${METHODS}; do
    [[ "${method}" == "${needle}" ]] && return 0
  done
  return 1
}

log_status() {
  local scene="$1" method="$2" stage="$3" status="$4" exit_code="$5" device="$6" log_file="$7"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$(date +%Y-%m-%dT%H:%M:%S)" "$scene" "$method" "$stage" "$status" "$exit_code" "$device" "$log_file" >> "${STATUS_FILE}"
}

print_cmd() {
  printf "%q " "$@"
  printf "\n"
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

maybe_skip_file() {
  local file="$1"
  [[ "${SKIP_EXISTING}" == "true" && -s "${file}" ]]
}

run_memory() {
  local scene="$1"
  local gpu="$2"
  local scene_root="${CAMBRIDGE_ROOT}/${scene}"
  local memory_dir="${RUN_ROOT}/${MEMORY_DIRNAME}/${scene}"
  local memory_path="${memory_dir}/memory_ace_fcn_sparse_sp_r4.pt"
  local depth_dir="${scene_root}/${DEPTH_REL_DIR}"
  mkdir -p "${memory_dir}"

  if maybe_skip_file "${memory_path}"; then
    log_status "${scene}" "memory" "extract" "skipped_existing" 0 "cuda:${gpu}" "${memory_dir}/extract.log"
    return 0
  fi

  run_logged "${scene}" "memory" "extract" "cuda:${gpu}" "${memory_dir}/extract.log" \
    conda run --no-capture-output -n "${CONDA_ENV}" "${PYTHON_BIN}" -m ace_dinov2_lmc.memory_extraction.extract_memory_ace_fcn \
      "${scene_root}" \
      "${memory_path}" \
      --ace_encoder_path "${ACE_ENCODER_PATH}" \
      --device "cuda:${gpu}" \
      --image_resolution "${MEMORY_IMAGE_RESOLUTION}" \
      --coord_source "${COORD_SOURCE}" \
      --depth_dir "${depth_dir}" \
      --samples_per_image "${MEMORY_SAMPLES_PER_IMAGE}" \
      --voxel_size "${MEMORY_VOXEL_SIZE}" \
      --max_points "${MEMORY_MAX_POINTS}" \
      --num_workers "${MEMORY_NUM_WORKERS}" \
      --sanity_hard_fail True
}

run_stage1() {
  local scene="$1"
  local gpu="$2"
  local scene_root="${CAMBRIDGE_ROOT}/${scene}"
  local train_root="${scene_root}/train"
  local memory_path="${RUN_ROOT}/${MEMORY_DIRNAME}/${scene}/memory_ace_fcn_sparse_sp_r4.pt"
  local output_suffix="ace_fcn_local_stage1.pt"
  local stage1_root="${RUN_ROOT}/${STAGE1_SUBDIR}"
  local checkpoint_pattern="best_K${NUM_LATENT_TOKENS}_it${LMC_ITERATIONS}_${output_suffix}"
  local existing_ckpt
  existing_ckpt=$(find "${stage1_root}" -type f -name "${checkpoint_pattern}" -path "*${scene}*" 2>/dev/null | sort | tail -n 1 || true)
  if [[ -n "${existing_ckpt}" && "${SKIP_EXISTING}" == "true" ]]; then
    log_status "${scene}" "stage1" "train" "skipped_existing" 0 "cuda:${gpu}" "${stage1_root}/skip_${scene}.log"
    return 0
  fi

  run_logged "${scene}" "stage1" "train" "cuda:${gpu}" "${stage1_root}/train_${scene}.log" \
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
      --ace_lmc_global_head_mode none \
      --device "cuda:${gpu}" \
      --post_train_eval_device "cuda:${gpu}" \
      --experiment_root "${RUN_ROOT}" \
      --experiment_subdir "${STAGE1_SUBDIR}" \
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
      --ace_g_fusion_in_s2 True \
      --ace_g_cross_iter_eval True \
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

find_stage1_ckpt() {
  local scene="$1"
  local output_suffix="ace_fcn_local_stage1.pt"
  local checkpoint_pattern="best_K${NUM_LATENT_TOKENS}_it${LMC_ITERATIONS}_${output_suffix}"
  find "${RUN_ROOT}/${STAGE1_SUBDIR}" -type f -name "${checkpoint_pattern}" -path "*${scene}*" 2>/dev/null | sort | tail -n 1 || true
}

run_stage2() {
  local scene="$1"
  local gpu="$2"
  local scene_root="${CAMBRIDGE_ROOT}/${scene}"
  local train_root="${scene_root}/train"
  local memory_path="${RUN_ROOT}/${MEMORY_DIRNAME}/${scene}/memory_ace_fcn_sparse_sp_r4.pt"
  local stage1_ckpt
  stage1_ckpt="$(find_stage1_ckpt "${scene}")"
  if [[ -z "${stage1_ckpt}" ]]; then
    log_status "${scene}" "stage2" "preflight" "failed_missing_stage1_ckpt" 2 "cuda:${gpu}" "${RUN_ROOT}/${STAGE2_SUBDIR}/preflight_${scene}.log"
    return 2
  fi

  local output_suffix="ace_fcn_glace_global_stage2.pt"
  local stage2_root="${RUN_ROOT}/${STAGE2_SUBDIR}"
  local checkpoint_pattern="best_K${NUM_LATENT_TOKENS}_it${LMC_ITERATIONS}_${output_suffix}"
  local existing_ckpt
  existing_ckpt=$(find "${stage2_root}" -type f -name "${checkpoint_pattern}" -path "*${scene}*" 2>/dev/null | sort | tail -n 1 || true)
  if [[ -n "${existing_ckpt}" && "${SKIP_EXISTING}" == "true" ]]; then
    log_status "${scene}" "stage2" "train" "skipped_existing" 0 "cuda:${gpu}" "${stage2_root}/skip_${scene}.log"
    return 0
  fi

  run_logged "${scene}" "stage2" "train" "cuda:${gpu}" "${stage2_root}/train_${scene}.log" \
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
      --glace_feat_name "${GLACE_FEAT_NAME}" \
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
      --ace_g_fusion_in_s2 True \
      --ace_g_cross_iter_eval True \
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

run_scenes_on_gpu() {
  local method="$1" gpu="$2"
  shift 2
  local scene
  for scene in "$@"; do
    case "${method}" in
      memory) run_memory "${scene}" "${gpu}" ;;
      stage1) run_stage1 "${scene}" "${gpu}" ;;
      stage2) run_stage2 "${scene}" "${gpu}" ;;
      *) echo "ERROR: unknown method ${method}" >&2; return 2 ;;
    esac
  done
}

SCENES_ARR=(${SCENES})
N_SCENES=${#SCENES_ARR[@]}
HALF=$(( (N_SCENES + 1) / 2 ))
GROUP_A=("${SCENES_ARR[@]:0:${HALF}}")
GROUP_B=("${SCENES_ARR[@]:${HALF}}")

printf "Run root: %s\n" "${RUN_ROOT}"
printf "Scenes  : %s\n" "${SCENES}"
printf "Methods : %s\n" "${METHODS}"
printf "Fusion  : %s\n" "${LMC_FUSION_REFINEMENT_MODE}"
printf "GPUs    : %s, %s\n" "${GPU_0}" "${GPU_1}"
printf "Groups  : A(%d)=[%s] on GPU %s | B(%d)=[%s] on GPU %s\n" \
  "${#GROUP_A[@]}" "${GROUP_A[*]}" "${GPU_0}" "${#GROUP_B[@]}" "${GROUP_B[*]}" "${GPU_1}"
printf "Dry run : %s\n" "${DRY_RUN}"

if has_method memory; then
  printf "[dual] Memory extraction: GPU %s ← [%s] | GPU %s ← [%s]\n" \
    "${GPU_0}" "${GROUP_A[*]}" "${GPU_1}" "${GROUP_B[*]}"
  run_scenes_on_gpu memory "${GPU_0}" "${GROUP_A[@]}" & pid_a=$!
  run_scenes_on_gpu memory "${GPU_1}" "${GROUP_B[@]}" & pid_b=$!
  wait "${pid_a}" "${pid_b}"
fi

if has_method stage1; then
  printf "[dual] Stage1: GPU %s ← [%s] | GPU %s ← [%s]\n" \
    "${GPU_0}" "${GROUP_A[*]}" "${GPU_1}" "${GROUP_B[*]}"
  run_scenes_on_gpu stage1 "${GPU_0}" "${GROUP_A[@]}" & pid_a=$!
  run_scenes_on_gpu stage1 "${GPU_1}" "${GROUP_B[@]}" & pid_b=$!
  wait "${pid_a}" "${pid_b}"
fi

if has_method stage2; then
  printf "[dual] Stage2: GPU %s ← [%s] | GPU %s ← [%s]\n" \
    "${GPU_0}" "${GROUP_A[*]}" "${GPU_1}" "${GROUP_B[*]}"
  run_scenes_on_gpu stage2 "${GPU_0}" "${GROUP_A[@]}" & pid_a=$!
  run_scenes_on_gpu stage2 "${GPU_1}" "${GROUP_B[@]}" & pid_b=$!
  wait "${pid_a}" "${pid_b}"
fi

printf "Done. Status: %s\n" "${STATUS_FILE}"
