#!/usr/bin/env bash
set -uo pipefail

# Wayspots multi-scene runner:
#   1. ACE baseline
#   2. GLACE baseline
#   3. ACE-FCN feature-space memory extraction
#   4. ACE-FCN Stage1 local LMC
#   5. ACE-FCN Stage2 + GLACE global concat
#
# Run from /home/xwh/project/ace_depth after activating the env:
#   conda activate mapanything
#   bash ace_dinov2_lmc/scripts/run_wayspots_ace_fcn_lmc_suite.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="${ROOT_DIR}/ace_dinov2_lmc/scripts"
cd "${ROOT_DIR}"

CONDA_ENV="${CONDA_ENV:-mapanything}"
PYTHON_BIN="${PYTHON_BIN:-python}"
WAYSPOTS_ROOT="${WAYSPOTS_ROOT:-/data/xwh/Wayspots}"
RUN_ROOT="${RUN_ROOT:-/data/xwh/ace_dinov2_lmc/04_evaluation/wayspots_ace_fcn_lmc_suite/$(date +%Y%m%d_%H%M%S)}"
SCENES="${SCENES:-wayspots_bears}"
EXCLUDE_SCENES="${EXCLUDE_SCENES:-}"
METHODS="${METHODS:-ace glace memory stage1 stage2}"
DRY_RUN="${DRY_RUN:-false}"
SKIP_EXISTING="${SKIP_EXISTING:-true}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-true}"

BASELINE_RUN_ROOT="${BASELINE_RUN_ROOT:-${RUN_ROOT}/baselines}"
MEMORY_DIRNAME="${MEMORY_DIRNAME:-memory}"
STAGE1_SUBDIR="${STAGE1_SUBDIR:-stage1_local_ace_memory_it12}"
STAGE2_SUBDIR="${STAGE2_SUBDIR:-stage2_glace_concat_it12}"
STATUS_FILE="${RUN_ROOT}/status.tsv"

GPU_0="${GPU_0:-0}"
GPU_1="${GPU_1:-1}"

ACE_ENCODER_PATH="${ACE_ENCODER_PATH:-${ROOT_DIR}/ace_encoder_pretrained.pt}"
GLACE_ROOT="${GLACE_ROOT:-/home/xwh/project/glace}"
GLACE_FEAT_NAME="${GLACE_FEAT_NAME:-features.npy}"

COORD_SOURCE="${COORD_SOURCE:-sparse_depth}"
DEPTH_REL_DIR="${DEPTH_REL_DIR:-train/sparse_depth_superpoint_strict_nms_r4}"
MEMORY_IMAGE_RESOLUTION="${MEMORY_IMAGE_RESOLUTION:-512}"
MEMORY_SAMPLES_PER_IMAGE="${MEMORY_SAMPLES_PER_IMAGE:-1024}"
MEMORY_VOXEL_SIZE="${MEMORY_VOXEL_SIZE:-0.05}"
MEMORY_MAX_POINTS="${MEMORY_MAX_POINTS:-300000}"
MEMORY_NUM_WORKERS="${MEMORY_NUM_WORKERS:-2}"

LMC_ITERATIONS="${LMC_ITERATIONS:-12}"
NUM_LATENT_TOKENS="${NUM_LATENT_TOKENS:-64}"
IMAGE_RESOLUTION="${IMAGE_RESOLUTION:-512}"
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

is_excluded() {
  local scene="$1"
  local exc
  for exc in ${EXCLUDE_SCENES}; do
    [[ "${scene}" == "${exc}" ]] && return 0
  done
  return 1
}

filtered_scenes() {
  local scene
  for scene in ${SCENES}; do
    is_excluded "${scene}" && continue
    printf "%s " "${scene}"
  done
}

ACTIVE_SCENES="$(filtered_scenes)"

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

summarize_suite() {
  local scene_args=()
  local scene
  # Summarize ALL scenes (including excluded) so old results appear in the table
  for scene in ${SCENES}; do
    scene_args+=("${scene}")
  done
  run_logged "all" "summary" "aggregate" "-" "${RUN_ROOT}/summary.log" \
    conda run --no-capture-output -n "${CONDA_ENV}" "${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_wayspots_ace_fcn_lmc_suite.py" \
      --suite-root "${RUN_ROOT}" \
      --baseline-root "${BASELINE_RUN_ROOT}" \
      --scenes "${scene_args[@]}" \
      --memory-dirname "${MEMORY_DIRNAME}" \
      --stage1-dirname "${STAGE1_SUBDIR}" \
      --stage2-dirname "${STAGE2_SUBDIR}"
}

run_memory() {
  local scene="$1"
  local scene_root="${WAYSPOTS_ROOT}/${scene}"
  local memory_dir="${RUN_ROOT}/${MEMORY_DIRNAME}/${scene}"
  local memory_path="${memory_dir}/memory_ace_fcn_sparse_sp_r4.pt"
  local depth_dir="${scene_root}/${DEPTH_REL_DIR}"
  mkdir -p "${memory_dir}"

  if maybe_skip_file "${memory_path}"; then
    log_status "${scene}" "memory" "extract" "skipped_existing" 0 "cuda:${GPU_MEMORY}" "${memory_dir}/extract.log"
    return 0
  fi

  run_logged "${scene}" "memory" "extract" "cuda:${GPU_MEMORY}" "${memory_dir}/extract.log" \
    conda run --no-capture-output -n "${CONDA_ENV}" "${PYTHON_BIN}" -m ace_dinov2_lmc.memory_extraction.extract_memory_ace_fcn \
      "${scene_root}" \
      "${memory_path}" \
      --ace_encoder_path "${ACE_ENCODER_PATH}" \
      --device "cuda:${GPU_MEMORY}" \
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
  local scene_root="${WAYSPOTS_ROOT}/${scene}"
  local memory_path="${RUN_ROOT}/${MEMORY_DIRNAME}/${scene}/memory_ace_fcn_sparse_sp_r4.pt"
  local aux_depth_dir="${scene_root}/train/sparse_depth"
  local output_suffix="ace_fcn_local_stage1.pt"
  local stage1_root="${RUN_ROOT}/${STAGE1_SUBDIR}"
  local checkpoint_pattern="best_K${NUM_LATENT_TOKENS}_it${LMC_ITERATIONS}_${output_suffix}"
  local existing_ckpt
  existing_ckpt=$(find "${stage1_root}" -type f -name "${checkpoint_pattern}" -path "*${scene}*" 2>/dev/null | sort | tail -n 1 || true)
  if [[ -n "${existing_ckpt}" && "${SKIP_EXISTING}" == "true" ]]; then
    log_status "${scene}" "stage1" "train" "skipped_existing" 0 "cuda:${GPU_STAGE1}" "${stage1_root}/skip_${scene}.log"
    return 0
  fi

  run_logged "${scene}" "stage1" "train" "cuda:${GPU_STAGE1}" "${stage1_root}/train_${scene}.log" \
    conda run --no-capture-output -n "${CONDA_ENV}" "${PYTHON_BIN}" ace_dinov2_lmc/train_ace_dinov2_lmc.py \
      "${scene_root}" "${output_suffix}" \
      --model_backend ace_fcn_lmc \
      --data_backend ace \
      --use_lmc True \
      --lmc_flow ace_g \
      --memory_path "${memory_path}" \
      --use_scale_token False \
      --ace_encoder_path "${ACE_ENCODER_PATH}" \
      --ace_lmc_global_head_mode none \
      --device "cuda:${GPU_STAGE1}" \
      --post_train_eval_device "cuda:${GPU_STAGE1}" \
      --experiment_root "${RUN_ROOT}" \
      --experiment_subdir "${STAGE1_SUBDIR}" \
      --lmc_iterations "${LMC_ITERATIONS}" \
      --num_latent_tokens "${NUM_LATENT_TOKENS}" \
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
      --c1_aux_depth_root "${aux_depth_dir}" \
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
  local scene_root="${WAYSPOTS_ROOT}/${scene}"
  local memory_path="${RUN_ROOT}/${MEMORY_DIRNAME}/${scene}/memory_ace_fcn_sparse_sp_r4.pt"
  local aux_depth_dir="${scene_root}/train/sparse_depth"
  local stage1_ckpt
  stage1_ckpt="$(find_stage1_ckpt "${scene}")"
  if [[ -z "${stage1_ckpt}" ]]; then
    log_status "${scene}" "stage2" "preflight" "failed_missing_stage1_ckpt" 2 "cuda:${GPU_STAGE2}" "${RUN_ROOT}/${STAGE2_SUBDIR}/preflight_${scene}.log"
    return 2
  fi
  local output_suffix="ace_fcn_glace_global_stage2.pt"
  local stage2_root="${RUN_ROOT}/${STAGE2_SUBDIR}"
  local checkpoint_pattern="best_K${NUM_LATENT_TOKENS}_it${LMC_ITERATIONS}_${output_suffix}"
  local existing_ckpt
  existing_ckpt=$(find "${stage2_root}" -type f -name "${checkpoint_pattern}" -path "*${scene}*" 2>/dev/null | sort | tail -n 1 || true)
  if [[ -n "${existing_ckpt}" && "${SKIP_EXISTING}" == "true" ]]; then
    log_status "${scene}" "stage2" "train" "skipped_existing" 0 "cuda:${GPU_STAGE2}" "${stage2_root}/skip_${scene}.log"
    return 0
  fi

  run_logged "${scene}" "stage2" "train" "cuda:${GPU_STAGE2}" "${stage2_root}/train_${scene}.log" \
    conda run --no-capture-output -n "${CONDA_ENV}" "${PYTHON_BIN}" ace_dinov2_lmc/train_ace_dinov2_lmc.py \
      "${scene_root}" "${output_suffix}" \
      --model_backend ace_fcn_lmc \
      --data_backend ace \
      --use_lmc True \
      --lmc_flow ace_g \
      --memory_path "${memory_path}" \
      --use_scale_token False \
      --ace_encoder_path "${ACE_ENCODER_PATH}" \
      --ace_lmc_global_head_mode glace_concat \
      --ace_lmc_local_checkpoint_path "${stage1_ckpt}" \
      --ace_lmc_freeze_local_stack True \
      --glace_feat_name "${GLACE_FEAT_NAME}" \
      --device "cuda:${GPU_STAGE2}" \
      --post_train_eval_device "cuda:${GPU_STAGE2}" \
      --experiment_root "${RUN_ROOT}" \
      --experiment_subdir "${STAGE2_SUBDIR}" \
      --lmc_iterations "${LMC_ITERATIONS}" \
      --num_latent_tokens "${NUM_LATENT_TOKENS}" \
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
      --c1_aux_depth_root "${aux_depth_dir}" \
      --c1_aux_depth_kind sparse_depth \
      --post_train_eval_seeds "${POST_TRAIN_EVAL_SEEDS[@]}" \
      --post_train_hypotheses "${POST_TRAIN_HYPOTHESES}"
}

run_ace_baseline() {
  run_logged "all" "ace" "baseline_run" "cuda:${GPU_0}" "${BASELINE_RUN_ROOT}/ace/run.log" \
    env \
      CONDA_ENV="${CONDA_ENV}" \
      WAYSPOTS_ROOT="${WAYSPOTS_ROOT}" \
      RUN_ROOT="${BASELINE_RUN_ROOT}" \
      SCENES="${ACTIVE_SCENES}" \
      METHODS="ace" \
      GPU_ID="${GPU_0}" \
      DEVICE="cuda:${GPU_0}" \
      EVAL_DEVICE="cuda:${GPU_0}" \
      bash "${SCRIPT_DIR}/run_wayspots_baselines_all.sh"
}

# ── Dual-GPU helpers ──
# Split ACTIVE_SCENES into two groups for parallel execution on both GPUs.
SCENES_ARR=(${ACTIVE_SCENES})
N_SCENES=${#SCENES_ARR[@]}
HALF=$(( (N_SCENES + 1) / 2 ))
GROUP_A=("${SCENES_ARR[@]:0:${HALF}}")
GROUP_B=("${SCENES_ARR[@]:${HALF}}")

run_baseline_on_gpu() {
  local gpu="$1" method="$2"; shift 2
  local scenes_str="$*"
  local extra_env=()
  if [[ "${method}" == "glace" ]]; then
    extra_env=(GLACE_ROOT="${GLACE_ROOT}" GLACE_RENDER_FLIPPED_PORTRAIT="true")
  fi
  run_logged "all" "${method}" "baseline_run" "cuda:${gpu}" "${BASELINE_RUN_ROOT}/${method}/run_gpu${gpu}.log" \
    env \
      CONDA_ENV="${CONDA_ENV}" \
      WAYSPOTS_ROOT="${WAYSPOTS_ROOT}" \
      RUN_ROOT="${BASELINE_RUN_ROOT}" \
      SCENES="${scenes_str}" \
      METHODS="${method}" \
      GPU_ID="${gpu}" \
      DEVICE="cuda:${gpu}" \
      EVAL_DEVICE="cuda:${gpu}" \
      "${extra_env[@]}" \
      bash "${SCRIPT_DIR}/run_wayspots_baselines_all.sh"
}

run_scenes_on_gpu() {
  local method="$1" gpu="$2"; shift 2
  for scene in "$@"; do
    case "${method}" in
      memory) GPU_MEMORY="${gpu}" run_memory "${scene}" ;;
      stage1) GPU_STAGE1="${gpu}" run_stage1 "${scene}" ;;
      stage2) GPU_STAGE2="${gpu}" run_stage2 "${scene}" ;;
    esac
  done
}

printf "Run root: %s\n" "${RUN_ROOT}"
printf "Scenes  : %s\n" "${ACTIVE_SCENES}"
[[ -n "${EXCLUDE_SCENES}" ]] && printf "Excluded: %s\n" "${EXCLUDE_SCENES}"
printf "Methods : %s\n" "${METHODS}"
printf "GPUs    : %s, %s (dual-GPU per phase)\n" "${GPU_0}" "${GPU_1}"
printf "Groups  : A(%d)=[%s] on GPU %s | B(%d)=[%s] on GPU %s\n" \
  "${#GROUP_A[@]}" "${GROUP_A[*]}" "${GPU_0}" "${#GROUP_B[@]}" "${GROUP_B[*]}" "${GPU_1}"
printf "Dry run : %s\n" "${DRY_RUN}"

# ── Phase 1: ACE baseline (single pass, usually all skipped) ──
if has_method ace; then
  run_ace_baseline
  summarize_suite || true
fi

# ── Phase 2: GLACE baseline (dual-GPU, scenes split) ──
if has_method glace; then
  printf "[dual] GLACE baseline: GPU %s ← [%s] | GPU %s ← [%s]\n" \
    "${GPU_0}" "${GROUP_A[*]}" "${GPU_1}" "${GROUP_B[*]}"
  run_baseline_on_gpu "${GPU_0}" glace "${GROUP_A[@]}" & pid_a=$!
  run_baseline_on_gpu "${GPU_1}" glace "${GROUP_B[@]}" & pid_b=$!
  wait "${pid_a}" "${pid_b}"
  summarize_suite || true
fi

# ── Phase 3: Memory extraction (dual-GPU) ──
if has_method memory; then
  printf "[dual] Memory extraction: GPU %s ← [%s] | GPU %s ← [%s]\n" \
    "${GPU_0}" "${GROUP_A[*]}" "${GPU_1}" "${GROUP_B[*]}"
  run_scenes_on_gpu memory "${GPU_0}" "${GROUP_A[@]}" & pid_a=$!
  run_scenes_on_gpu memory "${GPU_1}" "${GROUP_B[@]}" & pid_b=$!
  wait "${pid_a}" "${pid_b}"
  summarize_suite || true
fi

# ── Phase 4: Stage1 (dual-GPU) ──
if has_method stage1; then
  printf "[dual] Stage1: GPU %s ← [%s] | GPU %s ← [%s]\n" \
    "${GPU_0}" "${GROUP_A[*]}" "${GPU_1}" "${GROUP_B[*]}"
  run_scenes_on_gpu stage1 "${GPU_0}" "${GROUP_A[@]}" & pid_a=$!
  run_scenes_on_gpu stage1 "${GPU_1}" "${GROUP_B[@]}" & pid_b=$!
  wait "${pid_a}" "${pid_b}"
  summarize_suite || true
fi

# ── Phase 5: Stage2 (dual-GPU) ──
if has_method stage2; then
  printf "[dual] Stage2: GPU %s ← [%s] | GPU %s ← [%s]\n" \
    "${GPU_0}" "${GROUP_A[*]}" "${GPU_1}" "${GROUP_B[*]}"
  run_scenes_on_gpu stage2 "${GPU_0}" "${GROUP_A[@]}" & pid_a=$!
  run_scenes_on_gpu stage2 "${GPU_1}" "${GROUP_B[@]}" & pid_b=$!
  wait "${pid_a}" "${pid_b}"
  summarize_suite || true
fi

printf "Done. Summary: %s/summary.tsv and %s/summary.md\n" "${RUN_ROOT}" "${RUN_ROOT}"
