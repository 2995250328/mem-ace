#!/usr/bin/env bash
set -euo pipefail

# Indoor6 two-stage GLACE+LMC runner, mirroring the validated Wayspots ACE-FCN-LMC route:
#   1. ensure GLACE global features exist for train/test
#   2. extract ACE-FCN feature-space memory from COLMAP-derived pseudo depth
#   3. Stage1: ACE-FCN-LMC local, no GLACE global
#   4. Stage2: freeze Stage1 local stack, train GLACE global concat head
#
# This script is intentionally constrained to GPUs 0/1/2/3.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

CONDA_ENV="${CONDA_ENV:-mapanything}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_ROOT="${RUN_ROOT:-/data/xwh/ace_dinov2_lmc/04_evaluation/indoor6_ace_fcn_glace_lmc_all_$(date +%Y%m%d_%H%M%S)}"
ACE_ROOT="${ACE_ROOT:-/home/xwh/data/indoor6_ace}"
WAI_ROOT="${WAI_ROOT:-/home/xwh/data/mapanything-dataset/wai_data/indoor6}"
GLACE_ROOT="${GLACE_ROOT:-/home/xwh/project/glace}"
GLACE_CHECKPOINT="${GLACE_CHECKPOINT:-/home/xwh/project/glace/CVPR23_DeitS_Rerank.pth}"
GLACE_FEAT_NAME="${GLACE_FEAT_NAME:-features.npy}"
ACE_ENCODER_PATH="${ACE_ENCODER_PATH:-${ROOT_DIR}/ace_encoder_pretrained.pt}"

GPU_0="${GPU_0:-0}"
GPU_1="${GPU_1:-1}"
GPU_2="${GPU_2:-2}"
GPU_3="${GPU_3:-3}"
SCENES_GPU0_STR="${SCENES_GPU0_STR-scene1 scene2a}"
SCENES_GPU1_STR="${SCENES_GPU1_STR-scene3 scene4a}"
SCENES_GPU2_STR="${SCENES_GPU2_STR-scene5}"
SCENES_GPU3_STR="${SCENES_GPU3_STR-scene6}"
METHODS="${METHODS:-features memory stage1 stage2}"
DRY_RUN="${DRY_RUN:-false}"
SKIP_EXISTING="${SKIP_EXISTING:-true}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-false}"

FEATURE_BATCH_SIZE="${FEATURE_BATCH_SIZE:-256}"
FEATURE_NUM_WORKERS="${FEATURE_NUM_WORKERS:-4}"
AUX_DEPTH_KIND="${AUX_DEPTH_KIND:-colmap_depth}"
COORD_SOURCE="${COORD_SOURCE:-colmap_depth}"
MEMORY_DIRNAME="${MEMORY_DIRNAME:-memory}"
MEMORY_IMAGE_RESOLUTION="${MEMORY_IMAGE_RESOLUTION:-512}"
MEMORY_SAMPLES_PER_IMAGE="${MEMORY_SAMPLES_PER_IMAGE:-1024}"
MEMORY_VOXEL_SIZE="${MEMORY_VOXEL_SIZE:-0.05}"
MEMORY_MAX_POINTS="${MEMORY_MAX_POINTS:-300000}"
MEMORY_NUM_WORKERS="${MEMORY_NUM_WORKERS:-2}"
MEMORY_DEPTH_MIN="${MEMORY_DEPTH_MIN:-1e-6}"
MEMORY_DEPTH_MAX="${MEMORY_DEPTH_MAX:-1000.0}"
C1_AUX_DEPTH_MIN="${C1_AUX_DEPTH_MIN:-1e-6}"
C1_AUX_DEPTH_MAX="${C1_AUX_DEPTH_MAX:-1000.0}"

STAGE1_SUBDIR="${STAGE1_SUBDIR:-stage1_local_ace_memory_it12}"
STAGE2_SUBDIR="${STAGE2_SUBDIR:-stage2_glace_concat_it12}"
LMC_ITERATIONS="${LMC_ITERATIONS:-12}"
NUM_LATENT_TOKENS="${NUM_LATENT_TOKENS:-64}"
IMAGE_RESOLUTION="${IMAGE_RESOLUTION:-512}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
TRAINING_BUFFER_SIZE="${TRAINING_BUFFER_SIZE:-2800000}"
BUFFER_SIZE_FINAL="${BUFFER_SIZE_FINAL:-7600000}"
SAMPLES_PER_IMAGE="${SAMPLES_PER_IMAGE:-512}"
POST_TRAIN_HYPOTHESES="${POST_TRAIN_HYPOTHESES:-256}"
POST_TRAIN_EVAL_SEEDS=(${POST_TRAIN_EVAL_SEEDS:-1305 2026 4242})

# Stable defaults for ACE-FCN memory runs. The depth20 scene3 repair only fixes
# pseudo-depth outliers; leaving this route on legacy/raw/fixed_zero makes S1
# start in the known unstable regime for large Indoor6 scenes.
LMC_PROFILE="${LMC_PROFILE:-mapany_flow_v1}"
S1_LOSS_STEP_MODE="${S1_LOSS_STEP_MODE:-per_iter}"
PE_NORMALIZE_INPUT="${PE_NORMALIZE_INPUT:-True}"
LMC_COMPRESSOR_PE_SCALE_MODE="${LMC_COMPRESSOR_PE_SCALE_MODE:-scene_scale}"

if [[ "${GPU_0}" != "0" || "${GPU_1}" != "1" || "${GPU_2}" != "2" || "${GPU_3}" != "3" ]]; then
  echo "ERROR: this script is constrained to GPU_0=0, GPU_1=1, GPU_2=2, GPU_3=3." >&2
  echo "       Current: GPU_0=${GPU_0}, GPU_1=${GPU_1}, GPU_2=${GPU_2}, GPU_3=${GPU_3}" >&2
  exit 2
fi

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/${MEMORY_DIRNAME}"
STATUS_FILE="${RUN_ROOT}/status.tsv"
printf "timestamp\tscene\tmethod\tstage\tstatus\texit_code\tgpu\tlog\n" > "${STATUS_FILE}"

has_method() {
  local needle="$1" method
  for method in ${METHODS}; do
    [[ "${method}" == "${needle}" ]] && return 0
  done
  return 1
}

log_status() {
  local scene="$1" method="$2" stage="$3" status="$4" exit_code="$5" gpu="$6" log_file="$7"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$(date +%Y-%m-%dT%H:%M:%S)" "${scene}" "${method}" "${stage}" "${status}" "${exit_code}" "${gpu}" "${log_file}" >> "${STATUS_FILE}"
}

print_cmd() {
  printf "%q " "$@"
  printf "\n"
}

run_logged() {
  local scene="$1" method="$2" stage="$3" gpu="$4" log_file="$5"
  shift 5
  mkdir -p "$(dirname "${log_file}")"
  {
    printf "[%s] %s/%s/%s gpu=%s\n" "$(date)" "${scene}" "${method}" "${stage}" "${gpu}"
    print_cmd "$@"
  } > "${log_file}"

  if [[ "${DRY_RUN}" == "true" ]]; then
    log_status "${scene}" "${method}" "${stage}" "dry_run" 0 "${gpu}" "${log_file}"
    return 0
  fi

  set +e
  "$@" >> "${log_file}" 2>&1
  local exit_code=$?
  set -e
  if [[ "${exit_code}" -eq 0 ]]; then
    log_status "${scene}" "${method}" "${stage}" "ok" 0 "${gpu}" "${log_file}"
  else
    log_status "${scene}" "${method}" "${stage}" "failed" "${exit_code}" "${gpu}" "${log_file}"
  fi
  return "${exit_code}"
}

ensure_glace_features() {
  local scene="$1" gpu="$2"
  local scene_root="${ACE_ROOT}/${scene}"
  local train_feat="${scene_root}/train/${GLACE_FEAT_NAME}"
  local test_feat="${scene_root}/test/${GLACE_FEAT_NAME}"
  local log_file="${RUN_ROOT}/logs/${scene}_gpu${gpu}_features.log"

  if [[ "${SKIP_EXISTING}" == "true" && -s "${train_feat}" && -s "${test_feat}" ]]; then
    log_status "${scene}" "features" "extract" "skipped_existing" 0 "${gpu}" "${log_file}"
    return 0
  fi

  if [[ ! -d "${scene_root}/train/rgb" || ! -d "${scene_root}/test/rgb" ]]; then
    printf "ERROR: missing Indoor6 ACE train/test rgb dirs: %s\n" "${scene_root}" > "${log_file}"
    log_status "${scene}" "features" "preflight" "failed_bad_scene_root" 2 "${gpu}" "${log_file}"
    return 2
  fi

  run_logged "${scene}" "features" "extract" "${gpu}" "${log_file}" \
    env CUDA_VISIBLE_DEVICES="${gpu}" \
    conda run --no-capture-output -n "${CONDA_ENV}" "${PYTHON_BIN}" "${GLACE_ROOT}/datasets/extract_features.py" \
      "${scene_root}" \
      --checkpoint "${GLACE_CHECKPOINT}" \
      --batch_size "${FEATURE_BATCH_SIZE}" \
      --num_workers "${FEATURE_NUM_WORKERS}"
}

memory_path_for_scene() {
  local scene="$1"
  echo "${RUN_ROOT}/${MEMORY_DIRNAME}/${scene}/memory_ace_fcn_${AUX_DEPTH_KIND}_sp_r4.pt"
}

run_memory() {
  local scene="$1" gpu="$2"
  local scene_root="${ACE_ROOT}/${scene}"
  local depth_dir="${WAI_ROOT}/${scene}_train/${AUX_DEPTH_KIND}"
  local memory_path
  memory_path="$(memory_path_for_scene "${scene}")"
  local log_file="${RUN_ROOT}/${MEMORY_DIRNAME}/${scene}/extract.log"
  mkdir -p "$(dirname "${memory_path}")"

  if [[ "${SKIP_EXISTING}" == "true" && -s "${memory_path}" ]]; then
    log_status "${scene}" "memory" "extract" "skipped_existing" 0 "${gpu}" "${log_file}"
    return 0
  fi
  if [[ ! -d "${depth_dir}" ]]; then
    printf "ERROR: missing Indoor6 COLMAP depth dir: %s\n" "${depth_dir}" > "${log_file}"
    log_status "${scene}" "memory" "preflight" "failed_missing_depth" 2 "${gpu}" "${log_file}"
    return 2
  fi

  run_logged "${scene}" "memory" "extract" "${gpu}" "${log_file}" \
    conda run --no-capture-output -n "${CONDA_ENV}" "${PYTHON_BIN}" -m ace_dinov2_lmc.memory_extraction.extract_memory_ace_fcn \
      "${scene_root}" \
      "${memory_path}" \
      --ace_encoder_path "${ACE_ENCODER_PATH}" \
      --device "cuda:${gpu}" \
      --image_resolution "${MEMORY_IMAGE_RESOLUTION}" \
      --coord_source "${COORD_SOURCE}" \
      --depth_dir "${depth_dir}" \
      --depth_min "${MEMORY_DEPTH_MIN}" \
      --depth_max "${MEMORY_DEPTH_MAX}" \
      --samples_per_image "${MEMORY_SAMPLES_PER_IMAGE}" \
      --voxel_size "${MEMORY_VOXEL_SIZE}" \
      --max_points "${MEMORY_MAX_POINTS}" \
      --num_workers "${MEMORY_NUM_WORKERS}" \
      --sanity_hard_fail True
}

find_stage1_ckpt() {
  local scene="$1"
  find "${RUN_ROOT}/${STAGE1_SUBDIR}" -type f -name "best_K${NUM_LATENT_TOKENS}_it${LMC_ITERATIONS}_ace_fcn_local_stage1.pt" -path "*${scene}*" 2>/dev/null | sort | tail -n 1 || true
}

run_stage1() {
  local scene="$1" gpu="$2"
  local scene_root="${ACE_ROOT}/${scene}"
  local memory_path aux_depth_root log_file existing
  memory_path="$(memory_path_for_scene "${scene}")"
  aux_depth_root="${WAI_ROOT}/${scene}_train"
  log_file="${RUN_ROOT}/${STAGE1_SUBDIR}/train_${scene}.log"
  existing="$(find "${RUN_ROOT}/${STAGE1_SUBDIR}" -type f -name "best_K${NUM_LATENT_TOKENS}_it${LMC_ITERATIONS}_ace_fcn_local_stage1.pt" -path "*${scene}*" 2>/dev/null | sort | tail -n 1 || true)"
  if [[ "${SKIP_EXISTING}" == "true" && -n "${existing}" ]]; then
    log_status "${scene}" "stage1" "train" "skipped_existing" 0 "${gpu}" "${log_file}"
    return 0
  fi

  run_logged "${scene}" "stage1" "train" "${gpu}" "${log_file}" \
    conda run --no-capture-output -n "${CONDA_ENV}" "${PYTHON_BIN}" ace_dinov2_lmc/train_ace_dinov2_lmc.py \
      "${scene_root}" "ace_fcn_local_stage1.pt" \
      --model_backend ace_fcn_lmc \
      --data_backend ace \
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
      --lmc_profile "${LMC_PROFILE}" \
      --lmc_iterations "${LMC_ITERATIONS}" \
      --num_latent_tokens "${NUM_LATENT_TOKENS}" \
      --ace_g_fusion_in_s2 True \
      --ace_g_cross_iter_eval True \
      --s1_use_buffer True \
      --s1_loss_mode sample_per_image \
      --s1_loss_step_mode "${S1_LOSS_STEP_MODE}" \
      --s1_buffer_refill_mode full \
      --pe_normalize_input "${PE_NORMALIZE_INPUT}" \
      --lmc_compressor_pe_scale_mode "${LMC_COMPRESSOR_PE_SCALE_MODE}" \
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
      --c1_aux_depth_root "${aux_depth_root}" \
      --c1_aux_depth_kind "${AUX_DEPTH_KIND}" \
      --c1_aux_depth_min "${C1_AUX_DEPTH_MIN}" \
      --c1_aux_depth_max "${C1_AUX_DEPTH_MAX}" \
      --post_train_eval_seeds "${POST_TRAIN_EVAL_SEEDS[@]}" \
      --post_train_hypotheses "${POST_TRAIN_HYPOTHESES}" \
      ${EXTRA_TRAIN_ARGS:-}
}

run_stage2() {
  local scene="$1" gpu="$2"
  local scene_root="${ACE_ROOT}/${scene}"
  local memory_path aux_depth_root stage1_ckpt log_file existing
  memory_path="$(memory_path_for_scene "${scene}")"
  aux_depth_root="${WAI_ROOT}/${scene}_train"
  stage1_ckpt="$(find_stage1_ckpt "${scene}")"
  log_file="${RUN_ROOT}/${STAGE2_SUBDIR}/train_${scene}.log"
  mkdir -p "$(dirname "${log_file}")"
  if [[ -z "${stage1_ckpt}" || ! -s "${stage1_ckpt}" ]]; then
    if [[ "${DRY_RUN}" == "true" ]]; then
      stage1_ckpt="${RUN_ROOT}/${STAGE1_SUBDIR}/DRY_RUN_STAGE1_CKPT.pt"
    else
      printf "ERROR: missing Stage1 checkpoint for %s under %s/%s
" "${scene}" "${RUN_ROOT}" "${STAGE1_SUBDIR}" > "${log_file}"
      log_status "${scene}" "stage2" "preflight" "failed_missing_stage1" 2 "${gpu}" "${log_file}"
      return 2
    fi
  fi
  existing="$(find "${RUN_ROOT}/${STAGE2_SUBDIR}" -type f -name "best_K${NUM_LATENT_TOKENS}_it${LMC_ITERATIONS}_ace_fcn_glace_global_stage2.pt" -path "*${scene}*" 2>/dev/null | sort | tail -n 1 || true)"
  if [[ "${SKIP_EXISTING}" == "true" && -n "${existing}" ]]; then
    log_status "${scene}" "stage2" "train" "skipped_existing" 0 "${gpu}" "${log_file}"
    return 0
  fi

  run_logged "${scene}" "stage2" "train" "${gpu}" "${log_file}" \
    conda run --no-capture-output -n "${CONDA_ENV}" "${PYTHON_BIN}" ace_dinov2_lmc/train_ace_dinov2_lmc.py \
      "${scene_root}" "ace_fcn_glace_global_stage2.pt" \
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
      --device "cuda:${gpu}" \
      --post_train_eval_device "cuda:${gpu}" \
      --experiment_root "${RUN_ROOT}" \
      --experiment_subdir "${STAGE2_SUBDIR}" \
      --lmc_profile "${LMC_PROFILE}" \
      --lmc_iterations "${LMC_ITERATIONS}" \
      --num_latent_tokens "${NUM_LATENT_TOKENS}" \
      --ace_g_fusion_in_s2 True \
      --ace_g_cross_iter_eval True \
      --s1_use_buffer True \
      --s1_loss_mode sample_per_image \
      --s1_loss_step_mode "${S1_LOSS_STEP_MODE}" \
      --s1_buffer_refill_mode full \
      --pe_normalize_input "${PE_NORMALIZE_INPUT}" \
      --lmc_compressor_pe_scale_mode "${LMC_COMPRESSOR_PE_SCALE_MODE}" \
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
      --c1_aux_depth_root "${aux_depth_root}" \
      --c1_aux_depth_kind "${AUX_DEPTH_KIND}" \
      --c1_aux_depth_min "${C1_AUX_DEPTH_MIN}" \
      --c1_aux_depth_max "${C1_AUX_DEPTH_MAX}" \
      --post_train_eval_seeds "${POST_TRAIN_EVAL_SEEDS[@]}" \
      --post_train_hypotheses "${POST_TRAIN_HYPOTHESES}" \
      ${EXTRA_TRAIN_ARGS:-}
}

run_scene() {
  local scene="$1" gpu="$2"
  if has_method features; then
    echo "[features] ${scene} on cuda:${gpu}"
    ensure_glace_features "${scene}" "${gpu}" || return $?
  fi
  if has_method memory; then
    echo "[memory] ${scene} on cuda:${gpu}"
    run_memory "${scene}" "${gpu}" || return $?
  fi
  if has_method stage1; then
    echo "[stage1] ${scene} on cuda:${gpu}"
    run_stage1 "${scene}" "${gpu}" || return $?
  fi
  if has_method stage2; then
    echo "[stage2] ${scene} on cuda:${gpu}"
    run_stage2 "${scene}" "${gpu}" || return $?
  fi
  echo "[ok] ${scene} on cuda:${gpu}"
}

run_group() {
  local gpu="$1"
  shift
  local scene
  for scene in "$@"; do
    if ! run_scene "${scene}" "${gpu}"; then
      echo "[failed] ${scene} on cuda:${gpu}; see ${RUN_ROOT}/status.tsv" >&2
      if [[ "${CONTINUE_ON_ERROR}" != "true" ]]; then
        return 1
      fi
    fi
  done
}

read -r -a SCENES_GPU0 <<< "${SCENES_GPU0_STR}"
read -r -a SCENES_GPU1 <<< "${SCENES_GPU1_STR}"
read -r -a SCENES_GPU2 <<< "${SCENES_GPU2_STR}"
read -r -a SCENES_GPU3 <<< "${SCENES_GPU3_STR}"

printf "Run root : %s\n" "${RUN_ROOT}"
printf "GPU 0    : %s\n" "${SCENES_GPU0[*]}"
printf "GPU 1    : %s\n" "${SCENES_GPU1[*]}"
printf "GPU 2    : %s\n" "${SCENES_GPU2[*]}"
printf "GPU 3    : %s\n" "${SCENES_GPU3[*]}"
printf "Methods  : %s\n" "${METHODS}"
printf "Route    : ACE-FCN memory -> Stage1 local -> Stage2 GLACE concat\n"
printf "Depth    : %s from %s/<scene>_train/%s\n" "${AUX_DEPTH_KIND}" "${WAI_ROOT}" "${AUX_DEPTH_KIND}"
printf "DepthClip: memory=[%s,%s], aux=[%s,%s] meters\n" "${MEMORY_DEPTH_MIN}" "${MEMORY_DEPTH_MAX}" "${C1_AUX_DEPTH_MIN}" "${C1_AUX_DEPTH_MAX}"
printf "Stability: lmc_profile=%s, s1_loss_step_mode=%s, pe_normalize_input=%s, compressor_pe=%s\n" "${LMC_PROFILE}" "${S1_LOSS_STEP_MODE}" "${PE_NORMALIZE_INPUT}" "${LMC_COMPRESSOR_PE_SCALE_MODE}"
printf "Status   : %s\n" "${STATUS_FILE}"
printf "Dry run  : %s\n" "${DRY_RUN}"

run_group "${GPU_0}" "${SCENES_GPU0[@]}" &
pid0=$!
run_group "${GPU_1}" "${SCENES_GPU1[@]}" &
pid1=$!
run_group "${GPU_2}" "${SCENES_GPU2[@]}" &
pid2=$!
run_group "${GPU_3}" "${SCENES_GPU3[@]}" &
pid3=$!

set +e
wait "${pid0}"
code0=$?
wait "${pid1}"
code1=$?
wait "${pid2}"
code2=$?
wait "${pid3}"
code3=$?
set -e

if [[ "${code0}" -ne 0 || "${code1}" -ne 0 || "${code2}" -ne 0 || "${code3}" -ne 0 ]]; then
  echo "Done with failures. Status: ${STATUS_FILE}" >&2
  exit 1
fi

echo "Done. Status: ${STATUS_FILE}"
