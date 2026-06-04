#!/usr/bin/env bash
set -euo pipefail

# Run Indoor6 GLACE+LMC on all six scenes with exactly two GPUs.
#
# For each scene, this script first ensures GLACE global features exist, then
# extracts GLACE-encoder feature-space memory when no memory is found, and then
# runs the single-scene GLACE+LMC trainer. It does not run ACE/GLACE baselines.
#
# Example:
#   cd /home/xwh/project/ace_depth
#   RUN_ROOT=/data/xwh/ace_dinov2_lmc/04_evaluation/indoor6_glace_lmc_all \
#   bash ace_dinov2_lmc/scripts/run_indoor6_glace_lmc_all_dual_gpu.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

CONDA_ENV="${CONDA_ENV:-mapanything}"
RUN_ROOT="${RUN_ROOT:-/data/xwh/ace_dinov2_lmc/04_evaluation/indoor6_glace_lmc_all_$(date +%Y%m%d_%H%M%S)}"
GPU_0="${GPU_0:-0}"
GPU_1="${GPU_1:-1}"
SCENES_GPU0_STR="${SCENES_GPU0_STR-scene1 scene2a scene3}"
SCENES_GPU1_STR="${SCENES_GPU1_STR-scene4a scene5 scene6}"
ACE_ROOT="${ACE_ROOT:-/home/xwh/data/indoor6_ace}"
WAI_ROOT="${WAI_ROOT:-/home/xwh/data/mapanything-dataset/wai_data/indoor6}"
GLACE_ROOT="${GLACE_ROOT:-/home/xwh/project/glace}"
GLACE_ENCODER_PATH="${GLACE_ENCODER_PATH:-/home/xwh/project/glace/ace_encoder_pretrained.pt}"
GLACE_CHECKPOINT="${GLACE_CHECKPOINT:-/home/xwh/project/glace/CVPR23_DeitS_Rerank.pth}"
GLACE_FEAT_NAME="${GLACE_FEAT_NAME:-features.npy}"
DO_EXTRACT_FEATURES="${DO_EXTRACT_FEATURES:-true}"
SKIP_EXISTING_FEATURES="${SKIP_EXISTING_FEATURES:-true}"
FEATURE_BATCH_SIZE="${FEATURE_BATCH_SIZE:-256}"
FEATURE_NUM_WORKERS="${FEATURE_NUM_WORKERS:-4}"
MEMORY_MANIFEST="${MEMORY_MANIFEST:-}"
MEMORY_PATH_PREFIX="${MEMORY_PATH_PREFIX:-${RUN_ROOT}/memory}"
DO_EXTRACT_MEMORY="${DO_EXTRACT_MEMORY:-true}"
SKIP_EXISTING_MEMORY="${SKIP_EXISTING_MEMORY:-true}"
N_MEMORY="${N_MEMORY:-64}"
MEMORY_VOXEL_SIZE="${MEMORY_VOXEL_SIZE:-0.08}"
UNIMODAL_THRESHOLD="${UNIMODAL_THRESHOLD:-0.02}"
MEMORY_IMAGE_RESOLUTION="${MEMORY_IMAGE_RESOLUTION:-518}"
MEMORY_DEPTH_MIN="${MEMORY_DEPTH_MIN:-0.02}"
MEMORY_DEPTH_MAX="${MEMORY_DEPTH_MAX:-100.0}"
AUX_DEPTH_KIND="${AUX_DEPTH_KIND:-colmap_depth}"
DRY_RUN="${DRY_RUN:-false}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-false}"

if [[ "${GPU_0}" != "0" || "${GPU_1}" != "1" ]]; then
  echo "ERROR: this script is constrained to GPU_0=0 and GPU_1=1." >&2
  echo "       Current: GPU_0=${GPU_0}, GPU_1=${GPU_1}" >&2
  exit 2
fi

mkdir -p "${RUN_ROOT}/logs" "${MEMORY_PATH_PREFIX}"
STATUS_FILE="${RUN_ROOT}/status.tsv"
printf "timestamp\tscene\tstage\tstatus\texit_code\tgpu\tlog\tmemory_path\n" > "${STATUS_FILE}"

lookup_memory() {
  local scene="$1"

  if [[ -n "${MEMORY_PATH_PREFIX}" ]]; then
    local candidate
    for candidate in \
      "${MEMORY_PATH_PREFIX}/${scene}/memory_bse.pt" \
      "${MEMORY_PATH_PREFIX}/${scene}/memory.pt" \
      "${MEMORY_PATH_PREFIX}/memory_${scene}.pt" \
      "${MEMORY_PATH_PREFIX}/${scene}_memory_bse.pt" \
      "${RUN_ROOT}/memory/${scene}/memory_bse.pt"; do
      if [[ -s "${candidate}" ]]; then
        echo "${candidate}"
        return 0
      fi
    done
  fi

  if [[ -n "${MEMORY_MANIFEST}" && -s "${MEMORY_MANIFEST}" ]]; then
    awk -F '\t' -v s="${scene}" '$1 == s {p=$2} $2 == s {p=$3} END {print p}' "${MEMORY_MANIFEST}"
    return 0
  fi

  local env_name="MEMORY_PATH_${scene}"
  echo "${!env_name:-}"
}

log_status() {
  local scene="$1" stage="$2" status="$3" exit_code="$4" gpu="$5" log_file="$6" memory_path="$7"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$(date +%Y-%m-%dT%H:%M:%S)" \
    "${scene}" \
    "${stage}" \
    "${status}" \
    "${exit_code}" \
    "${gpu}" \
    "${log_file}" \
    "${memory_path}" >> "${STATUS_FILE}"
}

ensure_glace_features() {
  local scene="$1"
  local gpu="$2"
  local scene_root="${ACE_ROOT}/${scene}"
  local train_feat="${scene_root}/train/${GLACE_FEAT_NAME}"
  local test_feat="${scene_root}/test/${GLACE_FEAT_NAME}"
  local log_file="${RUN_ROOT}/logs/${scene}_gpu${gpu}_features.log"

  if [[ "${SKIP_EXISTING_FEATURES}" == "true" && -s "${train_feat}" && -s "${test_feat}" ]]; then
    log_status "${scene}" "features" "skipped_existing" 0 "${gpu}" "${log_file}" ""
    return 0
  fi

  if [[ "${DO_EXTRACT_FEATURES}" != "true" ]]; then
    if [[ ! -s "${train_feat}" || ! -s "${test_feat}" ]]; then
      printf "ERROR: missing GLACE features for %s\n" "${scene}" > "${log_file}"
      printf "train=%s\ntest=%s\n" "${train_feat}" "${test_feat}" >> "${log_file}"
      log_status "${scene}" "features" "failed_missing" 2 "${gpu}" "${log_file}" ""
      return 2
    fi
    return 0
  fi

  if [[ ! -d "${scene_root}/train/rgb" || ! -d "${scene_root}/test/rgb" ]]; then
    printf "ERROR: missing Indoor6 ACE train/test rgb dirs: %s\n" "${scene_root}" > "${log_file}"
    log_status "${scene}" "features" "failed_bad_scene_root" 2 "${gpu}" "${log_file}" ""
    return 2
  fi

  local cmd=(
    conda run --no-capture-output -n "${CONDA_ENV}" python "${GLACE_ROOT}/datasets/extract_features.py"
    "${scene_root}"
    --checkpoint "${GLACE_CHECKPOINT}"
    --batch_size "${FEATURE_BATCH_SIZE}"
    --num_workers "${FEATURE_NUM_WORKERS}"
  )

  {
    printf "[%s] extract GLACE features scene=%s gpu=%s\n" "$(date)" "${scene}" "${gpu}"
    printf "INDOOR6_GLACE_FEATURE_CMD: CUDA_VISIBLE_DEVICES=%q " "${gpu}"
    printf "%q " "${cmd[@]}"
    printf "\n"
  } > "${log_file}"

  if [[ "${DRY_RUN}" == "true" ]]; then
    log_status "${scene}" "features" "dry_run" 0 "${gpu}" "${log_file}" ""
    return 0
  fi

  set +e
  CUDA_VISIBLE_DEVICES="${gpu}" "${cmd[@]}" >> "${log_file}" 2>&1
  local exit_code=$?
  set -e
  if [[ "${exit_code}" -eq 0 ]]; then
    log_status "${scene}" "features" "ok" 0 "${gpu}" "${log_file}" ""
    return 0
  fi

  log_status "${scene}" "features" "failed" "${exit_code}" "${gpu}" "${log_file}" ""
  return "${exit_code}"
}

prepare_memory_shadow_split() {
  local scene="$1"
  local shadow_train="${RUN_ROOT}/_memory_shadow/${scene}/train"
  local ace_train="${ACE_ROOT}/${scene}/train"
  local wai_train="${WAI_ROOT}/${scene}_train"

  mkdir -p "${shadow_train}"
  for name in rgb poses calibration; do
    if [[ ! -d "${ace_train}/${name}" ]]; then
      echo "ERROR: missing ACE train component: ${ace_train}/${name}" >&2
      return 2
    fi
    ln -sfn "${ace_train}/${name}" "${shadow_train}/${name}"
  done

  if [[ ! -s "${ace_train}/${GLACE_FEAT_NAME}" ]]; then
    echo "ERROR: missing GLACE train features: ${ace_train}/${GLACE_FEAT_NAME}" >&2
    return 2
  fi
  ln -sfn "${ace_train}/${GLACE_FEAT_NAME}" "${shadow_train}/${GLACE_FEAT_NAME}"

  if [[ -d "${wai_train}/${AUX_DEPTH_KIND}" ]]; then
    ln -sfn "${wai_train}/${AUX_DEPTH_KIND}" "${shadow_train}/${AUX_DEPTH_KIND}"
  elif [[ -d "${wai_train}/colmap_depth" ]]; then
    ln -sfn "${wai_train}/colmap_depth" "${shadow_train}/colmap_depth"
  elif [[ -d "${ace_train}/sparse_depth" ]]; then
    ln -sfn "${ace_train}/sparse_depth" "${shadow_train}/sparse_depth"
  else
    echo "ERROR: missing depth for memory extraction: ${wai_train}/${AUX_DEPTH_KIND}, ${wai_train}/colmap_depth, or ${ace_train}/sparse_depth" >&2
    return 2
  fi

  echo "${shadow_train}"
}

extract_memory() {
  local scene="$1"
  local gpu="$2"
  local output_dir="${MEMORY_PATH_PREFIX}/${scene}"
  local output_path="${output_dir}/memory_bse.pt"
  local log_file="${RUN_ROOT}/logs/${scene}_gpu${gpu}_memory.log"
  local shadow_train

  mkdir -p "${output_dir}"
  if [[ "${SKIP_EXISTING_MEMORY}" == "true" && -s "${output_path}" ]]; then
    log_status "${scene}" "memory" "skipped_existing" 0 "${gpu}" "${log_file}" "${output_path}"
    echo "${output_path}"
    return 0
  fi

  set +e
  shadow_train="$(prepare_memory_shadow_split "${scene}")"
  local preflight_code=$?
  set -e
  if [[ "${preflight_code}" -ne 0 ]]; then
    log_status "${scene}" "memory" "failed_preflight" "${preflight_code}" "${gpu}" "${log_file}" "${output_path}"
    return "${preflight_code}"
  fi

  {
    printf "[%s] extract memory scene=%s gpu=%s\n" "$(date)" "${scene}" "${gpu}"
    printf "shadow_train=%s\n" "${shadow_train}"
    printf "output_path=%s\n" "${output_path}"
  } > "${log_file}"

  local cmd=(
    conda run --no-capture-output -n "${CONDA_ENV}" python -m ace_dinov2_lmc.memory_extraction.run_memory_extraction
    "${shadow_train}"
    "${output_path}"
    --dataset_loader ace
    --dataset_type custom
    --scene_name "${scene}"
    --device "cuda:${gpu}"
    --n_memory "${N_MEMORY}"
    --use_model glace_encoder
    --glace_root "${GLACE_ROOT}"
    --glace_encoder_path "${GLACE_ENCODER_PATH}"
    --glace_feat_name "${GLACE_FEAT_NAME}"
    --pool_mode bse
    --voxel_size "${MEMORY_VOXEL_SIZE}"
    --unimodal_threshold "${UNIMODAL_THRESHOLD}"
    --dataset_resolution "${MEMORY_IMAGE_RESOLUTION}"
    --depth_valid_range "${MEMORY_DEPTH_MIN}" "${MEMORY_DEPTH_MAX}"
    --patch_depth_sampling nearest_valid
    --global_merge true
    --enable_sor
    --postprocess_on_cpu
  )

  printf "INDOOR6_GLACE_MEMORY_CMD: " >> "${log_file}"
  printf "%q " "${cmd[@]}" >> "${log_file}"
  printf "\n" >> "${log_file}"

  if [[ "${DRY_RUN}" == "true" ]]; then
    log_status "${scene}" "memory" "dry_run" 0 "${gpu}" "${log_file}" "${output_path}"
    echo "${output_path}"
    return 0
  fi

  set +e
  "${cmd[@]}" >> "${log_file}" 2>&1
  local exit_code=$?
  set -e
  if [[ "${exit_code}" -eq 0 ]]; then
    log_status "${scene}" "memory" "ok" 0 "${gpu}" "${log_file}" "${output_path}"
    echo "${output_path}"
    return 0
  fi

  log_status "${scene}" "memory" "failed" "${exit_code}" "${gpu}" "${log_file}" "${output_path}"
  return "${exit_code}"
}

run_scene() {
  local scene="$1"
  local gpu="$2"
  local memory_path="$3"
  local log_file="${RUN_ROOT}/logs/${scene}_gpu${gpu}_train.log"

  {
    printf "[%s] train scene=%s gpu=%s\n" "$(date)" "${scene}" "${gpu}"
    printf "memory_path=%s\n" "${memory_path}"
  } > "${log_file}"

  if [[ "${DRY_RUN}" == "true" ]]; then
    SCENE="${scene}" GPU_ID="${gpu}" MEMORY_PATH="${memory_path}" RUN_ROOT="${RUN_ROOT}" DRY_RUN=true \
      bash ace_dinov2_lmc/scripts/run_indoor6_glace_lmc_scene.sh >> "${log_file}" 2>&1
    log_status "${scene}" "train" "dry_run" 0 "${gpu}" "${log_file}" "${memory_path}"
    return 0
  fi

  set +e
  SCENE="${scene}" GPU_ID="${gpu}" MEMORY_PATH="${memory_path}" RUN_ROOT="${RUN_ROOT}" CONDA_ENV="${CONDA_ENV}" AUX_DEPTH_KIND="${AUX_DEPTH_KIND}" \
      bash ace_dinov2_lmc/scripts/run_indoor6_glace_lmc_scene.sh >> "${log_file}" 2>&1
  local exit_code=$?
  set -e
  if [[ "${exit_code}" -eq 0 ]]; then
    log_status "${scene}" "train" "ok" 0 "${gpu}" "${log_file}" "${memory_path}"
    return 0
  fi

  log_status "${scene}" "train" "failed" "${exit_code}" "${gpu}" "${log_file}" "${memory_path}"
  return "${exit_code}"
}

run_group() {
  local gpu="$1"
  shift
  local scene memory_path

  for scene in "$@"; do
    if ! ensure_glace_features "${scene}" "${gpu}"; then
      echo "[failed] ${scene}: GLACE feature extraction/preflight failed; log=${RUN_ROOT}/logs/${scene}_gpu${gpu}_features.log" >&2
      if [[ "${CONTINUE_ON_ERROR}" != "true" ]]; then
        return 1
      fi
      continue
    fi

    memory_path="$(lookup_memory "${scene}")"
    if [[ -z "${memory_path}" || ! -s "${memory_path}" ]]; then
      if [[ "${DO_EXTRACT_MEMORY}" == "true" ]]; then
        echo "[memory] ${scene} on cuda:${gpu}"
        if ! memory_path="$(extract_memory "${scene}" "${gpu}")"; then
          echo "[failed] ${scene}: memory extraction failed; log=${RUN_ROOT}/logs/${scene}_gpu${gpu}_memory.log" >&2
          if [[ "${CONTINUE_ON_ERROR}" != "true" ]]; then
            return 1
          fi
          continue
        fi
      else
        local log_file="${RUN_ROOT}/logs/${scene}_gpu${gpu}_preflight.log"
        printf "ERROR: missing memory for %s\n" "${scene}" > "${log_file}"
        printf "Set MEMORY_PATH_%s, MEMORY_PATH_PREFIX, MEMORY_MANIFEST, or DO_EXTRACT_MEMORY=true.\n" "${scene}" >> "${log_file}"
        log_status "${scene}" "preflight" "failed_missing_memory" 2 "${gpu}" "${log_file}" "${memory_path:-}"
        echo "[failed] ${scene}: missing memory, log=${log_file}" >&2
        if [[ "${CONTINUE_ON_ERROR}" != "true" ]]; then
          return 2
        fi
        continue
      fi
    fi

    echo "[train] ${scene} on cuda:${gpu}"
    if run_scene "${scene}" "${gpu}" "${memory_path}"; then
      echo "[ok] ${scene} on cuda:${gpu}"
    else
      echo "[failed] ${scene} on cuda:${gpu}; log=${RUN_ROOT}/logs/${scene}_gpu${gpu}_train.log" >&2
      if [[ "${CONTINUE_ON_ERROR}" != "true" ]]; then
        return 1
      fi
    fi
  done
}

read -r -a SCENES_GPU0 <<< "${SCENES_GPU0_STR}"
read -r -a SCENES_GPU1 <<< "${SCENES_GPU1_STR}"

printf "Run root : %s\n" "${RUN_ROOT}"
printf "GPU 0    : %s\n" "${SCENES_GPU0[*]}"
printf "GPU 1    : %s\n" "${SCENES_GPU1[*]}"
printf "Features : extract=%s, batch=%s, workers=%s\n" "${DO_EXTRACT_FEATURES}" "${FEATURE_BATCH_SIZE}" "${FEATURE_NUM_WORKERS}"
printf "Depth    : %s (COLMAP-derived pseudo depth for Indoor6)\n" "${AUX_DEPTH_KIND}"
printf "Memory   : %s (extract=%s, n=%s, voxel=%s)\n" "${MEMORY_PATH_PREFIX}" "${DO_EXTRACT_MEMORY}" "${N_MEMORY}" "${MEMORY_VOXEL_SIZE}"
printf "Dry run  : %s\n" "${DRY_RUN}"
printf "Status   : %s\n" "${STATUS_FILE}"

run_group "${GPU_0}" "${SCENES_GPU0[@]}" &
pid0=$!
run_group "${GPU_1}" "${SCENES_GPU1[@]}" &
pid1=$!

set +e
wait "${pid0}"
code0=$?
wait "${pid1}"
code1=$?
set -e

if [[ "${code0}" -ne 0 || "${code1}" -ne 0 ]]; then
  echo "Done with failures. Status: ${STATUS_FILE}" >&2
  exit 1
fi

echo "Done. Status: ${STATUS_FILE}"
