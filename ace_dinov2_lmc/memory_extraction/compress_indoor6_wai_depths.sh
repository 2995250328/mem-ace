#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/xwh/project/ace_depth}"
REPO_ROOT="${REPO_ROOT:-${ROOT_DIR}/ace_dinov2_lmc}"
WAI_ROOT="${WAI_ROOT:-/data/xwh/mapanything-dataset/wai_data/indoor6}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DRY_RUN="${DRY_RUN:-false}"
DELETE_ORIGINAL="${DELETE_ORIGINAL:-true}"
SCENE="${SCENE:-}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/memory_extraction/compress_logs}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

mkdir -p "${LOG_ROOT}"
LOG_FILE="${LOG_ROOT}/compress_indoor6_wai_depths_${RUN_TAG}.log"

bool_true() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

run_one() {
  local modality="$1"
  local args=(
    memory_extraction/compress_wai_numpy_modality.py
    --wai-root "${WAI_ROOT}"
    --modality "${modality}"
  )

  if [ -n "${SCENE}" ]; then
    args+=(--scene "${SCENE}")
  fi
  if bool_true "${DRY_RUN}"; then
    args+=(--dry-run)
  fi
  if bool_true "${DELETE_ORIGINAL}"; then
    args+=(--delete-original)
  fi

  echo "[compress] modality=${modality} scene=${SCENE:-ALL} dry_run=${DRY_RUN} delete_original=${DELETE_ORIGINAL}" | tee -a "${LOG_FILE}"
  "${PYTHON_BIN}" "${args[@]}" 2>&1 | tee -a "${LOG_FILE}"
}

cd "${REPO_ROOT}"

run_one gt_depth
run_one colmap_depth

echo "[done] log=${LOG_FILE}"
