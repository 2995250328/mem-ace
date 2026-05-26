#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/xwh/project/ace_depth}"
REPO_ROOT="${REPO_ROOT:-${ROOT_DIR}/ace_dinov2_lmc}"

# Official ScanNet assets the user must obtain after ToU approval.
SCANNET_DOWNLOADER="${SCANNET_DOWNLOADER:-}"
SCANNET_SENSOR_READER="${SCANNET_SENSOR_READER:-}"

DOWNLOAD_PYTHON="${DOWNLOAD_PYTHON:-python2}"
READER_PYTHON="${READER_PYTHON:-python2}"

OUT_ROOT="${OUT_ROOT:-/data/xwh/scannet6}"
SCENE_LIST="${SCENE_LIST:-${REPO_ROOT}/tools/scannet6_scenes.txt}"

DOWNLOAD_SENS="${DOWNLOAD_SENS:-true}"
DOWNLOAD_INFO_TXT="${DOWNLOAD_INFO_TXT:-true}"
DOWNLOAD_LABEL_MAP="${DOWNLOAD_LABEL_MAP:-false}"
DOWNLOAD_MESH="${DOWNLOAD_MESH:-false}"
EXTRACT_FRAMES="${EXTRACT_FRAMES:-true}"

EXPORT_COLOR="${EXPORT_COLOR:-true}"
EXPORT_DEPTH="${EXPORT_DEPTH:-true}"
EXPORT_POSE="${EXPORT_POSE:-true}"
EXPORT_INTRINSIC="${EXPORT_INTRINSIC:-true}"
FRAME_SKIP="${FRAME_SKIP:-1}"

DRY_RUN="${DRY_RUN:-false}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/tools/download_logs}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
LOG_FILE="${LOG_ROOT}/download_scannet6_${RUN_TAG}.log"

mkdir -p "${LOG_ROOT}" "${OUT_ROOT}/scans"

bool_true() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

require_file() {
  local path="$1"
  local label="$2"
  if [ ! -f "${path}" ]; then
    echo "[FAIL] missing ${label}: ${path}" | tee -a "${LOG_FILE}"
    exit 1
  fi
}

run_cmd() {
  echo "[cmd] $*" | tee -a "${LOG_FILE}"
  if ! bool_true "${DRY_RUN}"; then
    "$@" 2>&1 | tee -a "${LOG_FILE}"
  fi
}

download_scene_file() {
  local scene="$1"
  local suffix="$2"
  local target="${OUT_ROOT}/scans/${scene}/${scene}${suffix}"
  if [ -f "${target}" ]; then
    echo "[skip] exists ${target}" | tee -a "${LOG_FILE}"
    return 0
  fi
  mkdir -p "${OUT_ROOT}/scans/${scene}"
  run_cmd "${DOWNLOAD_PYTHON}" "${SCANNET_DOWNLOADER}" -o "${OUT_ROOT}/scans" --id "${scene}" --type "${suffix}"
}

extract_scene() {
  local scene="$1"
  local scene_dir="${OUT_ROOT}/scans/${scene}"
  local sens_file="${scene_dir}/${scene}.sens"
  if [ ! -f "${sens_file}" ]; then
    echo "[skip] no sens for ${scene}" | tee -a "${LOG_FILE}"
    return 0
  fi

  local ready=true
  if bool_true "${EXPORT_COLOR}" && [ ! -d "${scene_dir}/color" ]; then ready=false; fi
  if bool_true "${EXPORT_DEPTH}" && [ ! -d "${scene_dir}/depth" ]; then ready=false; fi
  if bool_true "${EXPORT_POSE}" && [ ! -d "${scene_dir}/pose" ]; then ready=false; fi
  if bool_true "${EXPORT_INTRINSIC}" && [ ! -d "${scene_dir}/intrinsic" ]; then ready=false; fi
  if [ "${ready}" = true ]; then
    echo "[skip] extracted ${scene_dir}" | tee -a "${LOG_FILE}"
    return 0
  fi

  local args=( "${READER_PYTHON}" "${SCANNET_SENSOR_READER}" "--filename" "${sens_file}" "--output_path" "${scene_dir}" "--frame_skip" "${FRAME_SKIP}" )
  if bool_true "${EXPORT_COLOR}"; then args+=(--export_color_images); fi
  if bool_true "${EXPORT_DEPTH}"; then args+=(--export_depth_images); fi
  if bool_true "${EXPORT_POSE}"; then args+=(--export_poses); fi
  if bool_true "${EXPORT_INTRINSIC}"; then args+=(--export_intrinsics); fi
  run_cmd "${args[@]}"
}

main() {
  : > "${LOG_FILE}"
  echo "[info] out_root=${OUT_ROOT}" | tee -a "${LOG_FILE}"
  echo "[info] scene_list=${SCENE_LIST}" | tee -a "${LOG_FILE}"
  echo "[info] dry_run=${DRY_RUN}" | tee -a "${LOG_FILE}"

  require_file "${SCENE_LIST}" "scene list"
  require_file "${SCANNET_DOWNLOADER}" "official ScanNet downloader"
  if bool_true "${EXTRACT_FRAMES}"; then
    require_file "${SCANNET_SENSOR_READER}" "official ScanNet SensorData.py"
  fi

  while IFS= read -r scene || [ -n "${scene}" ]; do
    [ -n "${scene}" ] || continue
    echo "[scene] ${scene}" | tee -a "${LOG_FILE}"

    if bool_true "${DOWNLOAD_SENS}"; then
      download_scene_file "${scene}" ".sens"
    fi
    if bool_true "${DOWNLOAD_INFO_TXT}"; then
      download_scene_file "${scene}" ".txt"
    fi
    if bool_true "${DOWNLOAD_MESH}"; then
      download_scene_file "${scene}" "_vh_clean_2.ply"
    fi

    if bool_true "${EXTRACT_FRAMES}"; then
      extract_scene "${scene}"
    fi
  done < "${SCENE_LIST}"

  if bool_true "${DOWNLOAD_LABEL_MAP}"; then
    run_cmd "${DOWNLOAD_PYTHON}" "${SCANNET_DOWNLOADER}" -o "${OUT_ROOT}" --label_map
  fi

  echo "[done] log=${LOG_FILE}" | tee -a "${LOG_FILE}"
}

main "$@"
