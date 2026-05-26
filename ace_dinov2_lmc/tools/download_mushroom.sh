#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/xwh/project/ace_depth}"
REPO_ROOT="${REPO_ROOT:-${ROOT_DIR}/ace_dinov2_lmc}"
OUT_ROOT="${OUT_ROOT:-/data/xwh/MuSHRoom}"

MODE="${MODE:-subset3}"               # subset3 | full10
DOWNLOAD_KINECT="${DOWNLOAD_KINECT:-true}"
DOWNLOAD_IPHONE="${DOWNLOAD_IPHONE:-true}"
DOWNLOAD_IPHONE_COLMAP="${DOWNLOAD_IPHONE_COLMAP:-true}"
EXTRACT_ARCHIVES="${EXTRACT_ARCHIVES:-true}"
DRY_RUN="${DRY_RUN:-false}"

HTTP_PROXY_URL="${HTTP_PROXY_URL:-}"
HTTPS_PROXY_URL="${HTTPS_PROXY_URL:-${HTTP_PROXY_URL}}"

LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/tools/download_logs}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
LOG_FILE="${LOG_ROOT}/download_mushroom_${RUN_TAG}.log"

KINECT_RECORD="${KINECT_RECORD:-10724477}"
IPHONE_RECORD="${IPHONE_RECORD:-10230733}"
IPHONE_COLMAP_RECORD="${IPHONE_COLMAP_RECORD:-13986996}"
IPHONE_COLMAP_FILE="${IPHONE_COLMAP_FILE:-room_datasets_iphone_colmap.tar}"

mkdir -p "${LOG_ROOT}" "${OUT_ROOT}/kinect" "${OUT_ROOT}/iphone" "${OUT_ROOT}/iphone_colmap"

bool_true() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

log() {
  echo "$*" | tee -a "${LOG_FILE}"
}

run_cmd() {
  log "[cmd] $*"
  if ! bool_true "${DRY_RUN}"; then
    "$@" 2>&1 | tee -a "${LOG_FILE}"
  fi
}

archive_base_exists() {
  local base="$1"
  if [ -d "${base}" ]; then
    return 0
  fi
  if [ -d "${OUT_ROOT}/kinect/room_datasets/${base}" ]; then
    return 0
  fi
  if [ -d "${OUT_ROOT}/iphone/room_datasets/${base}" ]; then
    return 0
  fi
  return 1
}

download_one() {
  local record="$1"
  local filename="$2"
  local out_dir="$3"
  local target="${out_dir}/${filename}"
  run_cmd wget -c "https://zenodo.org/records/${record}/files/${filename}?download=1" -O "${target}"
}

extract_one() {
  local archive="$1"
  local out_dir="$2"
  local stem="$3"
  if archive_base_exists "${stem}"; then
    log "[skip] extracted ${stem}"
    return 0
  fi
  case "${archive}" in
    *.tar.gz) run_cmd tar -xzvf "${out_dir}/${archive}" -C "${out_dir}" ;;
    *.tar) run_cmd tar -xvf "${out_dir}/${archive}" -C "${out_dir}" ;;
    *) log "[FAIL] unsupported archive suffix: ${archive}"; exit 1 ;;
  esac
}

rooms_subset3=(
  coffee_room
  classroom
  vr_room
)

rooms_full10=(
  activity
  classroom
  coffee_room
  computer
  honka
  koivu
  kokko
  olohuone
  sauna
  vr_room
)

select_rooms() {
  case "${MODE}" in
    subset3) printf '%s\n' "${rooms_subset3[@]}" ;;
    full10) printf '%s\n' "${rooms_full10[@]}" ;;
    *) log "[FAIL] unsupported MODE=${MODE} (use subset3 or full10)"; exit 1 ;;
  esac
}

main() {
  : > "${LOG_FILE}"
  log "[info] out_root=${OUT_ROOT}"
  log "[info] mode=${MODE}"
  log "[info] dry_run=${DRY_RUN}"

  if [ -n "${HTTP_PROXY_URL}" ]; then
    export http_proxy="${HTTP_PROXY_URL}"
    export HTTP_PROXY="${HTTP_PROXY_URL}"
    log "[info] http_proxy=${HTTP_PROXY_URL}"
  fi
  if [ -n "${HTTPS_PROXY_URL}" ]; then
    export https_proxy="${HTTPS_PROXY_URL}"
    export HTTPS_PROXY="${HTTPS_PROXY_URL}"
    log "[info] https_proxy=${HTTPS_PROXY_URL}"
  fi

  mapfile -t rooms < <(select_rooms)

  if bool_true "${DOWNLOAD_KINECT}"; then
    for room in "${rooms[@]}"; do
      download_one "${KINECT_RECORD}" "${room}_kinect.tar.gz" "${OUT_ROOT}/kinect"
    done
  fi

  if bool_true "${DOWNLOAD_IPHONE}"; then
    for room in "${rooms[@]}"; do
      download_one "${IPHONE_RECORD}" "${room}_iphone.tar.gz" "${OUT_ROOT}/iphone"
    done
  fi

  if bool_true "${DOWNLOAD_IPHONE_COLMAP}"; then
    download_one "${IPHONE_COLMAP_RECORD}" "${IPHONE_COLMAP_FILE}" "${OUT_ROOT}/iphone_colmap"
  fi

  if bool_true "${EXTRACT_ARCHIVES}"; then
    if bool_true "${DOWNLOAD_KINECT}"; then
      for room in "${rooms[@]}"; do
        extract_one "${room}_kinect.tar.gz" "${OUT_ROOT}/kinect" "${room}"
      done
    fi
    if bool_true "${DOWNLOAD_IPHONE}"; then
      for room in "${rooms[@]}"; do
        extract_one "${room}_iphone.tar.gz" "${OUT_ROOT}/iphone" "${room}"
      done
    fi
    if bool_true "${DOWNLOAD_IPHONE_COLMAP}"; then
      extract_one "${IPHONE_COLMAP_FILE}" "${OUT_ROOT}/iphone_colmap" "room_datasets_iphone_colmap"
    fi
  fi

  log "[done] log=${LOG_FILE}"
}

main "$@"
