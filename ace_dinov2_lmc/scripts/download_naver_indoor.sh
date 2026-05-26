#!/usr/bin/env bash
set -euo pipefail

# Download NAVER LABS indoor localization dataset archives only.
# This script does not extract anything.
#
# Defaults target the recommended pilot scene:
#   Hyundai Department Store 4F
#
# Example:
#   bash scripts/download_naver_indoor.sh
#
# Background:
#   mkdir -p /home/xwh/project/ace_depth/ace_dinov2_lmc/04_evaluation/naver_indoor/logs
#   nohup bash scripts/download_naver_indoor.sh \
#     > /home/xwh/project/ace_depth/ace_dinov2_lmc/04_evaluation/naver_indoor/logs/hyundai_4f_download.log 2>&1 &

SITE="${SITE:-HyundaiDepartmentStore}"
FLOOR="${FLOOR:-4F}"
OUT_ROOT="${OUT_ROOT:-/data/xwh/dataset_staging/naver_indoor}"
COMPONENTS="${COMPONENTS:-mapping test validation mapping_lidar_only}"
BASE_URL="${BASE_URL:-https://download.europe.naverlabs.com/kapture}"

DATASET_PREFIX="${SITE}_${FLOOR}_release"
OUT_DIR="${OUT_ROOT}/${SITE}_${FLOOR}"
mkdir -p "${OUT_DIR}"

download() {
  local url="$1"
  local out="$2"
  echo "[NAVER] $(date '+%Y-%m-%d %H:%M:%S') downloading ${url}"
  if command -v wget >/dev/null 2>&1; then
    wget -c --tries=10 --timeout=30 --waitretry=5 -O "${out}" "${url}"
  else
    curl -L -C - --retry 10 --retry-all-errors --connect-timeout 30 --fail -o "${out}" "${url}"
  fi
  if [[ ! -s "${out}" ]]; then
    echo "[NAVER] Download failed or produced empty file: ${out}" >&2
    exit 1
  fi
}

echo "[NAVER] Site: ${SITE}"
echo "[NAVER] Floor: ${FLOOR}"
echo "[NAVER] Output: ${OUT_DIR}"
echo "[NAVER] Components: ${COMPONENTS}"

for component in ${COMPONENTS}; do
  archive_name="${DATASET_PREFIX}_${component}.tar.gz"
  url="${BASE_URL}/${archive_name}"
  out="${OUT_DIR}/${archive_name}"
  download "${url}" "${out}"
done

echo "[NAVER] Done."
