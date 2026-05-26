#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/xwh/project/ace_depth}"
REPO_ROOT="${REPO_ROOT:-${ROOT_DIR}/ace_dinov2_lmc}"
MUSHROOM_ROOT="${MUSHROOM_ROOT:-/data/xwh/MuSHRoom/kinect}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/xwh/MuSHRoom_ace}"
ROOMS_STR="${ROOMS_STR:-coffee_room classroom vr_room}"
COPY_FILES="${COPY_FILES:-false}"
SKIP_INCOMPLETE="${SKIP_INCOMPLETE:-true}"

bool_true() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

cd "${REPO_ROOT}"
read -r -a rooms <<< "${ROOMS_STR}"

args=(
  tools/convert_mushroom_kinect_to_ace.py
  --mushroom-root "${MUSHROOM_ROOT}"
  --output-root "${OUTPUT_ROOT}"
  --rooms "${rooms[@]}"
)
if bool_true "${COPY_FILES}"; then
  args+=(--copy-files)
fi
if bool_true "${SKIP_INCOMPLETE}"; then
  args+=(--skip-incomplete)
fi

python "${args[@]}"

for room in "${rooms[@]}"; do
  scene="${OUTPUT_ROOT}/${room}"
  if [ -d "${scene}/train/rgb" ] && [ -d "${scene}/test/rgb" ]; then
    python "${ROOT_DIR}/check_dataset_dinov2.py" "${scene}"
  else
    echo "[skip-check] ${scene} is missing train/test rgb dirs"
  fi
done
