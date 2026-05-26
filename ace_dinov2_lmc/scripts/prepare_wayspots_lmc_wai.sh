#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

CONDA_ENV="${CONDA_ENV:-mapanything}"
CONDA_RUN=(conda run --no-capture-output -n "${CONDA_ENV}")

WAYSPOTS_ACE_ROOT="${WAYSPOTS_ACE_ROOT:-/data/xwh/Wayspots}"
WAYSPOTS_WAI_ROOT="${WAYSPOTS_WAI_ROOT:-/data/xwh/Wayspots_wai/custom}"
SCENES="${SCENES:-wayspots_bears}"
COPY_FILES="${COPY_FILES:-false}"

ensure_link() {
  local link_path="$1"
  local target_path="$2"
  if [[ -e "${link_path}" || -L "${link_path}" ]]; then
    return 0
  fi
  ln -s "${target_path}" "${link_path}"
}

prepare_scene_depth_dirs() {
  local scene_root="$1"
  local train_root="${scene_root}/train"
  local test_root="${scene_root}/test"

  if [[ -d "${train_root}/sparse_depth_sift_filt_t4_e2_d02_observed_nms_r4" ]]; then
    ensure_link "${train_root}/sparse_depth_mapanything" "${train_root}/sparse_depth_sift_filt_t4_e2_d02_observed_nms_r4"
  fi

  if [[ -d "${train_root}/sparse_depth_sift_filt_t4_e2_d02_observed_nms_r4_sp_r4" ]]; then
    ensure_link "${train_root}/sparse_depth_sampling_sp" "${train_root}/sparse_depth_sift_filt_t4_e2_d02_observed_nms_r4_sp_r4"
    ensure_link "${train_root}/sparse_depth" "${train_root}/sparse_depth_sift_filt_t4_e2_d02_observed_nms_r4_sp_r4"
  fi

  if [[ -d "${test_root}/sparse_depth_sift_filt_t4_e2_d02_observed_nms_r4" ]]; then
    ensure_link "${test_root}/sparse_depth_mapanything" "${test_root}/sparse_depth_sift_filt_t4_e2_d02_observed_nms_r4"
  fi
}

for scene in ${SCENES}; do
  scene_root="${WAYSPOTS_ACE_ROOT}/${scene}"
  if [[ ! -d "${scene_root}/train/rgb" || ! -d "${scene_root}/test/rgb" ]]; then
    echo "Missing ACE-format Wayspots scene: ${scene_root}" >&2
    exit 1
  fi

  prepare_scene_depth_dirs "${scene_root}"

done

CMD=(
  "${CONDA_RUN[@]}" python "${ROOT_DIR}/ace_dinov2_lmc/tools/convert_ace_to_wai.py"
  "${WAYSPOTS_ACE_ROOT}"
  "${WAYSPOTS_WAI_ROOT}"
  --dataset-name wayspots_custom
  --scenes
)
for scene in ${SCENES}; do
  CMD+=("${scene}")
done
if [[ "${COPY_FILES}" == "true" ]]; then
  CMD+=(--copy-files)
fi
printf 'Running: '
printf '%q ' "${CMD[@]}"
printf '
'
"${CMD[@]}"

for scene in ${SCENES}; do
  echo "Prepared WAI scenes: ${WAYSPOTS_WAI_ROOT}/${scene}_train and ${WAYSPOTS_WAI_ROOT}/${scene}_test"
done
