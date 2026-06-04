#!/usr/bin/env bash
set -euo pipefail

# Sequential Indoor6 repair runner on GPU1:
#   1. scene3 uses COLMAP depth clipping at 20m because its pseudo depth has 100m+ outliers.
#   2. scene4a keeps the default 1000m clip to preserve the normal route.
#
# This wraps run_indoor6_ace_fcn_glace_lmc_all_dual_gpu.sh with empty groups on GPUs 0/2/3,
# so only GPU1 receives training jobs.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

CONDA_ENV="${CONDA_ENV:-mapanything}"
BASE_RUN_ROOT="${BASE_RUN_ROOT:-/data/xwh/ace_dinov2_lmc/04_evaluation/indoor6_scene3_scene4a_gpu1_depthfix}"
SCENE3_RUN_ROOT="${SCENE3_RUN_ROOT:-/data/xwh/ace_dinov2_lmc/04_evaluation/indoor6_scene3_depth20_repair}"
SCENE4A_RUN_ROOT="${SCENE4A_RUN_ROOT:-${BASE_RUN_ROOT}/scene4a_default_depth}"
RUNNER="ace_dinov2_lmc/scripts/run_indoor6_ace_fcn_glace_lmc_all_dual_gpu.sh"
SKIP_COMPLETED_SCENES="${SKIP_COMPLETED_SCENES:-true}"

COMMON_ENV=(
  CONDA_ENV="${CONDA_ENV}"
  AUX_DEPTH_KIND="${AUX_DEPTH_KIND:-colmap_depth}"
  COORD_SOURCE="${COORD_SOURCE:-colmap_depth}"
  LMC_ITERATIONS="${LMC_ITERATIONS:-12}"
  METHODS="${METHODS:-features memory stage1 stage2}"
  SCENES_GPU0_STR=""
  SCENES_GPU1_STR="__SCENE__"
  SCENES_GPU2_STR=""
  SCENES_GPU3_STR=""
  SKIP_EXISTING="${SKIP_EXISTING:-true}"
  CONTINUE_ON_ERROR="false"
)

scene_stage2_done() {
  local scene="$1"
  local run_root="$2"
  local status_file="${run_root}/status.tsv"

  [[ -s "${status_file}" ]] || return 1
  awk -F'\t' -v scene="${scene}" '
    $2 == scene && $3 == "stage2" && $4 == "train" && $5 == "ok" { found = 1 }
    END { exit(found ? 0 : 1) }
  ' "${status_file}"
}

run_one() {
  local scene="$1"
  local run_root="$2"
  local memory_depth_max="$3"
  local aux_depth_max="$4"

  echo "============================================================"
  echo "[run] ${scene} on GPU1"
  echo "Run root        : ${run_root}"
  echo "Depth clip      : memory<=${memory_depth_max}m aux<=${aux_depth_max}m"
  echo "============================================================"

  if [[ "${SKIP_COMPLETED_SCENES}" == "true" ]] && scene_stage2_done "${scene}" "${run_root}"; then
    echo "[skip] ${scene} already has stage2 train ok in ${run_root}/status.tsv"
    return 0
  fi

  local env_args=()
  local item
  for item in "${COMMON_ENV[@]}"; do
    env_args+=("${item/__SCENE__/${scene}}")
  done

  set +e
  env \
    "${env_args[@]}" \
    RUN_ROOT="${run_root}" \
    MEMORY_DEPTH_MAX="${memory_depth_max}" \
    C1_AUX_DEPTH_MAX="${aux_depth_max}" \
    bash "${RUNNER}"
  local rc=$?
  set -e

  if [[ "${rc}" -ne 0 ]]; then
    if scene_stage2_done "${scene}" "${run_root}"; then
      echo "[warn] ${scene} runner exited rc=${rc}, but stage2 train is marked ok; continuing."
      return 0
    fi
    return "${rc}"
  fi
}

# scene3 already has a clipped memory in SCENE3_RUN_ROOT; SKIP_EXISTING=true will reuse it.
run_one "scene3" "${SCENE3_RUN_ROOT}" "20" "20"

# scene4a follows the normal route and writes to its own run root.
run_one "scene4a" "${SCENE4A_RUN_ROOT}" "1000.0" "1000.0"

echo "Done."
echo "scene3 status : ${SCENE3_RUN_ROOT}/status.tsv"
echo "scene4a status: ${SCENE4A_RUN_ROOT}/status.tsv"
