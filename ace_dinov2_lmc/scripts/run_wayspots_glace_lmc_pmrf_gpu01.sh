#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE_SCRIPT="${ROOT_DIR}/ace_dinov2_lmc/scripts/run_wayspots_ace_fcn_lmc_suite.sh"
SCENE="${SCENE:-wayspots_bears}"
SOURCE_SUITE_ROOT="${SOURCE_SUITE_ROOT:-/data/xwh/ace_dinov2_lmc/04_evaluation/wayspots_ace_fcn_lmc_suite/20260530_181745}"
SOURCE_MEMORY="${SOURCE_MEMORY:-${SOURCE_SUITE_ROOT}/memory/${SCENE}/memory_ace_fcn_sparse_sp_r4.pt}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
PAIR_ROOT="${PAIR_ROOT:-/data/xwh/ace_dinov2_lmc/04_evaluation/pmrf_bears_ace_fcn_20260614/train/${STAMP}}"
DRY_RUN="${DRY_RUN:-false}"
TRACE_BASH="${TRACE_BASH:-false}"

if [[ ! -s "${SOURCE_MEMORY}" ]]; then
  echo "SOURCE_MEMORY not found or empty: ${SOURCE_MEMORY}" >&2
  exit 1
fi

mkdir -p "${PAIR_ROOT}/logs"

prepare_memory_link() {
  local run_root="$1"
  local memory_dir="${run_root}/memory/${SCENE}"
  local target="${memory_dir}/memory_ace_fcn_sparse_sp_r4.pt"
  mkdir -p "${memory_dir}"
  if [[ ! -e "${target}" && ! -L "${target}" ]]; then
    ln -s "${SOURCE_MEMORY}" "${target}"
  fi
}

launch_job() {
  local gpu_id="$1"
  local mode="$2"
  local tag="$3"
  local run_root="${PAIR_ROOT}/${tag}"
  local log_path="${PAIR_ROOT}/logs/${tag}.log"

  prepare_memory_link "${run_root}"

  local env_args=(
    SCENES="${SCENE}"
    METHODS="stage1 stage2"
    RUN_ROOT="${run_root}"
    SKIP_EXISTING=false
    CONTINUE_ON_ERROR=false
    GPU_0="${gpu_id}"
    GPU_1="${gpu_id}"
    LMC_FUSION_REFINEMENT_MODE="${mode}"
    LMC_FUSION_CASCADE_LAYERS=2
    LMC_FUSION_ASSEMBLY_GAMMA_INIT=0.0
  )

  if [[ "${DRY_RUN}" == "true" ]]; then
    env "${env_args[@]}" DRY_RUN=true bash -x "${BASE_SCRIPT}"
    return
  fi

  local bash_args=()
  if [[ "${TRACE_BASH}" == "true" ]]; then
    bash_args=(-x)
  fi

  setsid env "${env_args[@]}" bash "${bash_args[@]}" "${BASE_SCRIPT}" >"${log_path}" 2>&1 < /dev/null &
  local pid=$!
  printf "%s gpu=%s mode=%s pid=%s log=%s run_root=%s\n" "${tag}" "${gpu_id}" "${mode}" "${pid}" "${log_path}" "${run_root}"
  printf "%s\n" "${pid}" >"${PAIR_ROOT}/${tag}.pid"
}

launch_job 0 single single
launch_job 1 progressive_reread pmrf
