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
MATRIX_METHODS="${MATRIX_METHODS:-stage1}"
S1_LOSS_STEP_MODE="${S1_LOSS_STEP_MODE:-fixed_zero}"
S1_EARLY_STOP="${S1_EARLY_STOP:-False}"
LMC_LOG_RUNTIME_STATS="${LMC_LOG_RUNTIME_STATS:-False}"
LMC_RUNTIME_STATS_INTERVAL="${LMC_RUNTIME_STATS_INTERVAL:-100}"
LMC_RUNTIME_STATS_MAX_PIXELS="${LMC_RUNTIME_STATS_MAX_PIXELS:-4096}"
NUM_DATA_LOADER_WORKERS="${NUM_DATA_LOADER_WORKERS:-12}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-6}"
LMC_FUSION_CASCADE_LAYERS="${LMC_FUSION_CASCADE_LAYERS:-4}"
LMC_FUSION_REREAD_DELTA_ALPHA="${LMC_FUSION_REREAD_DELTA_ALPHA:-1.0}"
LMC_FUSION_REREAD_SCALAR_GATE="${LMC_FUSION_REREAD_SCALAR_GATE:-False}"
LMC_FUSION_REREAD_GATE_INIT="${LMC_FUSION_REREAD_GATE_INIT:-0.0}"

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
    METHODS="${MATRIX_METHODS}"
    RUN_ROOT="${run_root}"
    SKIP_EXISTING=false
    CONTINUE_ON_ERROR=false
    GPU_0="${gpu_id}"
    GPU_1="${gpu_id}"
    LMC_FUSION_REFINEMENT_MODE="${mode}"
    LMC_FUSION_CASCADE_LAYERS="${LMC_FUSION_CASCADE_LAYERS}"
    LMC_FUSION_ASSEMBLY_GAMMA_INIT=0.0
    LMC_FUSION_REREAD_DELTA_ALPHA="${LMC_FUSION_REREAD_DELTA_ALPHA}"
    LMC_FUSION_REREAD_SCALAR_GATE="${LMC_FUSION_REREAD_SCALAR_GATE}"
    LMC_FUSION_REREAD_GATE_INIT="${LMC_FUSION_REREAD_GATE_INIT}"
    S1_LOSS_STEP_MODE="${S1_LOSS_STEP_MODE}"
    S1_EARLY_STOP="${S1_EARLY_STOP}"
    LMC_LOG_RUNTIME_STATS="${LMC_LOG_RUNTIME_STATS}"
    LMC_RUNTIME_STATS_INTERVAL="${LMC_RUNTIME_STATS_INTERVAL}"
    LMC_RUNTIME_STATS_MAX_PIXELS="${LMC_RUNTIME_STATS_MAX_PIXELS}"
    NUM_DATA_LOADER_WORKERS="${NUM_DATA_LOADER_WORKERS}"
    EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS}"
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
  LAST_LAUNCHED_PID="${pid}"
  printf "%s gpu=%s mode=%s pid=%s log=%s run_root=%s\n" "${tag}" "${gpu_id}" "${mode}" "${pid}" "${log_path}" "${run_root}"
  printf "%s\n" "${pid}" >"${PAIR_ROOT}/${tag}.pid"
}

printf "Matrix methods: %s\n" "${MATRIX_METHODS}"
printf "Runtime stats : enabled=%s interval=%s max_pixels=%s\n" \
  "${LMC_LOG_RUNTIME_STATS}" "${LMC_RUNTIME_STATS_INTERVAL}" "${LMC_RUNTIME_STATS_MAX_PIXELS}"
printf "Data workers  : train=%s eval=%s\n" "${NUM_DATA_LOADER_WORKERS}" "${EVAL_NUM_WORKERS}"
printf "GPU limit     : 2 concurrent jobs on cuda:0,cuda:1\n"

if [[ "${DRY_RUN}" == "true" ]]; then
  launch_job 0 single single
  launch_job 1 progressive_reread pmrf
  launch_job 0 adapter_ffn adapter_ffn
  exit 0
fi

launch_job 0 single single
pid_single="${LAST_LAUNCHED_PID}"
launch_job 1 progressive_reread pmrf
pid_pmrf="${LAST_LAUNCHED_PID}"

set +e
wait "${pid_single}"
status_single=$?
set -e
printf "single completed exit=%s; launching adapter_ffn on gpu=0\n" "${status_single}"
launch_job 0 adapter_ffn adapter_ffn
pid_adapter="${LAST_LAUNCHED_PID}"

set +e
wait "${pid_pmrf}"
status_pmrf=$?
wait "${pid_adapter}"
status_adapter=$?
set -e

printf "matrix exits: single=%s pmrf=%s adapter_ffn=%s\n" \
  "${status_single}" "${status_pmrf}" "${status_adapter}"

if (( status_single != 0 || status_pmrf != 0 || status_adapter != 0 )); then
  exit 1
fi
