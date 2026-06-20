#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/xwh/project/ace_depth}"
RUN_ROOT="${RUN_ROOT:-/data/xwh/ace_dinov2_lmc/04_evaluation/weak_ffn_control_bears_$(date '+%Y%m%d_%H%M%S')}"
SCENE="${SCENE:-wayspots_bears}"
MEMORY_SOURCE="${MEMORY_SOURCE:-/data/xwh/ace_dinov2_lmc/04_evaluation/pmrf_nonorm_bears_test_20260617/train/20260617_110800/single_repeat/memory/wayspots_bears/memory_ace_fcn_sparse_sp_r4.pt}"

mkdir -p "${RUN_ROOT}"
if [[ ! -f "${MEMORY_SOURCE}" ]]; then
  echo "Missing MEMORY_SOURCE: ${MEMORY_SOURCE}" >&2
  exit 2
fi

run_variant() {
  local variant="$1"
  local gpu="$2"
  local mode="$3"
  local alpha="$4"
  local scalar_gate="$5"
  local gate_init="$6"
  local post_norm="$7"

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] START ${variant} gpu=${gpu} mode=${mode} alpha=${alpha} gate=${scalar_gate}/${gate_init} post_norm=${post_norm}"
  mkdir -p "${RUN_ROOT}/${variant}/memory/${SCENE}"
  ln -sf "${MEMORY_SOURCE}" "${RUN_ROOT}/${variant}/memory/${SCENE}/memory_ace_fcn_sparse_sp_r4.pt"
  cd "${PROJECT_ROOT}"
  RUN_ROOT="${RUN_ROOT}/${variant}" \
  SCENES="${SCENE}" \
  METHODS="stage1" \
  GPU_0="${gpu}" \
  GPU_1="${gpu}" \
  SKIP_EXISTING="false" \
  CONTINUE_ON_ERROR="false" \
  LMC_FUSION_CASCADE_LAYERS="4" \
  LMC_FUSION_REFINEMENT_MODE="${mode}" \
  LMC_FUSION_REREAD_DELTA_ALPHA="${alpha}" \
  LMC_FUSION_REREAD_SCALAR_GATE="${scalar_gate}" \
  LMC_FUSION_REREAD_GATE_INIT="${gate_init}" \
  LMC_FUSION_REREAD_POST_NORM="${post_norm}" \
  LMC_FUSION_ASSEMBLY_GAMMA_INIT="0.0" \
  S1_LOSS_STEP_MODE="fixed_zero" \
  S1_EARLY_STOP="False" \
  LMC_LOG_RUNTIME_STATS="True" \
  LMC_RUNTIME_STATS_INTERVAL="100" \
  LMC_RUNTIME_STATS_MAX_PIXELS="4096" \
  NUM_DATA_LOADER_WORKERS="12" \
  EVAL_NUM_WORKERS="6" \
  bash ace_dinov2_lmc/scripts/run_wayspots_ace_fcn_lmc_suite.sh
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE ${variant}"
}

run_queue_gpu0() {
  run_variant single_repeat 0 single 0.0 False 0.0 True \
    2>&1 | tee "${RUN_ROOT}/single_repeat.runner.log"
  run_variant weak_ffn_alpha010 0 weak_residual_ffn 0.10 True -2.0 False \
    2>&1 | tee "${RUN_ROOT}/weak_ffn_alpha010.runner.log"
}

run_queue_gpu1() {
  run_variant pmrf_nonorm_alpha010 1 progressive_reread 0.10 True -2.0 False \
    2>&1 | tee "${RUN_ROOT}/pmrf_nonorm_alpha010.runner.log"
}

run_queue_gpu0 &
pid_gpu0=$!
run_queue_gpu1 &
pid_gpu1=$!

wait "${pid_gpu0}" "${pid_gpu1}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ALL_DONE ${RUN_ROOT}"
