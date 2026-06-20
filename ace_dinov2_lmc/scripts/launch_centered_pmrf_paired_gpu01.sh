#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/xwh/project/ace_depth"
RUN_ROOT_BASE="${RUN_ROOT_BASE:-/data/xwh/ace_dinov2_lmc/04_evaluation/centered_pmrf_paired_20260619_gpu01}"
RUNNER="${ROOT_DIR}/ace_dinov2_lmc/scripts/run_wayspots_ace_fcn_lmc_suite.sh"

BEARS_MEM="${BEARS_MEM:-/data/xwh/ace_dinov2_lmc/04_evaluation/pmrf_nonorm_bears_test_20260617/train/20260617_110800/single_repeat/memory/wayspots_bears/memory_ace_fcn_sparse_sp_r4.pt}"
SQ_MEM="${SQ_MEM:-/data/xwh/ace_dinov2_lmc/04_evaluation/wayspots_ace_fcn_lmc_suite/20260530_181745/memory/wayspots_squarebench/memory_ace_fcn_sparse_sp_r4.pt}"

prepare_memory_links() {
  local run_root="$1"
  mkdir -p "${run_root}/memory/wayspots_bears"
  mkdir -p "${run_root}/memory/wayspots_squarebench"
  ln -sfn "${BEARS_MEM}" "${run_root}/memory/wayspots_bears/memory_ace_fcn_sparse_sp_r4.pt"
  ln -sfn "${SQ_MEM}" "${run_root}/memory/wayspots_squarebench/memory_ace_fcn_sparse_sp_r4.pt"
}

run_variant() {
  local variant="$1"
  local refinement_mode="$2"
  local run_root="${RUN_ROOT_BASE}/${variant}"

  prepare_memory_links "${run_root}"
  mkdir -p "${RUN_ROOT_BASE}/logs"

  echo "[$(date +%Y-%m-%dT%H:%M:%S)] start ${variant} mode=${refinement_mode} root=${run_root}"
  env \
    CONDA_ENV=mapanything \
    RUN_ROOT="${run_root}" \
    SCENES="wayspots_bears wayspots_squarebench" \
    METHODS=stage1 \
    GPU_0=0 \
    GPU_1=1 \
    SKIP_EXISTING=false \
    CONTINUE_ON_ERROR=false \
    LMC_FUSION_REFINEMENT_MODE="${refinement_mode}" \
    LMC_FUSION_REREAD_DELTA_ALPHA=0.10 \
    LMC_FUSION_REREAD_SCALAR_GATE=True \
    LMC_FUSION_REREAD_GATE_INIT=-2.0 \
    LMC_FUSION_REREAD_POST_NORM=False \
    LMC_FUSION_REREAD_TRUST_REGION_RATIO=0.0 \
    LMC_FUSION_REREAD_TEMPERATURE=1.0 \
    LMC_FUSION_ASSEMBLY_GAMMA_INIT=0.0 \
    S1_LOSS_STEP_MODE=fixed_zero \
    S1_EARLY_STOP=False \
    LMC_LOG_RUNTIME_STATS=True \
    LMC_RUNTIME_STATS_INTERVAL=100 \
    LMC_RUNTIME_STATS_MAX_PIXELS=4096 \
    NUM_DATA_LOADER_WORKERS=12 \
    EVAL_NUM_WORKERS=6 \
    EVAL_DETERMINISTIC=True \
    EVAL_DSACSTAR_SEED=1305 \
    EVAL_DSACSTAR_SEED_PER_FRAME=True \
    ITERATION_EVAL_SEED=1305 \
    POST_TRAIN_HYPOTHESES=256 \
    POST_TRAIN_EVAL_SEEDS="1305 2026 4242" \
    OMP_NUM_THREADS=8 \
    OMP_DYNAMIC=FALSE \
    bash "${RUNNER}" 2>&1 | tee "${RUN_ROOT_BASE}/logs/${variant}.runner.log"
  echo "[$(date +%Y-%m-%dT%H:%M:%S)] done ${variant}"
}

mkdir -p "${RUN_ROOT_BASE}"
cd "${ROOT_DIR}"

run_variant pmrf_base progressive_reread
run_variant centered_reread centered_reread

echo "[$(date +%Y-%m-%dT%H:%M:%S)] all done: ${RUN_ROOT_BASE}"
