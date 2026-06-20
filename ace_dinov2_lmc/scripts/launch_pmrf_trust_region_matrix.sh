#!/usr/bin/env bash
set -euo pipefail

cd /home/xwh/project/ace_depth

MATRIX_ROOT="${MATRIX_ROOT:-/data/xwh/ace_dinov2_lmc/04_evaluation/pmrf_trust_region_matrix_$(date '+%Y%m%d_%H%M%S')}"
BEARS_MEM="${BEARS_MEM:-/data/xwh/ace_dinov2_lmc/04_evaluation/pmrf_nonorm_bears_test_20260617/train/20260617_110800/single_repeat/memory/wayspots_bears/memory_ace_fcn_sparse_sp_r4.pt}"
SQ_MEM="${SQ_MEM:-/data/xwh/ace_dinov2_lmc/04_evaluation/wayspots_ace_fcn_lmc_suite/20260530_181745/memory/wayspots_squarebench/memory_ace_fcn_sparse_sp_r4.pt}"

run_variant() {
  local name="$1"
  local trust_ratio="$2"
  local run_root="${MATRIX_ROOT}/${name}"

  mkdir -p "${run_root}/memory/wayspots_bears" "${run_root}/memory/wayspots_squarebench"
  ln -sf "${BEARS_MEM}" "${run_root}/memory/wayspots_bears/memory_ace_fcn_sparse_sp_r4.pt"
  ln -sf "${SQ_MEM}" "${run_root}/memory/wayspots_squarebench/memory_ace_fcn_sparse_sp_r4.pt"

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] START ${name} trust_ratio=${trust_ratio}"
  env \
    CONDA_ENV=mapanything \
    RUN_ROOT="${run_root}" \
    SCENES="wayspots_bears wayspots_squarebench" \
    METHODS=stage1 \
    GPU_0=0 \
    GPU_1=2 \
    SKIP_EXISTING=false \
    CONTINUE_ON_ERROR=false \
    LMC_FUSION_REFINEMENT_MODE=progressive_reread \
    LMC_FUSION_REREAD_DELTA_ALPHA=0.10 \
    LMC_FUSION_REREAD_SCALAR_GATE=True \
    LMC_FUSION_REREAD_GATE_INIT=-2.0 \
    LMC_FUSION_REREAD_POST_NORM=False \
    LMC_FUSION_REREAD_TRUST_REGION_RATIO="${trust_ratio}" \
    LMC_FUSION_REREAD_TEMPERATURE=1.0 \
    LMC_FUSION_ASSEMBLY_GAMMA_INIT=0.0 \
    S1_LOSS_STEP_MODE=fixed_zero \
    S1_EARLY_STOP=False \
    LMC_LOG_RUNTIME_STATS=True \
    LMC_RUNTIME_STATS_INTERVAL=100 \
    LMC_RUNTIME_STATS_MAX_PIXELS=4096 \
    NUM_DATA_LOADER_WORKERS=12 \
    EVAL_NUM_WORKERS=6 \
    bash ace_dinov2_lmc/scripts/run_wayspots_ace_fcn_lmc_suite.sh \
    2>&1 | tee "${MATRIX_ROOT}/${name}.runner.log"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE ${name}"
}

mkdir -p "${MATRIX_ROOT}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] MATRIX_ROOT ${MATRIX_ROOT}"
run_variant pmrf_tr05 0.5
run_variant pmrf_tr025 0.25
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ALL_DONE ${MATRIX_ROOT}"
