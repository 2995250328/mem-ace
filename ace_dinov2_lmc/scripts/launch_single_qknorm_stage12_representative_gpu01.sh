#!/usr/bin/env bash
set -euo pipefail

# Full two-stage representative-scene run for the single first-read fusion trick.
# This intentionally keeps all baseline settings identical except:
#   LMC_FUSION_REFINEMENT_MODE=single_qknorm_layerscale

ROOT_DIR="/home/xwh/project/ace_depth"
RUNNER="${ROOT_DIR}/ace_dinov2_lmc/scripts/run_wayspots_ace_fcn_lmc_suite.sh"
RUN_ROOT="${RUN_ROOT:-/data/xwh/ace_dinov2_lmc/04_evaluation/single_qknorm_stage12_representative_20260623_gpu01}"
MEMORY_ROOT="${MEMORY_ROOT:-/data/xwh/ace_dinov2_lmc/04_evaluation/wayspots_ace_fcn_lmc_suite/20260530_181745/memory}"

# Representative subset for future PMRF/fusion work:
# - Bears: small/indoor-like, single is strong and PMRF often regresses Acc10/median.
# - Squarebench: large scene where PMRF-base showed high-hypo gains.
# - Cubes/Tendrils: additional geometry/texture regimes from the 4-scene PMRF matrix.
SCENES=(
  wayspots_bears
  wayspots_squarebench
  wayspots_cubes
  wayspots_tendrils
)

prepare_memory_link() {
  local scene="$1"
  local src="${MEMORY_ROOT}/${scene}/memory_ace_fcn_sparse_sp_r4.pt"
  local dst_dir="${RUN_ROOT}/memory/${scene}"
  if [[ ! -f "${src}" ]]; then
    echo "missing memory: ${src}" >&2
    return 2
  fi
  mkdir -p "${dst_dir}"
  ln -sfn "${src}" "${dst_dir}/memory_ace_fcn_sparse_sp_r4.pt"
}

mkdir -p "${RUN_ROOT}/logs"
for scene in "${SCENES[@]}"; do
  prepare_memory_link "${scene}"
done

cd "${ROOT_DIR}"

echo "RUN_ROOT=${RUN_ROOT}"
echo "SCENES=${SCENES[*]}"
echo "GPUS=0,1"
echo "METHODS=stage1 stage2"
echo "config=single_qknorm_layerscale complete two-stage"

exec env \
  CONDA_ENV=mapanything \
  RUN_ROOT="${RUN_ROOT}" \
  SCENES="${SCENES[*]}" \
  METHODS="stage1 stage2" \
  GPU_0=0 \
  GPU_1=1 \
  SKIP_EXISTING=false \
  CONTINUE_ON_ERROR=false \
  LMC_FUSION_REFINEMENT_MODE=single_qknorm_layerscale \
  LMC_FUSION_REREAD_DELTA_ALPHA=1.0 \
  LMC_FUSION_REREAD_SCALAR_GATE=False \
  LMC_FUSION_REREAD_GATE_INIT=0.0 \
  LMC_FUSION_REREAD_POST_NORM=True \
  LMC_FUSION_REREAD_TRUST_REGION_RATIO=0.0 \
  LMC_FUSION_REREAD_TEMPERATURE=1.0 \
  LMC_FUSION_REREAD_COMMON_SCALE=1.0 \
  LMC_FUSION_REREAD_EFFECTIVE_RATIO_CAP=0.0 \
  LMC_FUSION_REREAD_QKNORM_EPS=1e-6 \
  LMC_FUSION_REREAD_QKNORM_TAU_INIT=0.0 \
  LMC_FUSION_REREAD_LAYERSCALE_PATCH_INIT=0.01 \
  LMC_FUSION_REREAD_LAYERSCALE_COMMON_INIT=0.0 \
  LMC_FUSION_SINGLE_QKNORM_EPS=1e-6 \
  LMC_FUSION_SINGLE_QKNORM_TAU_INIT=0.0 \
  LMC_FUSION_SINGLE_LAYERSCALE_INIT=1.0 \
  LMC_FUSION_REREAD_WARMUP_MODE=none \
  LMC_FUSION_REREAD_WARMUP_ITERS=0 \
  LMC_FUSION_REREAD_WARMUP_START=0.0 \
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
  ITERATION_EVAL_HYPOTHESES=256 \
  POST_TRAIN_HYPOTHESES=256 \
  POST_TRAIN_EVAL_SEEDS="1305 2026 4242" \
  OMP_NUM_THREADS=8 \
  OMP_DYNAMIC=FALSE \
  bash "${RUNNER}" 2>&1 | tee "${RUN_ROOT}/logs/runner.tmux.log"
