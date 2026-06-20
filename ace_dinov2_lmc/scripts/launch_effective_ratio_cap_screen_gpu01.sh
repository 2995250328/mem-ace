#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/xwh/project/ace_depth"
RUNNER="${ROOT_DIR}/ace_dinov2_lmc/scripts/run_wayspots_ace_fcn_lmc_suite.sh"
RUN_ROOT_BASE="${RUN_ROOT_BASE:-/data/xwh/ace_dinov2_lmc/04_evaluation/pmrf_effective_ratio_cap_screen_20260620_gpu01_v2}"
MEMORY_ROOT="${MEMORY_ROOT:-/data/xwh/ace_dinov2_lmc/04_evaluation/wayspots_ace_fcn_lmc_suite/20260530_181745/memory}"
DRY_RUN="${DRY_RUN:-false}"

SCENES=(wayspots_squarebench wayspots_cubes)
VARIANTS=(cap006 cap008)
CAPS=(0.006 0.008)
GPUS=(0 1)

prepare_memory_link() {
  local run_root="$1"
  local scene="$2"
  local src="${MEMORY_ROOT}/${scene}/memory_ace_fcn_sparse_sp_r4.pt"
  local dst_dir="${run_root}/memory/${scene}"
  if [[ ! -f "${src}" ]]; then
    echo "missing memory: ${src}" >&2
    return 2
  fi
  mkdir -p "${dst_dir}"
  ln -sfn "${src}" "${dst_dir}/memory_ace_fcn_sparse_sp_r4.pt"
}

run_scene_variant() {
  local scene="$1"
  local variant="$2"
  local cap="$3"
  local gpu="$4"
  local run_root="${RUN_ROOT_BASE}/${variant}/${scene}"
  local log_dir="${RUN_ROOT_BASE}/logs/${variant}"

  echo "[$(date +%Y-%m-%dT%H:%M:%S)] scene=${scene} variant=${variant} gpu=${gpu} effective_ratio_cap=${cap} run_root=${run_root}"
  if [[ "${DRY_RUN}" == "true" ]]; then
    return 0
  fi

  mkdir -p "${log_dir}"
  prepare_memory_link "${run_root}" "${scene}"
  env \
    CONDA_ENV=mapanything \
    RUN_ROOT="${run_root}" \
    SCENES="${scene}" \
    METHODS=stage1 \
    GPU_0="${gpu}" \
    GPU_1="${gpu}" \
    SKIP_EXISTING=false \
    CONTINUE_ON_ERROR=false \
    LMC_FUSION_REFINEMENT_MODE=centered_reread \
    LMC_FUSION_REREAD_DELTA_ALPHA=0.10 \
    LMC_FUSION_REREAD_SCALAR_GATE=True \
    LMC_FUSION_REREAD_GATE_INIT=-2.0 \
    LMC_FUSION_REREAD_POST_NORM=False \
    LMC_FUSION_REREAD_TRUST_REGION_RATIO=0.0 \
    LMC_FUSION_REREAD_TEMPERATURE=1.0 \
    LMC_FUSION_REREAD_COMMON_SCALE=0.0 \
    LMC_FUSION_REREAD_EFFECTIVE_RATIO_CAP="${cap}" \
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
    bash "${RUNNER}" 2>&1 | tee "${log_dir}/${scene}.runner.log"
}

run_scene_pair() {
  local scene="$1"
  local pids=()
  local idx
  echo "[$(date +%Y-%m-%dT%H:%M:%S)] pair begin scene=${scene}"
  for idx in "${!VARIANTS[@]}"; do
    run_scene_variant "${scene}" "${VARIANTS[$idx]}" "${CAPS[$idx]}" "${GPUS[$idx]}" &
    pids+=("$!")
  done
  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if [[ "${failed}" != 0 ]]; then
    echo "pair failed: ${scene}" >&2
    return 1
  fi
  echo "[$(date +%Y-%m-%dT%H:%M:%S)] pair done scene=${scene}"
}

if [[ "${DRY_RUN}" != "true" ]]; then
  mkdir -p "${RUN_ROOT_BASE}/logs"
fi
cd "${ROOT_DIR}"
for scene in "${SCENES[@]}"; do
  run_scene_pair "${scene}"
done

echo "[$(date +%Y-%m-%dT%H:%M:%S)] screen done: ${RUN_ROOT_BASE}"
