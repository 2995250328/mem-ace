#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/xwh/project/ace_depth}"
REPO_ROOT="${REPO_ROOT:-${ROOT_DIR}/ace_dinov2_lmc}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
PYTHON_BIN="${PYTHON_BIN:-/home/xwh/miniforge3/envs/mapanything/bin/python}"
GPUS_STR="${GPUS_STR:-2 3}"
SCENES_STR="${SCENES_STR:-scene2a scene5}"
SIDECAR_ROOT="${SIDECAR_ROOT:-/data/xwh/ace_dinov2_lmc/04_evaluation/indoor6/dino_stgs_matrix/20260629_203557_it2_buf10M_pair_gpu23/sidecars}"

SMOKE_ROOT="${SMOKE_ROOT:-/data/xwh/ace_dinov2_lmc/04_evaluation/indoor6/dino_stgs_interframe_smoke/${STAMP}_it1_buf1M_pair_gpu23}"
FULL_ROOT="${FULL_ROOT:-/data/xwh/ace_dinov2_lmc/04_evaluation/indoor6/dino_stgs_interframe/${STAMP}_it2_buf10M_pair_gpu23}"
SMOKE_WATCHDOG_STALE_SEC="${SMOKE_WATCHDOG_STALE_SEC:-900}"
FULL_WATCHDOG_STALE_SEC="${FULL_WATCHDOG_STALE_SEC:-1800}"
WATCHDOG_CHECK_SEC="${WATCHDOG_CHECK_SEC:-60}"
FULL_INTERFRAME_VARIANTS="${FULL_INTERFRAME_VARIANTS:-guided,w001_s2g,w002_s2g,w001_s2}"

mkdir -p "${SMOKE_ROOT}" "${FULL_ROOT}"
MAIN_LOG="${FULL_ROOT}/smoke_then_full.log"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "${MAIN_LOG}"
}

run_matrix() {
  local run_root="$1"
  shift
  RUN_ROOT="${run_root}" \
  ROOT_DIR="${ROOT_DIR}" \
  REPO_ROOT="${REPO_ROOT}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  GPUS_STR="${GPUS_STR}" \
  SCENES_STR="${SCENES_STR}" \
  SIDECAR_ROOT="${SIDECAR_ROOT}" \
  bash "${REPO_ROOT}/scripts/launch_indoor6_dino_stgs_interframe_matrix_gpu23.sh" "$@"
}

validate_smoke() {
  local root="$1"
  if [ ! -s "${root}/metricwise_best.tsv" ]; then
    log "[smoke fail] missing metricwise_best.tsv under ${root}"
    return 1
  fi
  if rg -q 'stgsTerms=[1-9][0-9]*|\[STGS\] active' "${root}"; then
    log "[smoke ok] inter-frame terms detected under ${root}"
    return 0
  fi
  log "[smoke fail] no positive stgsTerms / [STGS] active found under ${root}"
  return 1
}

log "Smoke root: ${SMOKE_ROOT}"
log "Full root : ${FULL_ROOT}"
log "Scenes    : ${SCENES_STR}; GPUs=${GPUS_STR}; sidecars=${SIDECAR_ROOT}"

log "[smoke start] 1 iter, 1M buffer, variant=w001_s2g"
INTERFRAME_VARIANTS="w001_s2g" \
LMC_ITERATIONS="1" \
TRAINING_BUFFER_SIZE="1000000" \
BUFFER_SIZE_FINAL="1000000" \
POST_TRAIN_SEEDS_STR="1305" \
POST_TRAIN_HYPOTHESES="64" \
WATCHDOG_STALE_SEC="${SMOKE_WATCHDOG_STALE_SEC}" \
WATCHDOG_CHECK_SEC="${WATCHDOG_CHECK_SEC}" \
run_matrix "${SMOKE_ROOT}"

validate_smoke "${SMOKE_ROOT}"

log "[full start] 2 iter, 10M buffer, variants=${FULL_INTERFRAME_VARIANTS}"
INTERFRAME_VARIANTS="${FULL_INTERFRAME_VARIANTS}" \
LMC_ITERATIONS="2" \
TRAINING_BUFFER_SIZE="10000000" \
BUFFER_SIZE_FINAL="10000000" \
POST_TRAIN_SEEDS_STR="1305 2026 4242 7777 9001" \
POST_TRAIN_HYPOTHESES="256" \
WATCHDOG_STALE_SEC="${FULL_WATCHDOG_STALE_SEC}" \
WATCHDOG_CHECK_SEC="${WATCHDOG_CHECK_SEC}" \
run_matrix "${FULL_ROOT}"

log "[done] smoke_then_full finished"
