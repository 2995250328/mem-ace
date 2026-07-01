#!/usr/bin/env bash
set -euo pipefail

# Indoor6 DINOv2 + MapAnything-memory matrix on GPUs 2/3.
# Matrix:
#   1) fair DINO baseline with the same post-train multi-seed eval,
#   2) STGS f=0.10 on scene2a/scene5,
#   3) f sweep on scene2a/scene5 with f=0.05 and f=0.15.
# Full-scene Indoor6 validation is intentionally deferred until the final setup is fixed.

ROOT_DIR="${ROOT_DIR:-/home/xwh/project/ace_depth}"
REPO_ROOT="${REPO_ROOT:-${ROOT_DIR}/ace_dinov2_lmc}"
PYTHON_BIN="${PYTHON_BIN:-/home/xwh/miniforge3/envs/mapanything/bin/python}"
GPUS_STR="${GPUS_STR:-2 3}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-/data/xwh/ace_dinov2_lmc/04_evaluation/indoor6/dino_stgs_matrix/${STAMP}_it2_buf10M_gpu23}"
MATRIX_LOG="${RUN_ROOT}/matrix.log"

# User-requested short validation budget.
LMC_ITERATIONS="${LMC_ITERATIONS:-2}"
TRAINING_BUFFER_SIZE="${TRAINING_BUFFER_SIZE:-10000000}"
BUFFER_SIZE_FINAL="${BUFFER_SIZE_FINAL:-10000000}"
BUFFER_BATCH_SIZE="${BUFFER_BATCH_SIZE:-1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-10240}"
SAMPLES_PER_IMAGE="${SAMPLES_PER_IMAGE:-384}"
POST_TRAIN_SEEDS_STR="${POST_TRAIN_SEEDS_STR:-1305 2026 4242 7777 9001}"
POST_TRAIN_HYPOTHESES="${POST_TRAIN_HYPOTHESES:-256}"
DRY_RUN="${DRY_RUN:-False}"
PREP_ONLY="${PREP_ONLY:-False}"

PAIR_SCENES="${PAIR_SCENES:-scene2a scene5}"

mkdir -p "${RUN_ROOT}"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "${MATRIX_LOG}"
}

run_variant() {
  local variant="$1"
  local scenes="$2"
  local guided="$3"
  local frac="$4"

  log "[variant start] ${variant} scenes=${scenes} guided=${guided} f=${frac}"
  VARIANT_LABEL="${variant}" \
  SCENES_STR="${scenes}" \
  GPUS_STR="${GPUS_STR}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  RUN_ROOT="${RUN_ROOT}" \
  LMC_ITERATIONS="${LMC_ITERATIONS}" \
  TRAINING_BUFFER_SIZE="${TRAINING_BUFFER_SIZE}" \
  BUFFER_SIZE_FINAL="${BUFFER_SIZE_FINAL}" \
  BUFFER_BATCH_SIZE="${BUFFER_BATCH_SIZE}" \
  TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE}" \
  SAMPLES_PER_IMAGE="${SAMPLES_PER_IMAGE}" \
  POST_TRAIN_SEEDS_STR="${POST_TRAIN_SEEDS_STR}" \
  POST_TRAIN_HYPOTHESES="${POST_TRAIN_HYPOTHESES}" \
  USE_STGS_GUIDED_SAMPLING="${guided}" \
  USE_STGS_INTER_FRAME_LOSS="False" \
  SFM_TRACK_GUIDED_FRACTION="${frac}" \
  DRY_RUN="${DRY_RUN}" \
  PREP_ONLY="${PREP_ONLY}" \
  bash "${REPO_ROOT}/scripts/launch_indoor6_dino_stgs_short_gpu01.sh" 2>&1 | tee -a "${MATRIX_LOG}"
  log "[variant done] ${variant}"
}

log "Run root: ${RUN_ROOT}"
log "GPUs: ${GPUS_STR}; iters=${LMC_ITERATIONS}; buffer=${TRAINING_BUFFER_SIZE}/${BUFFER_SIZE_FINAL}; post_train_seeds=${POST_TRAIN_SEEDS_STR}"

run_variant "it2_buf10M_baseline" "${PAIR_SCENES}" "False" "0.00"
run_variant "it2_buf10M_stgs_f010" "${PAIR_SCENES}" "True" "0.10"
run_variant "it2_buf10M_stgs_f005" "${PAIR_SCENES}" "True" "0.05"
run_variant "it2_buf10M_stgs_f015" "${PAIR_SCENES}" "True" "0.15"

if [[ "${DRY_RUN,,}" != "true" && "${DRY_RUN}" != "True" && "${PREP_ONLY,,}" != "true" && "${PREP_ONLY}" != "True" ]]; then
  log "[aggregate final] ${RUN_ROOT}"
  bash "${REPO_ROOT}/scripts/aggregate_lmc_metricwise_best.sh" "${RUN_ROOT}" 2>&1 | tee -a "${MATRIX_LOG}"
fi

log "[matrix done] ${RUN_ROOT}"
