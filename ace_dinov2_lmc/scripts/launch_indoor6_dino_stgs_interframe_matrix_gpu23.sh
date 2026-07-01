#!/usr/bin/env bash
set -euo pipefail

# Indoor6 DINOv2 + MapAnything-memory STGS inter-frame loss matrix on GPUs 2/3.
# Pair-only short validation before any full-scene run.

ROOT_DIR="${ROOT_DIR:-/home/xwh/project/ace_depth}"
REPO_ROOT="${REPO_ROOT:-${ROOT_DIR}/ace_dinov2_lmc}"
PYTHON_BIN="${PYTHON_BIN:-/home/xwh/miniforge3/envs/mapanything/bin/python}"
GPUS_STR="${GPUS_STR:-2 3}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-/data/xwh/ace_dinov2_lmc/04_evaluation/indoor6/dino_stgs_interframe/${STAMP}_it2_buf10M_pair_gpu23}"
SIDECAR_ROOT="${SIDECAR_ROOT:-/data/xwh/ace_dinov2_lmc/04_evaluation/indoor6/dino_stgs_matrix/20260629_203557_it2_buf10M_pair_gpu23/sidecars}"
MATRIX_LOG="${RUN_ROOT}/matrix.log"

SCENES_STR="${SCENES_STR:-scene2a scene5}"
LMC_ITERATIONS="${LMC_ITERATIONS:-2}"
TRAINING_BUFFER_SIZE="${TRAINING_BUFFER_SIZE:-10000000}"
BUFFER_SIZE_FINAL="${BUFFER_SIZE_FINAL:-10000000}"
BUFFER_BATCH_SIZE="${BUFFER_BATCH_SIZE:-1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-10240}"
SAMPLES_PER_IMAGE="${SAMPLES_PER_IMAGE:-384}"
POST_TRAIN_SEEDS_STR="${POST_TRAIN_SEEDS_STR:-1305 2026 4242 7777 9001}"
POST_TRAIN_HYPOTHESES="${POST_TRAIN_HYPOTHESES:-256}"
SFM_TRACK_GUIDED_FRACTION="${SFM_TRACK_GUIDED_FRACTION:-0.10}"
SFM_TRACK_INTER_FRAME_START_RATIO="${SFM_TRACK_INTER_FRAME_START_RATIO:-0.2}"
SFM_TRACK_INTER_FRAME_DECAY_LAST_RATIO="${SFM_TRACK_INTER_FRAME_DECAY_LAST_RATIO:-0.3}"
SFM_TRACK_INTER_FRAME_MAX_PX="${SFM_TRACK_INTER_FRAME_MAX_PX:-100.0}"
SFM_TRACK_INTER_FRAME_DROPOUT="${SFM_TRACK_INTER_FRAME_DROPOUT:-0.5}"
DRY_RUN="${DRY_RUN:-False}"
PREP_ONLY="${PREP_ONLY:-False}"
INTERFRAME_VARIANTS="${INTERFRAME_VARIANTS:-guided,w001_s2g,w002_s2g,w001_s2}"
WATCHDOG_ENABLED="${WATCHDOG_ENABLED:-True}"
WATCHDOG_STALE_SEC="${WATCHDOG_STALE_SEC:-1800}"
WATCHDOG_CHECK_SEC="${WATCHDOG_CHECK_SEC:-60}"
WATCHDOG_PID=""

mkdir -p "${RUN_ROOT}"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "${MATRIX_LOG}"
}

bool_true() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

should_run_variant() {
  local key="$1"
  [[ ",${INTERFRAME_VARIANTS}," == *",${key},"* ]]
}

start_watchdog() {
  if bool_true "${WATCHDOG_ENABLED}" && ! bool_true "${DRY_RUN}" && ! bool_true "${PREP_ONLY}"; then
    bash "${REPO_ROOT}/scripts/watch_runroot_stall.sh" "${RUN_ROOT}" "${WATCHDOG_STALE_SEC}" "${WATCHDOG_CHECK_SEC}" &
    WATCHDOG_PID="$!"
    log "Watchdog: enabled pid=${WATCHDOG_PID} stale_sec=${WATCHDOG_STALE_SEC} check_sec=${WATCHDOG_CHECK_SEC}"
  else
    log "Watchdog: disabled"
  fi
}

stop_watchdog() {
  if [ -n "${WATCHDOG_PID}" ] && kill -0 "${WATCHDOG_PID}" 2>/dev/null; then
    kill "${WATCHDOG_PID}" 2>/dev/null || true
    wait "${WATCHDOG_PID}" 2>/dev/null || true
  fi
}
trap stop_watchdog EXIT

run_variant() {
  local variant="$1"
  local guided_mode="$2"
  local inter_enabled="$3"
  local inter_weight="$4"
  local apply_to="$5"

  log "[variant start] ${variant} scenes=${SCENES_STR} mode=${guided_mode} inter=${inter_enabled} weight=${inter_weight} apply=${apply_to}"
  VARIANT_LABEL="${variant}" \
  SCENES_STR="${SCENES_STR}" \
  GPUS_STR="${GPUS_STR}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  RUN_ROOT="${RUN_ROOT}" \
  SIDECAR_ROOT="${SIDECAR_ROOT}" \
  LMC_ITERATIONS="${LMC_ITERATIONS}" \
  TRAINING_BUFFER_SIZE="${TRAINING_BUFFER_SIZE}" \
  BUFFER_SIZE_FINAL="${BUFFER_SIZE_FINAL}" \
  BUFFER_BATCH_SIZE="${BUFFER_BATCH_SIZE}" \
  TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE}" \
  SAMPLES_PER_IMAGE="${SAMPLES_PER_IMAGE}" \
  POST_TRAIN_SEEDS_STR="${POST_TRAIN_SEEDS_STR}" \
  POST_TRAIN_HYPOTHESES="${POST_TRAIN_HYPOTHESES}" \
  USE_STGS_GUIDED_SAMPLING="True" \
  USE_STGS_INTER_FRAME_LOSS="${inter_enabled}" \
  SFM_TRACK_GUIDED_MODE="${guided_mode}" \
  SFM_TRACK_GUIDED_FRACTION="${SFM_TRACK_GUIDED_FRACTION}" \
  SFM_TRACK_INTER_FRAME_WEIGHT="${inter_weight}" \
  SFM_TRACK_INTER_FRAME_APPLY_TO="${apply_to}" \
  SFM_TRACK_INTER_FRAME_START_RATIO="${SFM_TRACK_INTER_FRAME_START_RATIO}" \
  SFM_TRACK_INTER_FRAME_DECAY_LAST_RATIO="${SFM_TRACK_INTER_FRAME_DECAY_LAST_RATIO}" \
  SFM_TRACK_INTER_FRAME_MAX_PX="${SFM_TRACK_INTER_FRAME_MAX_PX}" \
  SFM_TRACK_INTER_FRAME_DROPOUT="${SFM_TRACK_INTER_FRAME_DROPOUT}" \
  DRY_RUN="${DRY_RUN}" \
  PREP_ONLY="${PREP_ONLY}" \
  bash "${REPO_ROOT}/scripts/launch_indoor6_dino_stgs_short_gpu01.sh" 2>&1 | tee -a "${MATRIX_LOG}"
  log "[variant done] ${variant}"
}

log "Run root: ${RUN_ROOT}"
log "Sidecars: ${SIDECAR_ROOT}"
log "GPUs: ${GPUS_STR}; scenes=${SCENES_STR}; iters=${LMC_ITERATIONS}; buffer=${TRAINING_BUFFER_SIZE}/${BUFFER_SIZE_FINAL}; guided_fraction=${SFM_TRACK_GUIDED_FRACTION}"
log "Inter-frame schedule: start=${SFM_TRACK_INTER_FRAME_START_RATIO}, decay=${SFM_TRACK_INTER_FRAME_DECAY_LAST_RATIO}, max_px=${SFM_TRACK_INTER_FRAME_MAX_PX}, dropout=${SFM_TRACK_INTER_FRAME_DROPOUT}"
log "Variants: ${INTERFRAME_VARIANTS}"
start_watchdog

if should_run_variant "guided"; then
  run_variant "it${LMC_ITERATIONS}_buf${TRAINING_BUFFER_SIZE}_guided_f010" "anchor_only" "False" "0.0" "stage2"
fi
if should_run_variant "w001_s2g"; then
  run_variant "it${LMC_ITERATIONS}_buf${TRAINING_BUFFER_SIZE}_if_w001_s2g" "inter_frame" "True" "0.01" "stage2_g"
fi
if should_run_variant "w002_s2g"; then
  run_variant "it${LMC_ITERATIONS}_buf${TRAINING_BUFFER_SIZE}_if_w002_s2g" "inter_frame" "True" "0.02" "stage2_g"
fi
if should_run_variant "w003_s2g"; then
  run_variant "it${LMC_ITERATIONS}_buf${TRAINING_BUFFER_SIZE}_if_w003_s2g" "inter_frame" "True" "0.03" "stage2_g"
fi
if should_run_variant "w005_s2g"; then
  run_variant "it${LMC_ITERATIONS}_buf${TRAINING_BUFFER_SIZE}_if_w005_s2g" "inter_frame" "True" "0.05" "stage2_g"
fi
if should_run_variant "w001_s2"; then
  run_variant "it${LMC_ITERATIONS}_buf${TRAINING_BUFFER_SIZE}_if_w001_s2" "inter_frame" "True" "0.01" "stage2"
fi

if ! bool_true "${DRY_RUN}" && ! bool_true "${PREP_ONLY}"; then
  log "[aggregate final] ${RUN_ROOT}"
  bash "${REPO_ROOT}/scripts/aggregate_lmc_metricwise_best.sh" "${RUN_ROOT}" 2>&1 | tee -a "${MATRIX_LOG}"
fi

log "[matrix done] ${RUN_ROOT}"
