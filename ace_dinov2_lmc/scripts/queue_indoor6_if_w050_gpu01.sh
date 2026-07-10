#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/xwh/project/ace_depth}"
REPO_ROOT="${REPO_ROOT:-${ROOT_DIR}/ace_dinov2_lmc}"
PYTHON_BIN="${PYTHON_BIN:-/home/xwh/miniforge3/envs/mapanything/bin/python}"
GPUS_STR="${GPUS_STR:-0 1}"
SCENES_STR="${SCENES_STR:-scene2a scene5}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-/data/xwh/ace_dinov2_lmc/04_evaluation/indoor6/dino_stgs_interframe/${STAMP}_it2_buf10M_pair_w050_gpu01}"
SIDECAR_ROOT="${SIDECAR_ROOT:-/data/xwh/ace_dinov2_lmc/04_evaluation/indoor6/dino_stgs_matrix/20260629_203557_it2_buf10M_pair_gpu23/sidecars}"
QUEUE_LOG="${RUN_ROOT}/queue.log"
POLL_SEC="${POLL_SEC:-300}"
GPU_IDLE_GRACE_SEC="${GPU_IDLE_GRACE_SEC:-120}"
WATCHDOG_STALE_SEC="${WATCHDOG_STALE_SEC:-1800}"
WATCHDOG_CHECK_SEC="${WATCHDOG_CHECK_SEC:-60}"

mkdir -p "${RUN_ROOT}"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "${QUEUE_LOG}"
}

gpu_compute_pids() {
  local gpu="$1"
  nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | sed '/^[[:space:]]*$/d' \
    | tr '\n' ' ' \
    || true
}

busy_summary() {
  local out=""
  for gpu in ${GPUS_STR}; do
    local pids
    pids="$(gpu_compute_pids "${gpu}")"
    if [ -n "${pids}" ]; then
      out="${out} gpu${gpu}:${pids}"
    fi
  done
  printf '%s' "${out}"
}

wait_for_gpu_idle() {
  local idle_since=0
  log "queue start: wait for GPUs ${GPUS_STR}; poll=${POLL_SEC}s idle_grace=${GPU_IDLE_GRACE_SEC}s"
  while true; do
    local busy
    busy="$(busy_summary)"
    local now
    now="$(date +%s)"
    if [ -z "${busy}" ]; then
      if [ "${idle_since}" -eq 0 ]; then
        idle_since="${now}"
        log "GPUs ${GPUS_STR} look idle; waiting grace window"
      fi
      if [ $((now - idle_since)) -ge "${GPU_IDLE_GRACE_SEC}" ]; then
        log "GPUs ${GPUS_STR} idle for ${GPU_IDLE_GRACE_SEC}s; launching training"
        return 0
      fi
    else
      idle_since=0
      log "GPUs busy:${busy}; sleep ${POLL_SEC}s"
    fi
    sleep "${POLL_SEC}"
  done
}

main() {
  cd "${REPO_ROOT}"
  wait_for_gpu_idle

  bash "${REPO_ROOT}/scripts/watch_runroot_stall.sh" "${RUN_ROOT}" "${WATCHDOG_STALE_SEC}" "${WATCHDOG_CHECK_SEC}" &
  local watchdog_pid="$!"
  trap 'kill "${watchdog_pid}" 2>/dev/null || true; wait "${watchdog_pid}" 2>/dev/null || true' EXIT

  log "training start: run_root=${RUN_ROOT} variant=it2_buf10000000_if_w050_s2g gpus=${GPUS_STR}"
  VARIANT_LABEL="it2_buf10000000_if_w050_s2g" \
  SCENES_STR="${SCENES_STR}" \
  GPUS_STR="${GPUS_STR}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  RUN_ROOT="${RUN_ROOT}" \
  SIDECAR_ROOT="${SIDECAR_ROOT}" \
  LMC_ITERATIONS=2 \
  TRAINING_BUFFER_SIZE=10000000 \
  BUFFER_SIZE_FINAL=10000000 \
  BUFFER_BATCH_SIZE=1 \
  TRAIN_BATCH_SIZE=10240 \
  SAMPLES_PER_IMAGE=384 \
  POST_TRAIN_SEEDS_STR="1305 2026 4242 7777 9001" \
  POST_TRAIN_HYPOTHESES=256 \
  USE_STGS_GUIDED_SAMPLING=True \
  USE_STGS_INTER_FRAME_LOSS=True \
  SFM_TRACK_GUIDED_MODE=inter_frame \
  SFM_TRACK_GUIDED_FRACTION=0.10 \
  SFM_TRACK_INTER_FRAME_WEIGHT=0.5 \
  SFM_TRACK_INTER_FRAME_APPLY_TO=stage2_g \
  SFM_TRACK_INTER_FRAME_START_RATIO=0.2 \
  SFM_TRACK_INTER_FRAME_DECAY_LAST_RATIO=0.3 \
  SFM_TRACK_INTER_FRAME_MAX_PX=100.0 \
  SFM_TRACK_INTER_FRAME_DROPOUT=0.5 \
  bash "${REPO_ROOT}/scripts/launch_indoor6_dino_stgs_short_gpu01.sh" 2>&1 | tee -a "${QUEUE_LOG}"
  log "training command finished"
}

main "$@"
