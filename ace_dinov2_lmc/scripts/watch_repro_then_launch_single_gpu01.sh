#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/xwh/project/ace_depth"
SESSION="${SESSION:-pmrf_repro_base_vs_centered_c0_all8_20260622_gpu01}"
RUN_ROOT_BASE="${RUN_ROOT_BASE:-/data/xwh/ace_dinov2_lmc/04_evaluation/pmrf_repro_base_vs_centered_c0_all8_20260622_gpu01}"
POLL_SECONDS="${POLL_SECONDS:-300}"
LOG_DIR="${RUN_ROOT_BASE}/logs"
WATCH_LOG="${LOG_DIR}/single_watcher.log"
SINGLE_LOG="${LOG_DIR}/single.tmux.log"

mkdir -p "${LOG_DIR}"

log() {
  echo "[$(date +%Y-%m-%dT%H:%M:%S)] $*" | tee -a "${WATCH_LOG}"
}

variant_done() {
  local variant="$1"
  grep -q "variant done ${variant}" "${LOG_DIR}/${variant}.tmux.log" 2>/dev/null
}

variant_failed() {
  local variant="$1"
  [[ -s "${LOG_DIR}/${variant}.failed" ]] && return 0
  grep -Eq "variant finished with failures ${variant}|failed variant=${variant}|Traceback|RuntimeError|CUDA out of memory|No space left on device"     "${LOG_DIR}/${variant}.tmux.log" 2>/dev/null
}

single_started() {
  tmux list-windows -t "${SESSION}" -F '#W' 2>/dev/null | grep -qx "single" && return 0
  [[ -s "${SINGLE_LOG}" ]] && return 0
  return 1
}

launch_single() {
  log "launching single on GPU0"
  tmux new-window -t "${SESSION}" -n single     "cd ${ROOT_DIR} && RUN_ROOT_BASE=${RUN_ROOT_BASE} VARIANT=single bash ace_dinov2_lmc/scripts/launch_repro_pmrf_base_vs_centered_c0_all8_gpu01.sh 2>&1 | tee ${SINGLE_LOG}"
}

log "watcher start session=${SESSION} run_root=${RUN_ROOT_BASE} poll=${POLL_SECONDS}s"
while true; do
  if variant_failed pmrf_base; then
    log "pmrf_base failed; not launching single"
    exit 1
  fi
  if variant_failed centered_c0; then
    log "centered_c0 failed; not launching single"
    exit 1
  fi

  if variant_done pmrf_base && variant_done centered_c0; then
    if single_started; then
      log "single already started; watcher exiting"
      exit 0
    fi
    launch_single
    log "single launched; watcher exiting"
    exit 0
  fi

  log "waiting: pmrf_base_done=$(variant_done pmrf_base && echo yes || echo no) centered_c0_done=$(variant_done centered_c0 && echo yes || echo no)"
  sleep "${POLL_SECONDS}"
done
