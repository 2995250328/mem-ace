#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <run_root> [stale_sec] [check_sec]" >&2
  exit 2
fi

RUN_ROOT="$1"
STALE_SEC="${2:-1800}"
CHECK_SEC="${3:-60}"
PATTERN="${WATCHDOG_PATTERN:-${RUN_ROOT}}"
LOG_FILE="${RUN_ROOT}/watchdog_stall.log"
mkdir -p "${RUN_ROOT}"

log() {
  printf "[%s] %s\n" "$(date "+%Y-%m-%d %H:%M:%S")" "$*" | tee -a "${LOG_FILE}"
}

matching_pids() {
  pgrep -af "${PATTERN}" 2>/dev/null \
    | awk -v self="$$" "
        \$1 == self { next }
        /watch_runroot_stall\\.sh/ { next }
        /pgrep -af/ { next }
        { print \$1 }
      " \
    | sort -n -u
}

latest_log_mtime() {
  if [ ! -d "${RUN_ROOT}" ]; then
    echo 0
    return 0
  fi
  find "${RUN_ROOT}" -type f \
    \( -name "*.log" -o -name "matrix.log" -o -name "training_full_log.txt" \) \
    ! -name "watchdog*.log" \
    -printf "%T@\n" 2>/dev/null \
    | sort -nr \
    | head -n 1 \
    | awk "{ printf \"%d\\n\", \$1 }"
}

terminate_pids() {
  local pids="$1"
  if [ -z "${pids}" ]; then
    return 0
  fi
  log "stale timeout reached; sending TERM to pids: ${pids}"
  # shellcheck disable=SC2086
  kill -TERM ${pids} 2>/dev/null || true
  sleep 30
  local survivors=""
  for pid in ${pids}; do
    if kill -0 "${pid}" 2>/dev/null; then
      survivors="${survivors} ${pid}"
    fi
  done
  if [ -n "${survivors}" ]; then
    log "pids still alive after TERM; sending KILL:${survivors}"
    # shellcheck disable=SC2086
    kill -KILL ${survivors} 2>/dev/null || true
  fi
}

log "watchdog start: run_root=${RUN_ROOT} pattern=${PATTERN} stale_sec=${STALE_SEC} check_sec=${CHECK_SEC}"
start_time="$(date +%s)"
while true; do
  pids="$(matching_pids || true)"
  if [ -z "${pids}" ]; then
    log "no matching pids remain; watchdog exit"
    exit 0
  fi

  latest="$(latest_log_mtime)"
  if [ -z "${latest}" ] || [ "${latest}" -le 0 ]; then
    latest="${start_time}"
  fi
  now="$(date +%s)"
  age=$((now - latest))
  if [ "${age}" -ge "${STALE_SEC}" ]; then
    log "no non-watchdog log update for ${age}s; latest_log_mtime=${latest}; active_pids=$(echo ${pids})"
    terminate_pids "${pids}"
    exit 124
  fi
  sleep "${CHECK_SEC}"
done
