#!/usr/bin/env bash
set -euo pipefail

# Wait until the final sidecar queue is complete, then run strict inventory,
# final-matrix preflight, and launch the GPU worker queue.
#
# Intended use from /home/xwh/project/ace_depth:
#   tmux new-session -d -s final_full_autolaunch_20260705 \
#     "SIDECAR_RUN_ROOT=/data/... bash ace_dinov2_lmc/scripts/auto_launch_final_full_after_sidecars_20260705.sh"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/xwh/miniforge3/envs/mapanything/bin/python}"
POLL_SECONDS="${POLL_SECONDS:-120}"

SIDECAR_RUN_ROOT="${SIDECAR_RUN_ROOT:-/data/xwh/ace_dinov2_lmc/04_evaluation/shared/memory/stgs_sidecars/20260705_final_sidecars_final_all_value_raw_auto_ifw005_gpu01_expandable}"
FINAL_STAMP="${FINAL_STAMP:-20260705_final_full}"
GPUS_STR="${GPUS_STR:-0 1}"
FINAL_RUN_ROOT="${FINAL_RUN_ROOT:-/data/xwh/ace_dinov2_lmc/04_evaluation/shared/stage2/final_full/${FINAL_STAMP}_all_value_raw_auto_ifw005_it10_buf10m_h256_gpu01_expandable}"

SIDECAR_SCRIPT="${REPO_ROOT}/scripts/prepare_final_full_sidecars_20260705.sh"
FINAL_SCRIPT="${REPO_ROOT}/scripts/launch_final_full_matrix_20260705.sh"
MATRIX="${SIDECAR_RUN_ROOT}/matrix.tsv"

cd "${PROJECT_ROOT}"

log() {
  printf '[%(%F %T)T] %s\n' -1 "$*"
}

check_sidecars_once() {
  "${PYTHON_BIN}" - "${MATRIX}" <<'PY'
import csv
import sys
from collections import Counter

matrix = sys.argv[1]
try:
    with open(matrix, newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
except FileNotFoundError:
    print(f"missing matrix: {matrix}")
    sys.exit(4)

counts = Counter(r.get("status", "") for r in rows)
print("sidecar status:", dict(sorted(counts.items())))

bad = [r for r in rows if r.get("status") in {"failed", "error"}]
if bad:
    for r in bad:
        print("FAILED", r.get("job_id"), "exit", r.get("exit_code"), r.get("log_path"))
    sys.exit(3)

incomplete = [r for r in rows if r.get("status") != "ok"]
if incomplete:
    for r in incomplete[:8]:
        print("WAIT", r.get("job_id"), r.get("status"), r.get("claimed_by"))
    if len(incomplete) > 8:
        print(f"... {len(incomplete) - 8} more incomplete jobs")
    sys.exit(2)

zero_rows = []
for r in rows:
    try:
        n = int(r.get("rows") or 0)
    except ValueError:
        n = 0
    if n <= 0:
        zero_rows.append(r)

if zero_rows:
    for r in zero_rows:
        print("ZERO_ROWS", r.get("job_id"), r.get("rows"), r.get("log_path"))
    sys.exit(5)

print(f"all sidecars ok: {len(rows)} jobs")
sys.exit(0)
PY
}

log "auto-launch waiting for sidecars"
log "sidecar_run_root=${SIDECAR_RUN_ROOT}"
log "final_stamp=${FINAL_STAMP}"
log "final_run_root=${FINAL_RUN_ROOT}"
log "gpus=${GPUS_STR}"

while true; do
  set +e
  check_sidecars_once
  rc=$?
  set -e

  case "${rc}" in
    0)
      break
      ;;
    2|4)
      log "sidecars not ready; sleeping ${POLL_SECONDS}s"
      sleep "${POLL_SECONDS}"
      ;;
    *)
      log "sidecar queue failed validation with rc=${rc}; not launching final training"
      exit "${rc}"
      ;;
  esac
done

log "running sidecar inventory"
ACTION=inventory RUN_ROOT="${SIDECAR_RUN_ROOT}" bash "${SIDECAR_SCRIPT}"

log "running final matrix preflight"
ACTION=preflight \
  STAMP="${FINAL_STAMP}" \
  RUN_ROOT="${FINAL_RUN_ROOT}" \
  GPUS_STR="${GPUS_STR}" \
  SIDECAR_RUN_ROOT="${SIDECAR_RUN_ROOT}" \
  bash "${FINAL_SCRIPT}"

log "launching final matrix"
ACTION=launch \
  STAMP="${FINAL_STAMP}" \
  RUN_ROOT="${FINAL_RUN_ROOT}" \
  GPUS_STR="${GPUS_STR}" \
  SIDECAR_RUN_ROOT="${SIDECAR_RUN_ROOT}" \
  bash "${FINAL_SCRIPT}"

log "final matrix launch command completed"
