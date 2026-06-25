#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/xwh/project/ace_depth"
PMRF_SESSION="${PMRF_SESSION:-pmrf_v3_sq_bears_20260622_gpu01}"
TARGET_SESSION="${TARGET_SESSION:-single_qknorm_after_pmrf_20260623_gpu01}"
RUN_ROOT="${RUN_ROOT:-/data/xwh/ace_dinov2_lmc/04_evaluation/single_qknorm_stage12_representative_20260623_gpu01}"
LAUNCHER="${ROOT_DIR}/ace_dinov2_lmc/scripts/launch_single_qknorm_stage12_representative_gpu01.sh"
POLL_SECONDS="${POLL_SECONDS:-300}"

mkdir -p "${RUN_ROOT}/logs"
cd "${ROOT_DIR}"

echo "[$(date)] watcher started"
echo "PMRF_SESSION=${PMRF_SESSION}"
echo "TARGET_SESSION=${TARGET_SESSION}"
echo "RUN_ROOT=${RUN_ROOT}"
echo "LAUNCHER=${LAUNCHER}"

while tmux has-session -t "${PMRF_SESSION}" 2>/dev/null; do
  echo "[$(date)] waiting: tmux session ${PMRF_SESSION} still exists"
  sleep "${POLL_SECONDS}"
done

echo "[$(date)] PMRF session ${PMRF_SESSION} finished or absent; checking GPU0/1 compute apps"

mapfile -t GPU_UUIDS < <(nvidia-smi --query-gpu=uuid --format=csv,noheader,nounits -i 0,1)
GPU0_UUID="${GPU_UUIDS[0]:-}"
GPU1_UUID="${GPU_UUIDS[1]:-}"
if [[ -z "${GPU0_UUID}" || -z "${GPU1_UUID}" ]]; then
  echo "[$(date)] failed to resolve GPU0/1 UUIDs" >&2
  exit 2
fi

while true; do
  apps="$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits || true)"
  if ! grep -E "^(${GPU0_UUID}|${GPU1_UUID})," <<<"${apps}" >/dev/null; then
    break
  fi
  echo "[$(date)] waiting: GPU0/1 still has compute apps"
  grep -E "^(${GPU0_UUID}|${GPU1_UUID})," <<<"${apps}" || true
  sleep 60
done

echo "[$(date)] launching single_qknorm_layerscale on GPU0/1"
RUN_ROOT="${RUN_ROOT}" bash "${LAUNCHER}"
