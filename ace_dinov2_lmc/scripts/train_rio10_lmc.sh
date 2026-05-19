#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACE_DEPTH_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"

ACE_RIO10_ROOT="${ACE_RIO10_ROOT:-/data/xwh/RIO10_adapted/ace/rio10}"
SCENE="${SCENE:-scene01_seq01_01}"
MEMORY_PATH="${MEMORY_PATH:?Set MEMORY_PATH to a RIO10 memory_bse.pt file}"
OUTPUT_NAME="${OUTPUT_NAME:-${SCENE}_rio10_lmc.pt}"
DEVICE="${DEVICE:-cuda:0}"
TRAIN_PRESET="${TRAIN_PRESET:-memory_compare_ace_g_v2}"
EXPERIMENT_SUBDIR="${EXPERIMENT_SUBDIR:-rio10_lmc}"

cd "${ACE_DEPTH_ROOT}"
python "${REPO_ROOT}/train_ace_dinov2_lmc.py" \
  "${ACE_RIO10_ROOT}/${SCENE}" \
  "${OUTPUT_NAME}" \
  --train_preset "${TRAIN_PRESET}" \
  --data_backend ace \
  --device "${DEVICE}" \
  --post_train_eval_device "${DEVICE}" \
  --use_lmc True \
  --memory_path "${MEMORY_PATH}" \
  --experiment_subdir "${EXPERIMENT_SUBDIR}"
