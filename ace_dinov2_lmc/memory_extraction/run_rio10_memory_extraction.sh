#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACE_DEPTH_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"

WAI_ROOT="${WAI_ROOT:-/data/xwh/RIO10_adapted/mapanything_wai}"
SCENE_NAME="${SCENE_NAME:-scene01_seq01_01_train}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/memory_extraction/04_evaluation/memory_extract/rio10}"
OUT_PATH="${OUT_PATH:-${OUT_ROOT}/${SCENE_NAME}/memory_bse.pt}"
DEVICE="${DEVICE:-cuda:0}"
N_MEMORY="${N_MEMORY:-40}"
VOXEL_SIZE="${VOXEL_SIZE:-0.05}"
MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-}"

mkdir -p "$(dirname "${OUT_PATH}")"

args=(
  "${WAI_ROOT}"
  "${OUT_PATH}"
  --dataset_loader wai
  --dataset_type rio10
  --scene_name "${SCENE_NAME}"
  --n_memory "${N_MEMORY}"
  --device "${DEVICE}"
  --voxel_size "${VOXEL_SIZE}"
  --dataset_transform imgnorm
  --dataset_data_norm_type dinov2
  --dataset_aug_crop 0
  --wai_view_mode anchor_support
  --patch_depth_sampling nearest
  --depth_valid_range 0.1 10.0
)

if [[ -n "${MODEL_CHECKPOINT}" ]]; then
  args+=(--model_checkpoint "${MODEL_CHECKPOINT}")
fi

cd "${ACE_DEPTH_ROOT}"
python -m ace_dinov2_lmc.memory_extraction.run_memory_extraction "${args[@]}"
