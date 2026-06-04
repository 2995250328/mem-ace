#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

CONDA_ENV="${CONDA_ENV:-mapanything}"
SCENE="${SCENE:-Cambridge_KingsCollege}"
GPU_ID="${GPU_ID:-0}"
DEVICE="${DEVICE:-cuda:${GPU_ID}}"
CAMBRIDGE_ROOT="${CAMBRIDGE_ROOT:-/data/xwh/Cambridge}"
DINOV2_PATH="${DINOV2_PATH:-/home/xwh/data/checkpoints/dinov2_vitl14_pretrain.pth}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/xwh/ace_dinov2_lmc/04_evaluation/cambridge_dino_lmc/memory}"
N_MEMORY="${N_MEMORY:-64}"
VOXEL_SIZE="${VOXEL_SIZE:-0.08}"
UNIMODAL_THRESHOLD="${UNIMODAL_THRESHOLD:-0.02}"
IMAGE_RESOLUTION="${IMAGE_RESOLUTION:-518}"
DEPTH_MIN="${DEPTH_MIN:-0.1}"
DEPTH_MAX="${DEPTH_MAX:-1000.0}"
DRY_RUN="${DRY_RUN:-false}"

SCENE_ROOT="${CAMBRIDGE_ROOT}/${SCENE}"
TRAIN_ROOT="${SCENE_ROOT}/train"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_ROOT}/${SCENE}/${STAMP}"
OUTPUT_PATH="${OUTPUT_DIR}/memory_bse.pt"

if [[ ! -d "${TRAIN_ROOT}/rgb" || ! -d "${TRAIN_ROOT}/poses" || ! -d "${TRAIN_ROOT}/calibration" ]]; then
  echo "ERROR: incomplete Cambridge train root: ${TRAIN_ROOT}" >&2
  exit 2
fi
if [[ ! -d "${TRAIN_ROOT}/sparse_depth" ]]; then
  echo "ERROR: missing sparse depth: ${TRAIN_ROOT}/sparse_depth" >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}"

CMD=(
  conda run --no-capture-output -n "${CONDA_ENV}" python -m ace_dinov2_lmc.memory_extraction.run_memory_extraction
  "${TRAIN_ROOT}"
  "${OUTPUT_PATH}"
  --dataset_loader ace
  --dataset_type custom
  --scene_name "${SCENE}"
  --device "${DEVICE}"
  --n_memory "${N_MEMORY}"
  --use_model dinov2
  --dinov2_checkpoint "${DINOV2_PATH}"
  --pool_mode bse
  --voxel_size "${VOXEL_SIZE}"
  --unimodal_threshold "${UNIMODAL_THRESHOLD}"
  --dataset_resolution "${IMAGE_RESOLUTION}"
  --depth_valid_range "${DEPTH_MIN}" "${DEPTH_MAX}"
  --patch_depth_sampling nearest_valid
  --global_merge true
  --enable_sor
  --postprocess_on_cpu
)

printf "CAMBRIDGE_DINO_MEMORY_CMD: "
printf "%q " "${CMD[@]}"
printf "\n"

if [[ "${DRY_RUN}" == "true" ]]; then
  exit 0
fi

"${CMD[@]}"
echo "memory_path=${OUTPUT_PATH}"
