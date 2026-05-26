#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

CONDA_ENV="${CONDA_ENV:-mapanything}"
SCENE="${SCENE:-wayspots_bears}"
GPU_ID="${GPU_ID:-0}"
DEVICE="${DEVICE:-cuda:${GPU_ID}}"
WAYSPOTS_ROOT="${WAYSPOTS_ROOT:-/data/xwh/Wayspots}"
GLACE_ROOT="${GLACE_ROOT:-/home/xwh/project/glace}"
GLACE_ENCODER_PATH="${GLACE_ENCODER_PATH:-/home/xwh/project/glace/ace_encoder_pretrained.pt}"
GLACE_FEAT_NAME="${GLACE_FEAT_NAME:-features.npy}"
N_MEMORY="${N_MEMORY:-32}"
VOXEL_SIZE="${VOXEL_SIZE:-0.08}"
UNIMODAL_THRESHOLD="${UNIMODAL_THRESHOLD:-0.02}"
IMAGE_RESOLUTION="${IMAGE_RESOLUTION:-518}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/ace_dinov2_lmc/04_evaluation/wayspots_glace_lmc/memory}"
DRY_RUN="${DRY_RUN:-false}"

SCENE_TRAIN_ROOT="${WAYSPOTS_ROOT}/${SCENE}/train"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_ROOT}/${SCENE}/${STAMP}"
mkdir -p "${OUTPUT_DIR}"
OUTPUT_PATH="${OUTPUT_DIR}/memory_bse.pt"

CMD=(
  conda run --no-capture-output -n "${CONDA_ENV}" python -m ace_dinov2_lmc.memory_extraction.run_memory_extraction
  "${SCENE_TRAIN_ROOT}"
  "${OUTPUT_PATH}"
  --dataset_loader ace
  --dataset_type custom
  --device "${DEVICE}"
  --n_memory "${N_MEMORY}"
  --use_model glace_encoder
  --glace_root "${GLACE_ROOT}"
  --glace_encoder_path "${GLACE_ENCODER_PATH}"
  --glace_feat_name "${GLACE_FEAT_NAME}"
  --pool_mode bse
  --voxel_size "${VOXEL_SIZE}"
  --unimodal_threshold "${UNIMODAL_THRESHOLD}"
  --dataset_resolution "${IMAGE_RESOLUTION}"
  --depth_valid_range 0.1 18.0
  --patch_depth_sampling nearest_valid
  --global_merge true
  --enable_sor
  --postprocess_on_cpu
)

printf 'GLACE_MEMORY_CMD: '
printf '%q ' "${CMD[@]}"
printf '
'
if [[ "${DRY_RUN}" == "true" ]]; then
  exit 0
fi
"${CMD[@]}"
