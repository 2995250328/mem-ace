#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACE_DEPTH_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"

WAI_ROOT="${WAI_ROOT:-/data/xwh/RIO10_wai/mapanything_wai}"
SCENE_NAME="${SCENE_NAME:-scene01_seq01_01_train}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/memory_extraction/04_evaluation/memory_extract/rio10}"
DEVICE="${DEVICE:-cuda:0}"
N_MEMORY="${N_MEMORY:-40}"
VOXEL_SIZE="${VOXEL_SIZE:-0.05}"
MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-}"
PATCH_DEPTH_SAMPLING="${PATCH_DEPTH_SAMPLING:-nearest_valid}"
DATASET_RESOLUTION="${DATASET_RESOLUTION:-518}"
MAPANYTHING_USE_AMP="${MAPANYTHING_USE_AMP:-true}"
USE_PATCH_BASED="${USE_PATCH_BASED:-true}"
POSTPROCESS_ON_CPU="${POSTPROCESS_ON_CPU:-true}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_CUDA_ALLOC_CONF

OUT_SCENE_NAME="${SCENE_NAME}"
OUT_SCENE_NAME="${OUT_SCENE_NAME}_res${DATASET_RESOLUTION}_n${N_MEMORY}"
case "${USE_PATCH_BASED,,}" in
  true|1|yes|y|on)
    OUT_SCENE_NAME="${OUT_SCENE_NAME}_patch"
    ;;
  *)
    OUT_SCENE_NAME="${OUT_SCENE_NAME}_pixel"
    ;;
esac
case "${POSTPROCESS_ON_CPU,,}" in
  true|1|yes|y|on)
    OUT_SCENE_NAME="${OUT_SCENE_NAME}_cpu"
    ;;
esac
case "${MAPANYTHING_USE_AMP,,}" in
  true|1|yes|y|on)
    OUT_SCENE_NAME="${OUT_SCENE_NAME}_amp"
    ;;
esac
OUT_PATH="${OUT_PATH:-${OUT_ROOT}/${OUT_SCENE_NAME}/$(date +%Y%m%d_%H%M%S)/memory_bse.pt}"

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
  --dataset_resolution "${DATASET_RESOLUTION}"
  --wai_view_mode anchor_support
  --patch_depth_sampling "${PATCH_DEPTH_SAMPLING}"
  --depth_valid_range 0.1 10.0
)

case "${MAPANYTHING_USE_AMP,,}" in
  true|1|yes|y|on)
    args+=(--mapanything_use_amp)
    ;;
esac

case "${USE_PATCH_BASED,,}" in
  true|1|yes|y|on)
    args+=(--use_patch_based)
    ;;
esac

case "${POSTPROCESS_ON_CPU,,}" in
  true|1|yes|y|on)
    args+=(--postprocess_on_cpu)
    ;;
esac

if [[ -n "${MODEL_CHECKPOINT}" ]]; then
  args+=(--model_checkpoint "${MODEL_CHECKPOINT}")
fi

printf '[RIO10 Memory] output=%s\n' "${OUT_PATH}"
printf '[RIO10 Memory] resolution=%s n_memory=%s patch=%s cpu_post=%s amp=%s device=%s\n' \
  "${DATASET_RESOLUTION}" "${N_MEMORY}" "${USE_PATCH_BASED}" "${POSTPROCESS_ON_CPU}" "${MAPANYTHING_USE_AMP}" "${DEVICE}"

cd "${ACE_DEPTH_ROOT}"
python -m ace_dinov2_lmc.memory_extraction.run_memory_extraction "${args[@]}"
