#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

usage() {
  cat <<EOF
Usage:
  MODEL_DIR=/path/to/superpoint_workspace/triangulated_model \
    ace_dinov2_lmc/scripts/build_dino_stgs_keyframe_channel.sh /path/to/scene_root

  ace_dinov2_lmc/scripts/build_dino_stgs_keyframe_channel.sh /path/to/scene_root /path/to/model_dir [/path/to/output_dir]

Builds an STGS keyframe_channel.npz aligned to DINOv2 ACE training:
  output_subsample=14, round_image_multiple=14, image_resolution=518, image_width=700 by default.
EOF
}

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 2
fi

SCENE_ROOT="$1"
MODEL_DIR="${2:-${MODEL_DIR:-}}"
OUTPUT_DIR="${3:-${OUTPUT_DIR:-}}"
SPLIT="${SPLIT:-train}"
CONDA_ENV="${CONDA_ENV:-mapanything}"
IMAGE_RESOLUTION="${IMAGE_RESOLUTION:-518}"
IMAGE_WIDTH="${IMAGE_WIDTH:-700}"
MIN_TRACK_LENGTH="${MIN_TRACK_LENGTH:-4}"
MAX_REPROJ_ERROR="${MAX_REPROJ_ERROR:-2.0}"
MIN_PARALLAX_DEG="${MIN_PARALLAX_DEG:-2.0}"
MAX_PARALLAX_DEG="${MAX_PARALLAX_DEG:-60.0}"
MAX_ANCHOR_ALIGNMENT_PX="${MAX_ANCHOR_ALIGNMENT_PX:-3.5}"
MAX_TARGET_ALIGNMENT_PX="${MAX_TARGET_ALIGNMENT_PX:-3.5}"
ALIGNMENT_SIGMA_PX="${ALIGNMENT_SIGMA_PX:-3.5}"
IMAGE_MATCH_MODE="${IMAGE_MATCH_MODE:-auto}"
MAX_POSE_CENTER_DIST_M="${MAX_POSE_CENTER_DIST_M:-0.10}"
SPARSE_DEPTH_DIR="${SPARSE_DEPTH_DIR:-${SCENE_ROOT}/${SPLIT}/sparse_depth}"
if [[ "${IMAGE_WIDTH,,}" == "none" || "${IMAGE_WIDTH,,}" == "null" || -z "${IMAGE_WIDTH}" ]]; then
  IMAGE_WIDTH_ARG=()
  IMAGE_WIDTH_LABEL="var"
else
  IMAGE_WIDTH_ARG=(--image-width "${IMAGE_WIDTH}")
  IMAGE_WIDTH_LABEL="${IMAGE_WIDTH}"
fi
SIDECAR_NAME="${SIDECAR_NAME:-colmap_keyframe_channel_v1_sp_strict_dino_r${IMAGE_RESOLUTION}_w${IMAGE_WIDTH_LABEL}}"

if [[ -z "${MODEL_DIR}" ]]; then
  echo "MODEL_DIR is required. Pass it as arg2 or export MODEL_DIR." >&2
  exit 2
fi
if [[ ! -d "${SCENE_ROOT}/${SPLIT}/rgb" ]]; then
  echo "Missing ${SPLIT}/rgb under scene root: ${SCENE_ROOT}" >&2
  exit 2
fi
if [[ ! -f "${MODEL_DIR}/points3D.bin" ]]; then
  echo "Missing COLMAP points3D.bin: ${MODEL_DIR}/points3D.bin" >&2
  exit 2
fi

if [[ -z "${OUTPUT_DIR}" ]]; then
  if [[ -n "${SIDECAR_ROOT:-}" ]]; then
    OUTPUT_DIR="${SIDECAR_ROOT}/$(basename "${SCENE_ROOT}")/${SPLIT}/${SIDECAR_NAME}"
  else
    OUTPUT_DIR="${SCENE_ROOT}/${SPLIT}/${SIDECAR_NAME}"
  fi
fi
mkdir -p "${OUTPUT_DIR}"

cmd=(
  conda run --no-capture-output -n "${CONDA_ENV}" python ace_dinov2_lmc/tools/build_colmap_keyframe_channel.py
  "${SCENE_ROOT}"
  --split "${SPLIT}"
  --model-dir "${MODEL_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --target-backbone dinov2
  --image-resolution "${IMAGE_RESOLUTION}"
  "${IMAGE_WIDTH_ARG[@]}"
  --min-track-length "${MIN_TRACK_LENGTH}"
  --max-reproj-error "${MAX_REPROJ_ERROR}"
  --min-parallax-deg "${MIN_PARALLAX_DEG}"
  --max-parallax-deg "${MAX_PARALLAX_DEG}"
  --max-anchor-alignment-px "${MAX_ANCHOR_ALIGNMENT_PX}"
  --max-target-alignment-px "${MAX_TARGET_ALIGNMENT_PX}"
  --alignment-sigma-px "${ALIGNMENT_SIGMA_PX}"
  --image-match-mode "${IMAGE_MATCH_MODE}"
  --max-pose-center-dist-m "${MAX_POSE_CENTER_DIST_M}"
)

if [[ -d "${SPARSE_DEPTH_DIR}" ]]; then
  cmd+=(--sparse-depth-dir "${SPARSE_DEPTH_DIR}")
else
  echo "Sparse depth dir not found, building sidecar without sparse-depth anchor filter: ${SPARSE_DEPTH_DIR}" >&2
fi

echo "[DINO-STGS] scene=${SCENE_ROOT}"
echo "[DINO-STGS] model=${MODEL_DIR}"
echo "[DINO-STGS] output=${OUTPUT_DIR}"
echo "[DINO-STGS] image_resolution=${IMAGE_RESOLUTION} image_width=${IMAGE_WIDTH_LABEL} stride=14 round=14 max_anchor_align=${MAX_ANCHOR_ALIGNMENT_PX} max_target_align=${MAX_TARGET_ALIGNMENT_PX}"
echo "[DINO-STGS] image_match_mode=${IMAGE_MATCH_MODE} max_pose_center_dist_m=${MAX_POSE_CENTER_DIST_M}"
"${cmd[@]}"
