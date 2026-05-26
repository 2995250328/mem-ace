#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

CONDA_ENV="${CONDA_ENV:-mapanything}"
CONDA_RUN=(conda run --no-capture-output -n "${CONDA_ENV}")

SCENE="${SCENE:-wayspots_bears}"
GPU_ID="${GPU_ID:-1}"
DEVICE="${DEVICE:-cuda:${GPU_ID}}"
EVAL_DEVICE="${EVAL_DEVICE:-${DEVICE}}"
DINOV2_PATH="${DINOV2_PATH:-/home/xwh/data/checkpoints/dinov2_vitl14_pretrain.pth}"
WAI_REPO_ROOT="${WAI_REPO_ROOT:-/home/xwh/project/map-anything}"
WAYSPOTS_ACE_ROOT="${WAYSPOTS_ACE_ROOT:-/data/xwh/Wayspots}"
WAYSPOTS_WAI_ROOT="${WAYSPOTS_WAI_ROOT:-/data/xwh/Wayspots_wai/custom}"
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/ace_dinov2_lmc/04_evaluation/wayspots_lmc/$(date +%Y%m%d_%H%M%S)}"
DRY_RUN="${DRY_RUN:-false}"

# memory extraction recipe: current best custom/WAI setup
N_VIEWS="${N_VIEWS:-40}"
VOXEL_SIZE="${VOXEL_SIZE:-0.05}"
UNIMODAL_THRESHOLD="${UNIMODAL_THRESHOLD:-0.02}"
POOL_MODE="${POOL_MODE:-bse}"
WAI_VIEW_MODE="${WAI_VIEW_MODE:-anchor_support}"
ASB_ADAPTIVE="${ASB_ADAPTIVE:-true}"
ASB_POST_REPAIR="${ASB_POST_REPAIR:-true}"
ASB_POSE_PRUNE="${ASB_POSE_PRUNE:-true}"
ASB_CANDIDATE_POOL_RATIO="${ASB_CANDIDATE_POOL_RATIO:-1.0}"
WAI_RESOLUTION="${WAI_RESOLUTION:-518}"

# LMC training defaults
TRAIN_PRESET="${TRAIN_PRESET:-memory_compare_ace_g_v2}"
LMC_MODE="${LMC_MODE:-global}"
IMAGE_RESOLUTION="${IMAGE_RESOLUTION:-518}"
TRAINING_BUFFER_SIZE="${TRAINING_BUFFER_SIZE:-4000000}"
BUFFER_SIZE_FINAL="${BUFFER_SIZE_FINAL:-8000000}"
SAMPLES_PER_IMAGE="${SAMPLES_PER_IMAGE:-512}"
BATCH_SIZE="${BATCH_SIZE:-10240}"
POST_TRAIN_HYPOTHESES="${POST_TRAIN_HYPOTHESES:-256}"
POST_TRAIN_EVAL_SEEDS="${POST_TRAIN_EVAL_SEEDS:-1305 2026 4242}"
OUTPUT_MAP_SUFFIX="${OUTPUT_MAP_SUFFIX:-wayspots_bears_lmc.pt}"

# Sparse-depth-guided sampling defaults
SAMPLING_DEPTH_KIND="${SAMPLING_DEPTH_KIND:-sparse_depth_sampling_sp}"
BUFFER_SAMPLE_VALID_COORDS="${BUFFER_SAMPLE_VALID_COORDS:-True}"
BUFFER_VALID_COORD_SAMPLE_RATIO="${BUFFER_VALID_COORD_SAMPLE_RATIO:-1.0}"
BUFFER_VALID_COORD_NEIGHBOR_RADIUS="${BUFFER_VALID_COORD_NEIGHBOR_RADIUS:-1}"
BUFFER_VALID_COORD_NEIGHBOR_MODE="${BUFFER_VALID_COORD_NEIGHBOR_MODE:-cross}"

WAI_TRAIN_SCENE="${WAYSPOTS_WAI_ROOT}/${SCENE}_train"
WAI_TEST_SCENE="${WAYSPOTS_WAI_ROOT}/${SCENE}_test"
MEMORY_ROOT="${RUN_ROOT}/memory"
TRAIN_ROOT="${RUN_ROOT}/train"
mkdir -p "${MEMORY_ROOT}" "${TRAIN_ROOT}"

PREP_CMD=(bash "${ROOT_DIR}/ace_dinov2_lmc/scripts/prepare_wayspots_lmc_wai.sh")
MEMORY_CMD=(
  env
  "CUDA_VISIBLE_DEVICES=${GPU_ID}"
  "ACE_DATA_ROOT=/home/xwh/data"
  "DATASET_TYPE=custom"
  "DATASET_LOADER=wai"
  "DATASET_ROOT=${WAYSPOTS_WAI_ROOT}"
  "SCENE_TRAIN=${SCENE}_train"
  "SCENE_TEST=${SCENE}_test"
  "OUTPUT_SCENE_NAME=${SCENE}"
  "OUTPUT_ROOT=${MEMORY_ROOT}"
  "N_VIEWS=${N_VIEWS}"
  "VOXEL_SIZE=${VOXEL_SIZE}"
  "UNIMODAL_THRESHOLD=${UNIMODAL_THRESHOLD}"
  "POOL_MODE=${POOL_MODE}"
  "WAI_VIEW_MODE=${WAI_VIEW_MODE}"
  "ASB_ADAPTIVE=${ASB_ADAPTIVE}"
  "ASB_POST_REPAIR=${ASB_POST_REPAIR}"
  "ASB_POSE_PRUNE=${ASB_POSE_PRUNE}"
  "ASB_CANDIDATE_POOL_RATIO=${ASB_CANDIDATE_POOL_RATIO}"
  "WAI_RESOLUTION=${WAI_RESOLUTION}"
  "${CONDA_RUN[@]}" bash "${ROOT_DIR}/ace_dinov2_lmc/memory_extraction/extract_memory.sh"
)

find_memory() {
  find "${MEMORY_ROOT}/${SCENE}" -type f -name memory_bse.pt -printf '%T@ %p
' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-
}

TRAIN_CMD=(
  env
  "CUDA_VISIBLE_DEVICES=${GPU_ID}"
  "${CONDA_RUN[@]}" python "${ROOT_DIR}/ace_dinov2_lmc/train_ace_dinov2_lmc.py"
  "${WAI_TRAIN_SCENE}"
  "${OUTPUT_MAP_SUFFIX}"
  --data_backend wai
  --post_train_eval_scene "${WAI_TEST_SCENE}"
  --wai_repo_root "${WAI_REPO_ROOT}"
  --wai_image_modality image
  --use_lmc True
  --memory_path PLACEHOLDER_MEMORY
  --dinov2_path "${DINOV2_PATH}"
  --train_preset "${TRAIN_PRESET}"
  --lmc_flow ace_g
  --lmc_mode "${LMC_MODE}"
  --device "${DEVICE}"
  --post_train_eval_device "${EVAL_DEVICE}"
  --training_buffer_size "${TRAINING_BUFFER_SIZE}"
  --buffer_size_final "${BUFFER_SIZE_FINAL}"
  --samples_per_image "${SAMPLES_PER_IMAGE}"
  --batch_size "${BATCH_SIZE}"
  --image_resolution "${IMAGE_RESOLUTION}"
  --experiment_root "${TRAIN_ROOT}"
  --post_train_hypotheses "${POST_TRAIN_HYPOTHESES}"
  --post_train_eval_seeds ${POST_TRAIN_EVAL_SEEDS}
  --buffer_sample_valid_coords "${BUFFER_SAMPLE_VALID_COORDS}"
  --buffer_valid_coord_sample_ratio "${BUFFER_VALID_COORD_SAMPLE_RATIO}"
  --buffer_valid_coord_neighbor_radius "${BUFFER_VALID_COORD_NEIGHBOR_RADIUS}"
  --buffer_valid_coord_neighbor_mode "${BUFFER_VALID_COORD_NEIGHBOR_MODE}"
  --c1_aux_depth_root "${WAI_TRAIN_SCENE}"
  --c1_aux_depth_kind "${SAMPLING_DEPTH_KIND}"
)

for cmd_name in PREP_CMD MEMORY_CMD; do
  printf '%s: ' "${cmd_name}"
  eval 'printf "%q " "${'"${cmd_name}"'[@]}"'
  printf '
'
done
if [[ "${DRY_RUN}" == "true" ]]; then
  eval 'printf "TRAIN_CMD: "; printf "%q " "${TRAIN_CMD[@]}"; printf "\n"'
  exit 0
fi

"${PREP_CMD[@]}"
"${MEMORY_CMD[@]}"
MEMORY_PATH="$(find_memory)"
if [[ -z "${MEMORY_PATH}" || ! -f "${MEMORY_PATH}" ]]; then
  echo "Failed to locate memory_bse.pt under ${MEMORY_ROOT}/${SCENE}" >&2
  exit 1
fi
for i in "${!TRAIN_CMD[@]}"; do
  if [[ "${TRAIN_CMD[$i]}" == "PLACEHOLDER_MEMORY" ]]; then
    TRAIN_CMD[$i]="${MEMORY_PATH}"
  fi
done
printf 'TRAIN_CMD: '
printf '%q ' "${TRAIN_CMD[@]}"
printf '
'
"${TRAIN_CMD[@]}"
