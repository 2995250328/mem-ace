#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

CONDA_ENV="${CONDA_ENV:-mapanything}"
SCENE="${SCENE:-Cambridge_KingsCollege}"
GPU_ID="${GPU_ID:-0}"
DEVICE="${DEVICE:-cuda:${GPU_ID}}"
EVAL_DEVICE="${EVAL_DEVICE:-${DEVICE}}"
CAMBRIDGE_ROOT="${CAMBRIDGE_ROOT:-/data/xwh/Cambridge}"
DINOV2_PATH="${DINOV2_PATH:-/home/xwh/data/checkpoints/dinov2_vitl14_pretrain.pth}"
MEMORY_PATH="${MEMORY_PATH:-}"
RUN_ROOT="${RUN_ROOT:-/data/xwh/ace_dinov2_lmc/04_evaluation/cambridge_dino_lmc/train}"
OUTPUT_MAP_SUFFIX="${OUTPUT_MAP_SUFFIX:-${SCENE}_dino_lmc.pt}"
TRAIN_PRESET="${TRAIN_PRESET:-memory_compare_ace_g_v2}"
LMC_MODE="${LMC_MODE:-global}"
LMC_ITERATIONS="${LMC_ITERATIONS:-12}"
NUM_LATENT_TOKENS="${NUM_LATENT_TOKENS:-64}"
IMAGE_RESOLUTION="${IMAGE_RESOLUTION:-518}"
TRAINING_BUFFER_SIZE="${TRAINING_BUFFER_SIZE:-2800000}"
BUFFER_SIZE_FINAL="${BUFFER_SIZE_FINAL:-7600000}"
BUFFER_ON_CPU="${BUFFER_ON_CPU:-True}"
BUFFER_ON_CPU_FINAL="${BUFFER_ON_CPU_FINAL:-True}"
SAMPLES_PER_IMAGE="${SAMPLES_PER_IMAGE:-512}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
POST_TRAIN_HYPOTHESES="${POST_TRAIN_HYPOTHESES:-256}"
POST_TRAIN_EVAL_SEEDS="${POST_TRAIN_EVAL_SEEDS:-1305 2026 4242}"
BEST_METRIC="${BEST_METRIC:-pct5}"
DRY_RUN="${DRY_RUN:-false}"

SCENE_ROOT="${CAMBRIDGE_ROOT}/${SCENE}"
TRAIN_ROOT="${SCENE_ROOT}/train"

if [[ -z "${MEMORY_PATH}" ]]; then
  echo "ERROR: MEMORY_PATH is required for Cambridge DINO+LMC." >&2
  exit 2
fi
if [[ ! -s "${MEMORY_PATH}" ]]; then
  echo "ERROR: missing memory: ${MEMORY_PATH}" >&2
  exit 2
fi
if [[ ! -d "${SCENE_ROOT}/train/rgb" || ! -d "${SCENE_ROOT}/test/rgb" ]]; then
  echo "ERROR: incomplete Cambridge ACE scene root: ${SCENE_ROOT}" >&2
  exit 2
fi
if [[ ! -d "${TRAIN_ROOT}/sparse_depth" ]]; then
  echo "ERROR: missing sparse depth for valid-coordinate sampling: ${TRAIN_ROOT}/sparse_depth" >&2
  exit 2
fi

CMD=(
  conda run --no-capture-output -n "${CONDA_ENV}" python ace_dinov2_lmc/train_ace_dinov2_lmc.py
  "${SCENE_ROOT}"
  "${OUTPUT_MAP_SUFFIX}"
  --data_backend ace
  --post_train_eval_scene "${SCENE_ROOT}"
  --use_lmc True
  --memory_path "${MEMORY_PATH}"
  --dinov2_path "${DINOV2_PATH}"
  --train_preset "${TRAIN_PRESET}"
  --model_backend ace_dinov2
  --lmc_flow ace_g
  --lmc_mode "${LMC_MODE}"
  --lmc_iterations "${LMC_ITERATIONS}"
  --num_latent_tokens "${NUM_LATENT_TOKENS}"
  --best_metric "${BEST_METRIC}"
  --device "${DEVICE}"
  --post_train_eval_device "${EVAL_DEVICE}"
  --training_buffer_size "${TRAINING_BUFFER_SIZE}"
  --buffer_size_final "${BUFFER_SIZE_FINAL}"
  --buffer_on_cpu "${BUFFER_ON_CPU}"
  --buffer_on_cpu_final "${BUFFER_ON_CPU_FINAL}"
  --samples_per_image "${SAMPLES_PER_IMAGE}"
  --batch_size "${BATCH_SIZE}"
  --image_resolution "${IMAGE_RESOLUTION}"
  --experiment_root "${RUN_ROOT}"
  --experiment_subdir "dino_lmc/${SCENE}"
  --post_train_hypotheses "${POST_TRAIN_HYPOTHESES}"
  --post_train_eval_seeds ${POST_TRAIN_EVAL_SEEDS}
  --buffer_sample_valid_coords True
  --buffer_valid_coord_sample_ratio 1.0
  --buffer_valid_coord_neighbor_radius 1
  --buffer_valid_coord_neighbor_mode cross
  --c1_aux_depth_root "${TRAIN_ROOT}/sparse_depth"
  --c1_aux_depth_kind sparse_depth
)

printf "CAMBRIDGE_DINO_LMC_CMD: "
printf "%q " "${CMD[@]}"
printf "\n"

if [[ "${DRY_RUN}" == "true" ]]; then
  exit 0
fi

"${CMD[@]}"
