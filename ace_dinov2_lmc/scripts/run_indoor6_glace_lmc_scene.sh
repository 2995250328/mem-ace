#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

CONDA_ENV="${CONDA_ENV:-mapanything}"
SCENE="${SCENE:-scene3}"
GPU_ID="${GPU_ID:-0}"
DEVICE="${DEVICE:-cuda:${GPU_ID}}"
EVAL_DEVICE="${EVAL_DEVICE:-${DEVICE}}"
ACE_ROOT="${ACE_ROOT:-/home/xwh/data/indoor6_ace}"
WAI_ROOT="${WAI_ROOT:-/home/xwh/data/mapanything-dataset/wai_data/indoor6}"
GLACE_ROOT="${GLACE_ROOT:-/home/xwh/project/glace}"
GLACE_ENCODER_PATH="${GLACE_ENCODER_PATH:-/home/xwh/project/glace/ace_encoder_pretrained.pt}"
GLACE_FEAT_NAME="${GLACE_FEAT_NAME:-features.npy}"
DINOV2_PATH="${DINOV2_PATH:-/home/xwh/data/checkpoints/dinov2_vitl14_pretrain.pth}"
MEMORY_PATH="${MEMORY_PATH:-}"
RUN_ROOT="${RUN_ROOT:-/data/xwh/ace_dinov2_lmc/04_evaluation/indoor6_glace_lmc}"
OUTPUT_MAP_SUFFIX="${OUTPUT_MAP_SUFFIX:-${SCENE}_glace_lmc.pt}"
TRAIN_PRESET="${TRAIN_PRESET:-memory_compare_ace_g_v2}"
LMC_MODE="${LMC_MODE:-global}"
LMC_FUSION_TARGET="${LMC_FUSION_TARGET:-decoder}"
LMC_ITERATIONS="${LMC_ITERATIONS:-28}"
NUM_LATENT_TOKENS="${NUM_LATENT_TOKENS:-64}"
IMAGE_RESOLUTION="${IMAGE_RESOLUTION:-518}"
TRAINING_BUFFER_SIZE="${TRAINING_BUFFER_SIZE:-2600000}"
BUFFER_SIZE_FINAL="${BUFFER_SIZE_FINAL:-7700000}"
BUFFER_ON_CPU="${BUFFER_ON_CPU:-True}"
BUFFER_ON_CPU_FINAL="${BUFFER_ON_CPU_FINAL:-True}"
SAMPLES_PER_IMAGE="${SAMPLES_PER_IMAGE:-384}"
BATCH_SIZE="${BATCH_SIZE:-5120}"
POST_TRAIN_HYPOTHESES="${POST_TRAIN_HYPOTHESES:-256}"
POST_TRAIN_EVAL_SEEDS="${POST_TRAIN_EVAL_SEEDS:-1305 2026 4242}"
BEST_METRIC="${BEST_METRIC:-pct5}"
ACE_G_FUSION_LR_RATIO="${ACE_G_FUSION_LR_RATIO:-0.03}"
GLACE_RESIDUAL_MODE="${GLACE_RESIDUAL_MODE:-local_delta_tanh_scalar}"
GLACE_RESIDUAL_GATE_INIT="${GLACE_RESIDUAL_GATE_INIT:-0.20}"
AUX_DEPTH_KIND="${AUX_DEPTH_KIND:-colmap_depth}"
DRY_RUN="${DRY_RUN:-false}"

SCENE_ROOT="${ACE_ROOT}/${SCENE}"
TRAIN_ROOT="${SCENE_ROOT}/train"
TEST_ROOT="${SCENE_ROOT}/test"
AUX_DEPTH_ROOT="${WAI_ROOT}/${SCENE}_train"

if [[ -z "${MEMORY_PATH}" ]]; then
  echo "ERROR: MEMORY_PATH is required for Indoor6 GLACE+LMC." >&2
  exit 2
fi
if [[ ! -s "${MEMORY_PATH}" ]]; then
  echo "ERROR: missing memory: ${MEMORY_PATH}" >&2
  exit 2
fi
for split_root in "${TRAIN_ROOT}" "${TEST_ROOT}"; do
  if [[ ! -d "${split_root}/rgb" || ! -d "${split_root}/poses" || ! -d "${split_root}/calibration" ]]; then
    echo "ERROR: ACE split root is incomplete: ${split_root}" >&2
    exit 2
  fi
  if [[ ! -s "${split_root}/${GLACE_FEAT_NAME}" ]]; then
    echo "ERROR: missing GLACE global features: ${split_root}/${GLACE_FEAT_NAME}" >&2
    echo "Run /home/xwh/project/glace/datasets/extract_features.py for train/test before GLACE+LMC." >&2
    exit 2
  fi
done
if [[ ! -d "${AUX_DEPTH_ROOT}/${AUX_DEPTH_KIND}" ]]; then
  echo "ERROR: missing Indoor6 WAI depth root: ${AUX_DEPTH_ROOT}/${AUX_DEPTH_KIND}" >&2
  echo "       Indoor6 depth here is COLMAP-derived pseudo depth; set AUX_DEPTH_KIND=colmap_depth unless intentionally using another alias." >&2
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
  --use_scale_token False
  --dinov2_path "${DINOV2_PATH}"
  --train_preset "${TRAIN_PRESET}"
  --model_backend glace_lmc
  --glace_root "${GLACE_ROOT}"
  --glace_encoder_path "${GLACE_ENCODER_PATH}"
  --glace_feat_name "${GLACE_FEAT_NAME}"
  --glace_residual_mode "${GLACE_RESIDUAL_MODE}"
  --glace_residual_gate_init "${GLACE_RESIDUAL_GATE_INIT}"
  --lmc_flow ace_g
  --lmc_mode "${LMC_MODE}"
  --lmc_fusion_target "${LMC_FUSION_TARGET}"
  --lmc_iterations "${LMC_ITERATIONS}"
  --num_latent_tokens "${NUM_LATENT_TOKENS}"
  --ace_g_fusion_lr_ratio "${ACE_G_FUSION_LR_RATIO}"
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
  --experiment_subdir "glace_lmc/${SCENE}"
  --post_train_hypotheses "${POST_TRAIN_HYPOTHESES}"
  --post_train_eval_seeds ${POST_TRAIN_EVAL_SEEDS}
  --buffer_sample_valid_coords True
  --buffer_valid_coord_sample_ratio 1.0
  --buffer_valid_coord_neighbor_radius 1
  --buffer_valid_coord_neighbor_mode cross
  --c1_aux_depth_root "${AUX_DEPTH_ROOT}"
  --c1_aux_depth_kind "${AUX_DEPTH_KIND}"
)

if [[ -n "${EXTRA_TRAIN_ARGS:-}" ]]; then
  read -r -a EXTRA_TRAIN_ARGS_ARRAY <<< "${EXTRA_TRAIN_ARGS}"
  CMD+=("${EXTRA_TRAIN_ARGS_ARRAY[@]}")
fi

printf "INDOOR6_GLACE_LMC_CMD: "
printf "%q " "${CMD[@]}"
printf "
"
if [[ "${DRY_RUN}" == "true" ]]; then
  exit 0
fi
"${CMD[@]}"
