#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

CONDA_ENV="${CONDA_ENV:-mapanything}"
SCENE="${SCENE:-wayspots_bears}"
GPU_ID="${GPU_ID:-0}"
DEVICE="${DEVICE:-cuda:${GPU_ID}}"
EVAL_DEVICE="${EVAL_DEVICE:-${DEVICE}}"
WAYSPOTS_ROOT="${WAYSPOTS_ROOT:-/data/xwh/Wayspots}"
DINOV2_PATH="${DINOV2_PATH:-/home/xwh/data/checkpoints/dinov2_vitl14_pretrain.pth}"
GLACE_ROOT="${GLACE_ROOT:-/home/xwh/project/glace}"
GLACE_ENCODER_PATH="${GLACE_ENCODER_PATH:-/home/xwh/project/glace/ace_encoder_pretrained.pt}"
GLACE_FEAT_NAME="${GLACE_FEAT_NAME:-features.npy}"
GLACE_RESIDUAL_MODE="${GLACE_RESIDUAL_MODE:-local_delta_tanh_scalar}"
GLACE_RESIDUAL_GATE_INIT="${GLACE_RESIDUAL_GATE_INIT:-0.20}"
MEMORY_PATH="${MEMORY_PATH:-}"
OUTPUT_MAP_SUFFIX="${OUTPUT_MAP_SUFFIX:-${SCENE}_glace_lmc.pt}"
TRAIN_PRESET="${TRAIN_PRESET:-memory_compare_ace_g_v2}"
LMC_MODE="${LMC_MODE:-global}"
LMC_AUTO_MODE_BY_VISIBILITY="${LMC_AUTO_MODE_BY_VISIBILITY:-False}"
LMC_FPS_START_POLICY="${LMC_FPS_START_POLICY:-farthest_from_center}"
LMC_KEY_SLICE_IDX="${LMC_KEY_SLICE_IDX:-2}"
LMC_KEY_FEATURE_MODE="${LMC_KEY_FEATURE_MODE:-slice}"
S1_LOSS_STEP_MODE="${S1_LOSS_STEP_MODE:-per_iter}"
LMC_FUSION_REFINEMENT_MODE="${LMC_FUSION_REFINEMENT_MODE:-cascade_internal}"
LMC_FUSION_CASCADE_LAYERS="${LMC_FUSION_CASCADE_LAYERS:-4}"
LMC_FUSION_ASSEMBLY_GAMMA_INIT="${LMC_FUSION_ASSEMBLY_GAMMA_INIT:-0.05}"
ACE_G_FUSION_LR_RATIO="${ACE_G_FUSION_LR_RATIO:-0.03}"
BEST_METRIC="${BEST_METRIC:-pct5}"
IMAGE_RESOLUTION="${IMAGE_RESOLUTION:-518}"
TRAINING_BUFFER_SIZE="${TRAINING_BUFFER_SIZE:-3200000}"
BUFFER_SIZE_FINAL="${BUFFER_SIZE_FINAL:-8000000}"
BUFFER_ON_CPU="${BUFFER_ON_CPU:-True}"
BUFFER_ON_CPU_FINAL="${BUFFER_ON_CPU_FINAL:-True}"
SAMPLES_PER_IMAGE="${SAMPLES_PER_IMAGE:-512}"
BATCH_SIZE="${BATCH_SIZE:-10240}"
POST_TRAIN_HYPOTHESES="${POST_TRAIN_HYPOTHESES:-256}"
POST_TRAIN_EVAL_SEEDS="${POST_TRAIN_EVAL_SEEDS:-1305 2026 4242}"
SAMPLING_DEPTH_ROOT="${SAMPLING_DEPTH_ROOT:-${WAYSPOTS_ROOT}/${SCENE}/train}"
SAMPLING_DEPTH_KIND="${SAMPLING_DEPTH_KIND:-sparse_depth_sampling_sp}"
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/ace_dinov2_lmc/04_evaluation/wayspots_glace_lmc/train}"
DRY_RUN="${DRY_RUN:-false}"

if [[ -z "${MEMORY_PATH}" ]]; then
  echo 'MEMORY_PATH is required. Pass the current pooled/BSE memory_bse.pt path.' >&2
  exit 1
fi

SCENE_ROOT="${WAYSPOTS_ROOT}/${SCENE}"
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
  --model_backend glace_lmc
  --glace_root "${GLACE_ROOT}"
  --glace_encoder_path "${GLACE_ENCODER_PATH}"
  --glace_feat_name "${GLACE_FEAT_NAME}"
  --glace_residual_mode "${GLACE_RESIDUAL_MODE}"
  --glace_residual_gate_init "${GLACE_RESIDUAL_GATE_INIT}"
  --lmc_flow ace_g
  --lmc_mode "${LMC_MODE}"
  --lmc_auto_mode_by_visibility "${LMC_AUTO_MODE_BY_VISIBILITY}"
  --lmc_fps_start_policy "${LMC_FPS_START_POLICY}"
  --lmc_key_slice_idx "${LMC_KEY_SLICE_IDX}"
  --lmc_key_feature_mode "${LMC_KEY_FEATURE_MODE}"
  --s1_loss_step_mode "${S1_LOSS_STEP_MODE}"
  --lmc_fusion_refinement_mode "${LMC_FUSION_REFINEMENT_MODE}"
  --lmc_fusion_cascade_layers "${LMC_FUSION_CASCADE_LAYERS}"
  --lmc_fusion_assembly_gamma_init "${LMC_FUSION_ASSEMBLY_GAMMA_INIT}"
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
  --post_train_hypotheses "${POST_TRAIN_HYPOTHESES}"
  --post_train_eval_seeds ${POST_TRAIN_EVAL_SEEDS}
  --buffer_sample_valid_coords True
  --buffer_valid_coord_sample_ratio 1.0
  --buffer_valid_coord_neighbor_radius 1
  --buffer_valid_coord_neighbor_mode cross
  --c1_aux_depth_root "${SAMPLING_DEPTH_ROOT}"
  --c1_aux_depth_kind "${SAMPLING_DEPTH_KIND}"
)

printf 'GLACE_LMC_TRAIN_CMD: '
printf '%q ' "${CMD[@]}"
printf '
'
if [[ "${DRY_RUN}" == "true" ]]; then
  exit 0
fi
"${CMD[@]}"
