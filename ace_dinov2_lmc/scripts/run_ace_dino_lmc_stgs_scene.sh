#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

usage() {
  cat <<EOF
Usage:
  MEMORY_PATH=/path/to/memory.pt \
    ace_dinov2_lmc/scripts/run_ace_dino_lmc_stgs_scene.sh /path/to/ACE_scene_root

Optional arg2 overrides KEYFRAME_CHANNEL. The scene root must be ACE-format with train/rgb and usually test/rgb.
This runner keeps data_backend=ace so STGS sidecar image indices match the training dataset order.
EOF
}

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 2
fi

SCENE_ROOT="$1"
SCENE_NAME="$(basename "${SCENE_ROOT}")"
KEYFRAME_CHANNEL="${2:-${KEYFRAME_CHANNEL:-}}"
CONDA_ENV="${CONDA_ENV:-mapanything}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-0}"
DEVICE="${DEVICE:-cuda:${GPU_ID}}"
EVAL_DEVICE="${EVAL_DEVICE:-${DEVICE}}"
DINOV2_PATH="${DINOV2_PATH:-/home/xwh/data/checkpoints/dinov2_vitl14_pretrain.pth}"
MEMORY_PATH="${MEMORY_PATH:-}"
RUN_ROOT="${RUN_ROOT:-/data/xwh/ace_dinov2_lmc/04_evaluation/dino_lmc_stgs/train}"
OUTPUT_MAP_SUFFIX="${OUTPUT_MAP_SUFFIX:-${SCENE_NAME}_dino_lmc_stgs.pt}"
POST_TRAIN_EVAL_SCENE="${POST_TRAIN_EVAL_SCENE:-${SCENE_ROOT}}"
EXPERIMENT_SUBDIR="${EXPERIMENT_SUBDIR:-dino_lmc_stgs/${SCENE_NAME}}"
DRY_RUN="${DRY_RUN:-false}"

TRAIN_PRESET="${TRAIN_PRESET:-memory_compare_ace_g_v2}"
LMC_MODE="${LMC_MODE:-global}"
LMC_FLOW="${LMC_FLOW:-ace_g}"
LMC_ITERATIONS="${LMC_ITERATIONS:-12}"
NUM_LATENT_TOKENS="${NUM_LATENT_TOKENS:-64}"
IMAGE_RESOLUTION="${IMAGE_RESOLUTION:-518}"
BUFFER_BATCH_SIZE="${BUFFER_BATCH_SIZE:-10}"
BUFFER_IMAGE_WIDTH="${BUFFER_IMAGE_WIDTH:-$(( (IMAGE_RESOLUTION * 4 / 3 + 13) / 14 * 14 ))}"
TRAINING_BUFFER_SIZE="${TRAINING_BUFFER_SIZE:-4000000}"
BUFFER_SIZE_FINAL="${BUFFER_SIZE_FINAL:-8000000}"
BUFFER_ON_CPU="${BUFFER_ON_CPU:-True}"
BUFFER_ON_CPU_FINAL="${BUFFER_ON_CPU_FINAL:-True}"
SAMPLES_PER_IMAGE="${SAMPLES_PER_IMAGE:-512}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
NUM_DATA_LOADER_WORKERS="${NUM_DATA_LOADER_WORKERS:-8}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-4}"
BEST_METRIC="${BEST_METRIC:-median_error}"
POST_TRAIN_HYPOTHESES="${POST_TRAIN_HYPOTHESES:-256}"
POST_TRAIN_EVAL_SEEDS="${POST_TRAIN_EVAL_SEEDS:-1305 2026 4242}"
EVAL_DETERMINISTIC="${EVAL_DETERMINISTIC:-True}"
EVAL_DSACSTAR_SEED="${EVAL_DSACSTAR_SEED:-1305}"
EVAL_DSACSTAR_SEED_PER_FRAME="${EVAL_DSACSTAR_SEED_PER_FRAME:-True}"
ITERATION_EVAL_SEED="${ITERATION_EVAL_SEED:-1305}"
ITERATION_EVAL_HYPOTHESES="${ITERATION_EVAL_HYPOTHESES:-256}"

USE_STGS_GUIDED_SAMPLING="${USE_STGS_GUIDED_SAMPLING:-True}"
USE_STGS_INTER_FRAME_LOSS="${USE_STGS_INTER_FRAME_LOSS:-False}"
SFM_TRACK_SIDECAR_MISMATCH_POLICY="${SFM_TRACK_SIDECAR_MISMATCH_POLICY:-strict}"
SFM_TRACK_GUIDED_BATCH_SIZE="${SFM_TRACK_GUIDED_BATCH_SIZE:-512}"
SFM_TRACK_GUIDED_SAMPLING_STRATEGY="${SFM_TRACK_GUIDED_SAMPLING_STRATEGY:-balanced_replace}"
SFM_TRACK_GUIDED_FRACTION="${SFM_TRACK_GUIDED_FRACTION:-0.10}"
SFM_TRACK_GUIDED_MODE="${SFM_TRACK_GUIDED_MODE:-anchor_only}"
SFM_TRACK_GUIDED_MAIN_LOSS_MODE="${SFM_TRACK_GUIDED_MAIN_LOSS_MODE:-include}"
SFM_TRACK_GUIDED_SOURCE_TARGET_MODE="${SFM_TRACK_GUIDED_SOURCE_TARGET_MODE:-patch_center}"
SFM_TRACK_INTER_FRAME_TARGET_MODE="${SFM_TRACK_INTER_FRAME_TARGET_MODE:-patch_center}"
SFM_TRACK_GUIDED_AUX_NORMALIZER="${SFM_TRACK_GUIDED_AUX_NORMALIZER:-full_batch}"
SFM_TRACK_ANCHOR_SELF_WEIGHT="${SFM_TRACK_ANCHOR_SELF_WEIGHT:-0.5}"
SFM_TRACK_ANCHOR_USE_ALIGNMENT_WEIGHT="${SFM_TRACK_ANCHOR_USE_ALIGNMENT_WEIGHT:-True}"
SFM_TRACK_INTER_FRAME_WEIGHT="${SFM_TRACK_INTER_FRAME_WEIGHT:-0.0}"
SFM_TRACK_INTER_FRAME_START_RATIO="${SFM_TRACK_INTER_FRAME_START_RATIO:-0.2}"
SFM_TRACK_INTER_FRAME_DECAY_LAST_RATIO="${SFM_TRACK_INTER_FRAME_DECAY_LAST_RATIO:-0.3}"
SFM_TRACK_INTER_FRAME_MAX_PX="${SFM_TRACK_INTER_FRAME_MAX_PX:-100.0}"
SFM_TRACK_INTER_FRAME_DROPOUT="${SFM_TRACK_INTER_FRAME_DROPOUT:-0.5}"
SFM_TRACK_MIN_ALIGNMENT_WEIGHT="${SFM_TRACK_MIN_ALIGNMENT_WEIGHT:-0.0}"
SFM_TRACK_MAX_COLMAP_REPROJ_ERROR_PX="${SFM_TRACK_MAX_COLMAP_REPROJ_ERROR_PX:-0.0}"
SFM_TRACK_MIN_TRACK_LENGTH="${SFM_TRACK_MIN_TRACK_LENGTH:-0}"
SFM_TRACK_MAX_ANCHOR_PATCH_OFFSET_PX="${SFM_TRACK_MAX_ANCHOR_PATCH_OFFSET_PX:-3.5}"
SFM_TRACK_MAX_TARGET_PATCH_OFFSET_PX="${SFM_TRACK_MAX_TARGET_PATCH_OFFSET_PX:-3.5}"

BUFFER_SAMPLE_VALID_COORDS="${BUFFER_SAMPLE_VALID_COORDS:-True}"
BUFFER_VALID_COORD_SAMPLE_RATIO="${BUFFER_VALID_COORD_SAMPLE_RATIO:-1.0}"
BUFFER_VALID_COORD_NEIGHBOR_RADIUS="${BUFFER_VALID_COORD_NEIGHBOR_RADIUS:-1}"
BUFFER_VALID_COORD_NEIGHBOR_MODE="${BUFFER_VALID_COORD_NEIGHBOR_MODE:-cross}"
C1_AUX_DEPTH_KIND="${C1_AUX_DEPTH_KIND:-sparse_depth}"
C1_AUX_DEPTH_ROOT="${C1_AUX_DEPTH_ROOT:-${SCENE_ROOT}/train/sparse_depth}"

if [[ -z "${MEMORY_PATH}" || ! -s "${MEMORY_PATH}" ]]; then
  echo "ERROR: MEMORY_PATH is required and must exist." >&2
  exit 2
fi
if [[ ! -d "${SCENE_ROOT}/train/rgb" || ! -d "${SCENE_ROOT}/train/poses" || ! -d "${SCENE_ROOT}/train/calibration" ]]; then
  echo "ERROR: incomplete ACE train scene: ${SCENE_ROOT}" >&2
  exit 2
fi

if [[ -z "${KEYFRAME_CHANNEL}" ]]; then
  SIDECAR_NAME="${SIDECAR_NAME:-colmap_keyframe_channel_v1_sp_strict_dino_r${IMAGE_RESOLUTION}_w${BUFFER_IMAGE_WIDTH}}"
  if [[ -n "${SIDECAR_ROOT:-}" ]]; then
    KEYFRAME_CHANNEL="${SIDECAR_ROOT}/${SCENE_NAME}/train/${SIDECAR_NAME}/keyframe_channel.npz"
  else
    KEYFRAME_CHANNEL="${SCENE_ROOT}/train/${SIDECAR_NAME}/keyframe_channel.npz"
  fi
fi
if [[ ! -s "${KEYFRAME_CHANNEL}" ]]; then
  echo "ERROR: missing DINO STGS keyframe channel: ${KEYFRAME_CHANNEL}" >&2
  echo "Build it with ace_dinov2_lmc/scripts/build_dino_stgs_keyframe_channel.sh." >&2
  exit 2
fi

CMD=(
  env -u CUDA_VISIBLE_DEVICES
  conda run --no-capture-output -n "${CONDA_ENV}" "${PYTHON_BIN}" ace_dinov2_lmc/train_ace_dinov2_lmc.py
  "${SCENE_ROOT}"
  "${OUTPUT_MAP_SUFFIX}"
  --data_backend ace
  --post_train_eval_scene "${POST_TRAIN_EVAL_SCENE}"
  --use_lmc True
  --memory_path "${MEMORY_PATH}"
  --dinov2_path "${DINOV2_PATH}"
  --train_preset "${TRAIN_PRESET}"
  --model_backend ace_dinov2
  --lmc_flow "${LMC_FLOW}"
  --lmc_mode "${LMC_MODE}"
  --lmc_iterations "${LMC_ITERATIONS}"
  --num_latent_tokens "${NUM_LATENT_TOKENS}"
  --best_metric "${BEST_METRIC}"
  --device "${DEVICE}"
  --post_train_eval_device "${EVAL_DEVICE}"
  --training_buffer_size "${TRAINING_BUFFER_SIZE}"
  --buffer_size_final "${BUFFER_SIZE_FINAL}"
  --buffer_batch_size "${BUFFER_BATCH_SIZE}"
  --buffer_image_width "${BUFFER_IMAGE_WIDTH}"
  --buffer_on_cpu "${BUFFER_ON_CPU}"
  --buffer_on_cpu_final "${BUFFER_ON_CPU_FINAL}"
  --samples_per_image "${SAMPLES_PER_IMAGE}"
  --batch_size "${BATCH_SIZE}"
  --image_resolution "${IMAGE_RESOLUTION}"
  --num_data_loader_workers "${NUM_DATA_LOADER_WORKERS}"
  --eval_num_workers "${EVAL_NUM_WORKERS}"
  --experiment_root "${RUN_ROOT}"
  --experiment_subdir "${EXPERIMENT_SUBDIR}"
  --post_train_hypotheses "${POST_TRAIN_HYPOTHESES}"
  --post_train_eval_seeds ${POST_TRAIN_EVAL_SEEDS}
  --eval_deterministic "${EVAL_DETERMINISTIC}"
  --eval_dsacstar_seed "${EVAL_DSACSTAR_SEED}"
  --eval_dsacstar_seed_per_frame "${EVAL_DSACSTAR_SEED_PER_FRAME}"
  --iteration_eval_seed "${ITERATION_EVAL_SEED}"
  --iteration_eval_hypotheses "${ITERATION_EVAL_HYPOTHESES}"
  --buffer_sample_valid_coords "${BUFFER_SAMPLE_VALID_COORDS}"
  --buffer_valid_coord_sample_ratio "${BUFFER_VALID_COORD_SAMPLE_RATIO}"
  --buffer_valid_coord_neighbor_radius "${BUFFER_VALID_COORD_NEIGHBOR_RADIUS}"
  --buffer_valid_coord_neighbor_mode "${BUFFER_VALID_COORD_NEIGHBOR_MODE}"
  --use_sfm_track_guided_sampling "${USE_STGS_GUIDED_SAMPLING}"
  --use_sfm_track_inter_frame_loss "${USE_STGS_INTER_FRAME_LOSS}"
  --sfm_track_keyframe_channel_path "${KEYFRAME_CHANNEL}"
  --sfm_track_sidecar_mismatch_policy "${SFM_TRACK_SIDECAR_MISMATCH_POLICY}"
  --sfm_track_guided_batch_size "${SFM_TRACK_GUIDED_BATCH_SIZE}"
  --sfm_track_guided_sampling_strategy "${SFM_TRACK_GUIDED_SAMPLING_STRATEGY}"
  --sfm_track_guided_fraction "${SFM_TRACK_GUIDED_FRACTION}"
  --sfm_track_guided_mode "${SFM_TRACK_GUIDED_MODE}"
  --sfm_track_guided_main_loss_mode "${SFM_TRACK_GUIDED_MAIN_LOSS_MODE}"
  --sfm_track_guided_source_target_mode "${SFM_TRACK_GUIDED_SOURCE_TARGET_MODE}"
  --sfm_track_inter_frame_target_mode "${SFM_TRACK_INTER_FRAME_TARGET_MODE}"
  --sfm_track_guided_aux_normalizer "${SFM_TRACK_GUIDED_AUX_NORMALIZER}"
  --sfm_track_anchor_self_weight "${SFM_TRACK_ANCHOR_SELF_WEIGHT}"
  --sfm_track_anchor_use_alignment_weight "${SFM_TRACK_ANCHOR_USE_ALIGNMENT_WEIGHT}"
  --sfm_track_inter_frame_weight "${SFM_TRACK_INTER_FRAME_WEIGHT}"
  --sfm_track_inter_frame_start_ratio "${SFM_TRACK_INTER_FRAME_START_RATIO}"
  --sfm_track_inter_frame_decay_last_ratio "${SFM_TRACK_INTER_FRAME_DECAY_LAST_RATIO}"
  --sfm_track_inter_frame_max_px "${SFM_TRACK_INTER_FRAME_MAX_PX}"
  --sfm_track_inter_frame_dropout "${SFM_TRACK_INTER_FRAME_DROPOUT}"
  --sfm_track_min_alignment_weight "${SFM_TRACK_MIN_ALIGNMENT_WEIGHT}"
  --sfm_track_max_colmap_reproj_error_px "${SFM_TRACK_MAX_COLMAP_REPROJ_ERROR_PX}"
  --sfm_track_min_track_length "${SFM_TRACK_MIN_TRACK_LENGTH}"
  --sfm_track_max_anchor_patch_offset_px "${SFM_TRACK_MAX_ANCHOR_PATCH_OFFSET_PX}"
  --sfm_track_max_target_patch_offset_px "${SFM_TRACK_MAX_TARGET_PATCH_OFFSET_PX}"
)

if [[ -d "${C1_AUX_DEPTH_ROOT}" ]]; then
  CMD+=(--c1_aux_depth_root "${C1_AUX_DEPTH_ROOT}" --c1_aux_depth_kind "${C1_AUX_DEPTH_KIND}")
else
  echo "WARN: C1_AUX_DEPTH_ROOT not found, running without sparse-depth aux path: ${C1_AUX_DEPTH_ROOT}" >&2
fi

printf 'ACE_DINO_LMC_STGS_CMD: '
printf '%q ' "${CMD[@]}"
printf '
'

if [[ "${DRY_RUN}" == "true" ]]; then
  exit 0
fi

"${CMD[@]}"
