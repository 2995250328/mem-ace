#!/usr/bin/env bash
set -euo pipefail

# Run the ACE-FCN + MapAnything/BSE-memory ablation end to end:
#   Stage 1: ACE-FCN LMC with mismatched MapAnything/BSE memory.
#   Stage 2: load Stage-1 local stack, concat GLACE global feature, train final head.
#
# Run from an activated environment, e.g.:
#   cd /home/xwh/project/ace_depth
#   conda activate mapanything
#   bash ace_dinov2_lmc/scripts/run_ace_fcn_mapanything_two_stage.sh

ROOT_DIR="${ROOT_DIR:-/home/xwh/project/ace_depth}"
SCENE="${SCENE:-/data/xwh/Wayspots/wayspots_bears}"
MEMORY_PATH="${MEMORY_PATH:-${ROOT_DIR}/ace_dinov2_lmc/04_evaluation/wayspots_lmc/memory/wayspots_bears/32v_v0.05_bilinear_bse_ut0.02_noaug_asb_adaptive_pool1.0_rgate4_prepair8_sor_gm_l2/20260525_223355/memory_bse.pt}"
ACE_ENCODER_PATH="${ACE_ENCODER_PATH:-${ROOT_DIR}/ace_encoder_pretrained.pt}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-/data/xwh/ace_dinov2_lmc/04_evaluation/ace_fcn_lmc_two_stage}"

STAGE1_SUBDIR="${STAGE1_SUBDIR:-stage1_mapanything_memory_it12}"
STAGE2_SUBDIR="${STAGE2_SUBDIR:-stage2_mapanything_memory_glace_concat_it12}"
STAGE1_OUT="${STAGE1_OUT:-ace_fcn_mapanything_memory_stage1.pt}"
STAGE2_OUT="${STAGE2_OUT:-ace_fcn_mapanything_memory_glace_stage2.pt}"

GPU_STAGE1="${GPU_STAGE1:-0}"
GPU_STAGE2="${GPU_STAGE2:-1}"
LMC_ITERATIONS="${LMC_ITERATIONS:-12}"
NUM_LATENT_TOKENS="${NUM_LATENT_TOKENS:-64}"
IMAGE_RESOLUTION="${IMAGE_RESOLUTION:-512}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
TRAINING_BUFFER_SIZE="${TRAINING_BUFFER_SIZE:-2800000}"
BUFFER_SIZE_FINAL="${BUFFER_SIZE_FINAL:-7600000}"
SAMPLES_PER_IMAGE="${SAMPLES_PER_IMAGE:-512}"
POST_TRAIN_HYPOTHESES="${POST_TRAIN_HYPOTHESES:-256}"
POST_TRAIN_EVAL_SEEDS=(${POST_TRAIN_EVAL_SEEDS:-1305 2026 4242})
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "${ROOT_DIR}"

echo "[Stage1] ACE-FCN query + MapAnything/BSE memory"
"${PYTHON_BIN}" ace_dinov2_lmc/train_ace_dinov2_lmc.py \
  "${SCENE}" "${STAGE1_OUT}" \
  --model_backend ace_fcn_lmc \
  --data_backend ace \
  --use_lmc True \
  --lmc_flow ace_g \
  --memory_path "${MEMORY_PATH}" \
  --ace_lmc_allow_mismatched_memory True \
  --ace_encoder_path "${ACE_ENCODER_PATH}" \
  --ace_lmc_global_head_mode none \
  --device "cuda:${GPU_STAGE1}" \
  --post_train_eval_device "cuda:${GPU_STAGE1}" \
  --experiment_root "${EXPERIMENT_ROOT}" \
  --experiment_subdir "${STAGE1_SUBDIR}" \
  --lmc_iterations "${LMC_ITERATIONS}" \
  --num_latent_tokens "${NUM_LATENT_TOKENS}" \
  --ace_g_fusion_in_s2 True \
  --ace_g_cross_iter_eval True \
  --s1_use_buffer True \
  --s1_loss_mode sample_per_image \
  --s1_buffer_refill_mode full \
  --image_resolution "${IMAGE_RESOLUTION}" \
  --batch_size "${BATCH_SIZE}" \
  --training_buffer_size "${TRAINING_BUFFER_SIZE}" \
  --buffer_size_final "${BUFFER_SIZE_FINAL}" \
  --buffer_on_cpu True \
  --buffer_on_cpu_final True \
  --samples_per_image "${SAMPLES_PER_IMAGE}" \
  --post_train_eval_seeds "${POST_TRAIN_EVAL_SEEDS[@]}" \
  --post_train_hypotheses "${POST_TRAIN_HYPOTHESES}"

STAGE1_SEARCH_ROOT="${EXPERIMENT_ROOT}/${STAGE1_SUBDIR}"
STAGE1_PATTERN="best_K${NUM_LATENT_TOKENS}_it${LMC_ITERATIONS}_${STAGE1_OUT}"
STAGE1_CKPT="$(find "${STAGE1_SEARCH_ROOT}" -type f -name "${STAGE1_PATTERN}" | sort | tail -n 1)"

if [[ -z "${STAGE1_CKPT}" ]]; then
  echo "ERROR: Stage1 checkpoint not found under ${STAGE1_SEARCH_ROOT} matching ${STAGE1_PATTERN}" >&2
  exit 2
fi

echo "[Stage2] Using Stage1 checkpoint: ${STAGE1_CKPT}"
"${PYTHON_BIN}" ace_dinov2_lmc/train_ace_dinov2_lmc.py \
  "${SCENE}" "${STAGE2_OUT}" \
  --model_backend ace_fcn_lmc \
  --data_backend ace \
  --use_lmc True \
  --lmc_flow ace_g \
  --memory_path "${MEMORY_PATH}" \
  --ace_lmc_allow_mismatched_memory True \
  --ace_encoder_path "${ACE_ENCODER_PATH}" \
  --ace_lmc_global_head_mode glace_concat \
  --ace_lmc_local_checkpoint_path "${STAGE1_CKPT}" \
  --ace_lmc_freeze_local_stack True \
  --glace_feat_name features.npy \
  --device "cuda:${GPU_STAGE2}" \
  --post_train_eval_device "cuda:${GPU_STAGE2}" \
  --experiment_root "${EXPERIMENT_ROOT}" \
  --experiment_subdir "${STAGE2_SUBDIR}" \
  --lmc_iterations "${LMC_ITERATIONS}" \
  --num_latent_tokens "${NUM_LATENT_TOKENS}" \
  --ace_g_fusion_in_s2 True \
  --ace_g_cross_iter_eval True \
  --s1_use_buffer True \
  --s1_loss_mode sample_per_image \
  --s1_buffer_refill_mode full \
  --image_resolution "${IMAGE_RESOLUTION}" \
  --batch_size "${BATCH_SIZE}" \
  --training_buffer_size "${TRAINING_BUFFER_SIZE}" \
  --buffer_size_final "${BUFFER_SIZE_FINAL}" \
  --buffer_on_cpu True \
  --buffer_on_cpu_final True \
  --samples_per_image "${SAMPLES_PER_IMAGE}" \
  --post_train_eval_seeds "${POST_TRAIN_EVAL_SEEDS[@]}" \
  --post_train_hypotheses "${POST_TRAIN_HYPOTHESES}"

echo "[Done] Stage1 and Stage2 completed."
