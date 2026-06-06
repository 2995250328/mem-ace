#!/usr/bin/env bash
set -euo pipefail

# Non-destructive resume for wayspots_therock K1024 GLACE+LMC.
# It initializes from a previous best checkpoint but writes a new run directory.
# Usage:
#   bash ace_dinov2_lmc/scripts/run_wayspots_therock_k1024_resume_bigbuffer_gpu3.sh
#   CONTINUE_ITERS=2 bash ace_dinov2_lmc/scripts/run_wayspots_therock_k1024_resume_bigbuffer_gpu3.sh
#   DRY_RUN=true bash ace_dinov2_lmc/scripts/run_wayspots_therock_k1024_resume_bigbuffer_gpu3.sh

ROOT_DIR="${ROOT_DIR:-/home/xwh/project/ace_depth}"
CONDA_ENV="${CONDA_ENV:-mapanything}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-3}"
DRY_RUN="${DRY_RUN:-false}"

RUN_ROOT="${RUN_ROOT:-/home/xwh/project/ace_depth/ace_dinov2_lmc/04_evaluation/wayspots_ace_fcn_lmc_suite/20260530_181745}"
SOURCE_RUN_DIR="${SOURCE_RUN_DIR:-${RUN_ROOT}/stage2_glace_concat_it12_K1024/extracted/wayspots_therock/dino_ace_lmc_ace_g/20260605_164730_aceg_fS2cie_global_res512_buf2.8M_F7.6M_K1024_it12_ep24_bs4096_spi_s1buf_sp512_onecycle_improved}"
META_PATH="${META_PATH:-${SOURCE_RUN_DIR}/best_checkpoint_meta.json}"

WAYSPOTS_ROOT="${WAYSPOTS_ROOT:-/data/xwh/Wayspots}"
SCENE="${SCENE:-wayspots_therock}"
SCENE_ROOT="${WAYSPOTS_ROOT}/${SCENE}"
MEMORY_PATH="${MEMORY_PATH:-${RUN_ROOT}/memory/${SCENE}/memory_ace_fcn_sparse_sp_r4.pt}"
ACE_ENCODER_PATH="${ACE_ENCODER_PATH:-/home/xwh/project/ace_depth/ace_encoder_pretrained.pt}"
GLACE_FEAT_NAME="${GLACE_FEAT_NAME:-features.npy}"
AUX_DEPTH_DIR="${AUX_DEPTH_DIR:-${SCENE_ROOT}/train/sparse_depth}"

NUM_LATENT_TOKENS="${NUM_LATENT_TOKENS:-1024}"
CONTINUE_ITERS="${CONTINUE_ITERS:-1}"
IMAGE_RESOLUTION="${IMAGE_RESOLUTION:-512}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
BIG_BUFFER_SIZE="${BIG_BUFFER_SIZE:-7600000}"
SAMPLES_PER_IMAGE="${SAMPLES_PER_IMAGE:-512}"
S2_LEARNING_RATE_MAX="${S2_LEARNING_RATE_MAX:-0.0002}"
S2_LR_BOOST_FIRST="${S2_LR_BOOST_FIRST:-1.0}"
S2_LR_BOOST_LATER="${S2_LR_BOOST_LATER:-1.0}"
ACE_G_FUSION_LR_RATIO="${ACE_G_FUSION_LR_RATIO:-0.005}"
HEAD_LR_MULTIPLIER_S2="${HEAD_LR_MULTIPLIER_S2:-1.0}"
POST_TRAIN_HYPOTHESES="${POST_TRAIN_HYPOTHESES:-256}"
POST_TRAIN_EVAL_SEEDS=(${POST_TRAIN_EVAL_SEEDS:-1305 2026 4242})
EXPERIMENT_SUBDIR="${EXPERIMENT_SUBDIR:-stage2_glace_concat_it12_K1024_therock_resume_bigbuf}"
RUN_NAME="${RUN_NAME:-resume_bigbuf_from_iter3}"
LOG_DIR="${LOG_DIR:-${RUN_ROOT}/token_sweep_logs_therock_resume_bigbuf}"

if [[ ! -f "${META_PATH}" ]]; then
  echo "Missing meta: ${META_PATH}" >&2
  exit 2
fi
if ! command -v jq >/dev/null 2>&1; then
  echo "jq is required to read ${META_PATH}" >&2
  exit 2
fi

RESUME_CKPT="$(jq -r '.best_checkpoint_path' "${META_PATH}")"
BEST_ITER="$(jq -r '.best_iter' "${META_PATH}")"
BEST_SCORE="$(jq -r '.best_score' "${META_PATH}")"
STAGE1_CKPT="$(jq -r '.ace_lmc_local_checkpoint_path' "${META_PATH}")"
TOTAL_ITERS="$((BEST_ITER + CONTINUE_ITERS))"

if [[ ! -f "${RESUME_CKPT}" ]]; then
  echo "Missing resume checkpoint: ${RESUME_CKPT}" >&2
  exit 2
fi
if [[ ! -f "${STAGE1_CKPT}" ]]; then
  echo "Missing stage1 checkpoint from meta: ${STAGE1_CKPT}" >&2
  exit 2
fi
if (( TOTAL_ITERS <= BEST_ITER )); then
  echo "TOTAL_ITERS=${TOTAL_ITERS} must be greater than BEST_ITER=${BEST_ITER}" >&2
  exit 2
fi

mkdir -p "${LOG_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/train_${SCENE}_K${NUM_LATENT_TOKENS}_resume_bigbuf_${STAMP}.log"

cmd=(
  conda run --no-capture-output -n "${CONDA_ENV}" "${PYTHON_BIN}" ace_dinov2_lmc/train_ace_dinov2_lmc.py
  "${SCENE_ROOT}" ace_fcn_glace_global_stage2_resume_bigbuffer.pt
  --model_backend ace_fcn_lmc
  --data_backend ace
  --use_lmc True
  --lmc_flow ace_g
  --memory_path "${MEMORY_PATH}"
  --use_scale_token False
  --ace_encoder_path "${ACE_ENCODER_PATH}"
  --ace_lmc_global_head_mode glace_concat
  --ace_lmc_local_checkpoint_path "${STAGE1_CKPT}"
  --ace_lmc_freeze_local_stack True
  --glace_feat_name "${GLACE_FEAT_NAME}"
  --device "cuda:${GPU_ID}"
  --post_train_eval_device "cuda:${GPU_ID}"
  --experiment_root "${RUN_ROOT}"
  --experiment_subdir "${EXPERIMENT_SUBDIR}"
  --run_name "${RUN_NAME}"
  --resume_checkpoint_path "${RESUME_CKPT}"
  --resume_meta_path "${META_PATH}"
  --resume_best_iter "${BEST_ITER}"
  --resume_best_score "${BEST_SCORE}"
  --lmc_iterations "${TOTAL_ITERS}"
  --num_latent_tokens "${NUM_LATENT_TOKENS}"
  --ace_g_fusion_in_s2 True
  --ace_g_cross_iter_eval True
  --s2_learning_rate_max "${S2_LEARNING_RATE_MAX}"
  --s2_lr_boost_first "${S2_LR_BOOST_FIRST}"
  --s2_lr_boost_later "${S2_LR_BOOST_LATER}"
  --ace_g_fusion_lr_ratio "${ACE_G_FUSION_LR_RATIO}"
  --head_lr_multiplier_s2 "${HEAD_LR_MULTIPLIER_S2}"
  --s1_use_buffer True
  --s1_loss_mode sample_per_image
  --s1_buffer_refill_mode full
  --image_resolution "${IMAGE_RESOLUTION}"
  --batch_size "${BATCH_SIZE}"
  --training_buffer_size "${BIG_BUFFER_SIZE}"
  --buffer_size_final "${BIG_BUFFER_SIZE}"
  --buffer_on_cpu True
  --buffer_on_cpu_final True
  --samples_per_image "${SAMPLES_PER_IMAGE}"
  --buffer_sample_valid_coords True
  --buffer_valid_coord_sample_ratio 1.0
  --buffer_valid_coord_neighbor_radius 1
  --buffer_valid_coord_neighbor_mode cross
  --c1_aux_depth_root "${AUX_DEPTH_DIR}"
  --c1_aux_depth_kind sparse_depth
  --post_train_eval_seeds "${POST_TRAIN_EVAL_SEEDS[@]}"
  --post_train_hypotheses "${POST_TRAIN_HYPOTHESES}"
)

echo "============================================================"
echo "[resume-bigbuf] ${SCENE} K${NUM_LATENT_TOKENS} on GPU${GPU_ID}"
echo "Source run  : ${SOURCE_RUN_DIR}"
echo "Best iter   : ${BEST_ITER}"
echo "Best score  : ${BEST_SCORE}"
echo "Continue    : ${CONTINUE_ITERS} iter(s), total lmc_iterations=${TOTAL_ITERS}"
echo "Big buffer  : ${BIG_BUFFER_SIZE} samples for every resumed S2 iter"
echo "S2 LR      : max=${S2_LEARNING_RATE_MAX}, head_mult=${HEAD_LR_MULTIPLIER_S2}, boost=(${S2_LR_BOOST_FIRST},${S2_LR_BOOST_LATER}), fusion_ratio=${ACE_G_FUSION_LR_RATIO}"
echo "Log file    : ${LOG_FILE}"
echo "============================================================"

if [[ "${DRY_RUN}" == "true" ]]; then
  printf 'cd %q && ' "${ROOT_DIR}"
  printf '%q ' "${cmd[@]}"
  printf '\n'
  exit 0
fi

cd "${ROOT_DIR}"
"${cmd[@]}" 2>&1 | tee "${LOG_FILE}"
