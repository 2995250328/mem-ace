#!/usr/bin/env bash
#
# 7Scenes heads - LMC map-anything flow 对照预设
# 目标：固定关键口径，减少手工传参漂移。
#
# 用法（在 ace_depth 根目录）:
#   bash scripts/run_7scenes_heads_lmc_mapany_flow.sh
#
# 可选环境变量：
#   DEVICE=cuda:0
#   SCENE_PATH=/data/xwh/7Scenes/pgt_7scenes_heads
#   MEMORY_PATH=/home/xwh/project/map-anything-experiments/memory_extract/heads_train/20views/20260127_235757/7Scenes_heads_train_pooled_GT_patch.pt
#   OUTPUT_STEM=chess_lmc.pt
#   LMC_MODE=global
#   LMC_ITERS=10
#   S1_WARMUP_STEPS=1600
#   S1_TRAIN_STEPS=400
#   S1_LR_MAX=1e-4
#   S2_LR_MAX=1e-3
#   BUFFER_SIZE=2560000
#   BUFFER_SIZE_FINAL=7680000
#   EPOCHS=24
#   BATCH_SIZE=5120
#   SAMPLES_PER_IMAGE=384
#   S1_BATCH_SIZE=2
#   EVAL_EACH_ITER=True
#   EVAL_AFTER_TRAIN=True
#   RUN_NAME=auto

set -euo pipefail

SCRIPT_PATH=$(dirname "$(realpath -s "$0")")
REPO_PATH=$(realpath -s "${SCRIPT_PATH}/..")
cd "$REPO_PATH" || exit 1

DEVICE="${DEVICE:-cuda:0}"
SCENE_PATH="${SCENE_PATH:-/data/xwh/7Scenes/pgt_7scenes_heads}"
MEMORY_PATH="${MEMORY_PATH:-/home/xwh/project/map-anything-experiments/memory_extract/heads_train/20views/20260127_235757/7Scenes_heads_train_pooled_GT_patch.pt}"
OUTPUT_STEM="${OUTPUT_STEM:-chess_lmc.pt}"
RUN_NAME="${RUN_NAME:-auto}"

LMC_MODE="${LMC_MODE:-global}"
LMC_ITERS="${LMC_ITERS:-28}"
S1_WARMUP_STEPS="${S1_WARMUP_STEPS:-2000}"
S1_TRAIN_STEPS="${S1_TRAIN_STEPS:-500}"
S1_LR_MAX="${S1_LR_MAX:-1e-4}"
S2_LR_MAX="${S2_LR_MAX:-1e-3}"

BUFFER_SIZE="${BUFFER_SIZE:-3840000}"
BUFFER_SIZE_FINAL="${BUFFER_SIZE_FINAL:-10000000}"
EPOCHS="${EPOCHS:-24}"
BATCH_SIZE="${BATCH_SIZE:-5120}"
SAMPLES_PER_IMAGE="${SAMPLES_PER_IMAGE:-384}"
S1_BATCH_SIZE="${S1_BATCH_SIZE:-12}"

EVAL_EACH_ITER="${EVAL_EACH_ITER:-True}"
EVAL_AFTER_TRAIN="${EVAL_AFTER_TRAIN:-True}"

echo "============================================================"
echo "7Scenes heads | mapany_flow_v1 preset"
echo "Device=${DEVICE} | Mode=${LMC_MODE} | Iters=${LMC_ITERS}"
echo "S1: warmup=${S1_WARMUP_STEPS}, train=${S1_TRAIN_STEPS}, lr_max=${S1_LR_MAX}"
echo "S2: epochs=${EPOCHS}, lr_max=${S2_LR_MAX}, batch=${BATCH_SIZE}"
echo "Buffer: base=${BUFFER_SIZE}, final=${BUFFER_SIZE_FINAL}, samples_per_image=${SAMPLES_PER_IMAGE}"
echo "Scene=${SCENE_PATH}"
echo "Memory=${MEMORY_PATH}"
echo "============================================================"

python train_ace_dinov2_lmc.py \
  "${SCENE_PATH}" \
  "${OUTPUT_STEM}" \
  --device "${DEVICE}" \
  --run_name "${RUN_NAME}" \
  --use_lmc True \
  --lmc_profile mapany_flow_v1 \
  --memory_path "${MEMORY_PATH}" \
  --lmc_mode "${LMC_MODE}" \
  --num_latent_tokens 64 \
  --lmc_iterations "${LMC_ITERS}" \
  --lmc_warmup_steps "${S1_WARMUP_STEPS}" \
  --lmc_train_steps "${S1_TRAIN_STEPS}" \
  --s1_learning_rate_max "${S1_LR_MAX}" \
  --s2_learning_rate_max "${S2_LR_MAX}" \
  --training_buffer_size "${BUFFER_SIZE}" \
  --buffer_size_final "${BUFFER_SIZE_FINAL}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --samples_per_image "${SAMPLES_PER_IMAGE}" \
  --s1_batch_size "${S1_BATCH_SIZE}" \
  --eval_each_iteration "${EVAL_EACH_ITER}" \
  --eval_after_train "${EVAL_AFTER_TRAIN}" \
  --keep_best_only True \
  --best_metric pct5

