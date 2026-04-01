#!/usr/bin/env bash
#
# mapany_flow_v1 冒烟脚本（快速门禁）
# 验证点：
#   1) repro_step_mode=global_monotonic
#   2) S2 rewind=0
#   3) Buffer sampling replacement=False 且 duplicate_ratio 接近 0
#
# 用法：
#   bash scripts/run_7scenes_heads_lmc_mapany_flow_smoke.sh

set -euo pipefail

SCRIPT_PATH=$(dirname "$(realpath -s "$0")")
REPO_PATH=$(realpath -s "${SCRIPT_PATH}/..")
cd "$REPO_PATH" || exit 1

DEVICE="${DEVICE:-cuda:0}"
SCENE_PATH="${SCENE_PATH:-/mnt/storage/xwh/7Scenes/pgt_7scenes_heads}"
MEMORY_PATH="${MEMORY_PATH:-/home/xwh/project/map-anything-experiments/memory_extract/heads_train/20views/20260127_235757/7Scenes_heads_train_pooled_GT_patch.pt}"

python train_ace_dinov2_lmc.py \
  "${SCENE_PATH}" \
  output/chess_lmc.pt \
  --device "${DEVICE}" \
  --use_lmc True \
  --lmc_profile mapany_flow_v1 \
  --memory_path "${MEMORY_PATH}" \
  --lmc_mode global \
  --lmc_iterations 1 \
  --lmc_warmup_steps 60 \
  --epochs 1 \
  --training_buffer_size 128000 \
  --batch_size 5120 \
  --samples_per_image 384 \
  --s1_batch_size 2 \
  --eval_each_iteration False \
  --eval_after_train False \
  --keep_best_only True

