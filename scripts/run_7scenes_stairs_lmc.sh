#!/usr/bin/env bash
#
# 7Scenes stairs 单场景 LMC 训练（ACE DINOv2 + GeoLMC）
# 对齐 map-anything bash_scripts/tasks/train_ace.sh 的「小步快跑」策略，促进 S2 收敛：
#   - 更多 LMC 轮次 + 后续轮次更短 S1（500 步），更频繁 S1-S2 同步
#   - 首轮 S1 关早停或放宽，保证首轮 compressor 充分训练
#   - 最后一轮使用大 buffer (8M)，S2 更多 epochs
#
# 用法（在 ace_depth 仓库根目录执行）：
#   ./scripts/run_7scenes_stairs_lmc.sh
#
# 可选环境变量：
#   DEVICE=cuda:0         默认 cuda:0
#   MEMORY_PATH=<path>    覆盖 memory 路径（默认见下方）
#
SCRIPT_PATH=$(dirname "$(realpath -s "$0")")
REPO_PATH=$(realpath -s "${SCRIPT_PATH}/..")
cd "$REPO_PATH" || exit 1

DEVICE="${DEVICE:-cuda:0}"
MEMORY_PATH="${MEMORY_PATH:-/home/xwh/project/map-anything-experiments/memory_extract/stairs_train/20views/20260126_234545/7Scenes_stairs_train_pooled_GT.pt}"

python train_ace_dinov2_lmc.py \
  /data/xwh/7Scenes/pgt_7scenes_stairs \
  output/chess_lmc.pt \
  --device "$DEVICE" \
  --use_lmc True \
  --memory_path "$MEMORY_PATH" \
  --lmc_mode global \
  --num_latent_tokens 64 \
  --lmc_iterations 30 \
  --lmc_warmup_steps 2000 \
  --lmc_train_steps 500 \
  --buffer_batch_size 10 \
  --training_buffer_size 2560000 \
  --buffer_size_final 8192000 \
  --samples_per_image 384 \
  --batch_size 5120 \
  --image_resolution 518 \
  --epochs 28 \
  --s1_learning_rate_max 5e-5 \
  --s2_learning_rate_max 1e-3 \
  --s2_lr_boost_first 2.0 \
  --s1_batch_size 4 \
  --s1_lr_scale_later 0.2 \
  --s1_early_stop False \
  --lmc_lr_div_factor 40 \
  --s1_empty_cache_interval 20 \
  --output_layout hierarchical \
  --eval_each_iteration True \
  --keep_best_only True \
  --best_metric pct5
