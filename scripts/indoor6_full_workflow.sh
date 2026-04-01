#!/usr/bin/env bash
# =============================================================================
# Indoor6 多场景完整工作流：先提取 memory，再生成/执行 baseline 与 ACE-G 训练命令
# =============================================================================
# 用法：
#   1) 仅提取 memory（默认 6 个场景）：
#      bash scripts/indoor6_full_workflow.sh extract
#   2) 仅生成训练命令（不执行训练，需已提取过 memory）：
#      bash scripts/indoor6_full_workflow.sh train-cmds
#   3) 提取 + 生成训练命令：
#      bash scripts/indoor6_full_workflow.sh extract train-cmds
#   4) 指定场景与 GPU（环境变量）：
#      SCENES="scene1 scene2a" GPU_EXTRACT=0 GPU_BASELINE=2 GPU_ACEG=3 bash scripts/indoor6_full_workflow.sh extract train-cmds
#
# 环境变量：
#   SCENES          空格分隔的场景名，默认 "scene1 scene2a scene3 scene4a scene5 scene6"
#   GPU_EXTRACT      提取 memory 用的 GPU 编号，默认 0
#   GPU_BASELINE     baseline 训练用的 GPU，默认 2
#   GPU_ACEG         ACE-G 训练用的 GPU，默认 3
#   MEM_ROOT         memory 输出根目录，默认 map-anything-experiments/memory_extract
#   MAPANYTHING_ROOT map-anything 项目根目录，默认 上级目录
#   ACE_ROOT         ace_depth 项目根目录，默认 本脚本所在目录的上级
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MAPANYTHING_ROOT="${MAPANYTHING_ROOT:-$(cd "$ACE_ROOT/../map-anything" 2>/dev/null && pwd)}"
MEM_ROOT="${MEM_ROOT:-$ACE_ROOT/../map-anything-experiments/memory_extract}"
# 解析为绝对路径（若目录不存在则保留原字符串供提示）
if [ -d "$MEM_ROOT" ]; then
  MEM_ROOT="$(cd "$MEM_ROOT" && pwd)"
fi

SCENES="${SCENES:-scene1 scene2a scene3 scene4a scene5 scene6}"
GPU_EXTRACT="${GPU_EXTRACT:-0}"
GPU_BASELINE="${GPU_BASELINE:-2}"
GPU_ACEG="${GPU_ACEG:-3}"
N_VIEWS="${N_VIEWS:-40}"
DATASET="${DATASET:-indoor6_dataset}"

DATA_ROOT="${DATA_ROOT:-/mnt/storage/xwh/indoor6_ace}"
DINOV2_PATH="${DINOV2_PATH:-/mnt/storage/xwh/checkpoints/dinov2_vitl14_pretrain.pth}"

do_extract() {
  if [ ! -d "$MAPANYTHING_ROOT" ]; then
    echo "[ERROR] map-anything 目录不存在: $MAPANYTHING_ROOT" >&2
    echo "        请设置 MAPANYTHING_ROOT 或确保 ace_depth 与 map-anything 同属一个父目录。" >&2
    return 1
  fi
  echo "============================================================"
  echo "Phase 1: 提取 Memory（map-anything）"
  echo "  MAPANYTHING_ROOT=$MAPANYTHING_ROOT"
  echo "  SCENES=$SCENES  N_VIEWS=$N_VIEWS  GPU_ID=$GPU_EXTRACT"
  echo "============================================================"
  cd "$MAPANYTHING_ROOT"
  for scene in $SCENES; do
    scene_train="${scene}_train"
    scene_test="${scene}_test"
    echo ""
    echo ">>> 提取 $scene_train"
    DATASET="$DATASET" SCENE_TRAIN="$scene_train" SCENE_TEST="$scene_test" N_VIEWS="$N_VIEWS" GPU_ID="$GPU_EXTRACT" bash bash_scripts/ace/fps_memory.sh
  done
  echo ""
  echo "[OK] Phase 1 完成。Memory 输出目录: $MEM_ROOT/<scene_train>/${N_VIEWS}views/<timestamp>/"
}

# 解析各场景最新 pooled_GT.pt 路径（Indoor6_<scene_train>_pooled_GT.pt）
get_latest_memory_pt() {
  local scene="$1"
  local scene_train="${scene}_train"
  local dir="$MEM_ROOT/$scene_train/${N_VIEWS}views"
  if [ ! -d "$dir" ]; then
    echo ""
    return 1
  fi
  local latest_ts
  latest_ts=$(ls -1d "$dir"/[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9][0-9][0-9] 2>/dev/null | sort -r | head -1)
  if [ -z "$latest_ts" ]; then
    echo ""
    return 1
  fi
  local pt="$latest_ts/Indoor6_${scene_train}_pooled_GT.pt"
  if [ -f "$pt" ]; then
    echo "$pt"
  else
    echo ""
    return 1
  fi
}

do_train_cmds() {
  echo "============================================================"
  echo "Phase 2: 训练命令（Baseline + ACE-G）"
  echo "  ACE_ROOT=$ACE_ROOT  DATA_ROOT=$DATA_ROOT"
  echo "  GPU_BASELINE=cuda:$GPU_BASELINE  GPU_ACEG=cuda:$GPU_ACEG"
  echo "============================================================"
  for scene in $SCENES; do
    memory_pt="$(get_latest_memory_pt "$scene")"
    if [ -z "$memory_pt" ]; then
      echo ""
      echo "# [$scene] 未找到 memory .pt，请先执行 Phase 1 或检查: $MEM_ROOT/${scene}_train/${N_VIEWS}views/"
      continue
    fi
    echo ""
    echo "# ---------- $scene ----------"
    echo "# Memory: $memory_pt"
    echo ""
    echo "# Baseline (vanilla, 28 iters) — 建议 cuda:$GPU_BASELINE"
    echo "cd $ACE_ROOT"
    echo "python train_ace_dinov2_lmc.py \\"
    echo "    $DATA_ROOT/$scene \\"
    echo "    ${scene}_vanilla_it28.pt \\"
    echo "    --experiment_root output \\"
    echo "    --device cuda:$GPU_BASELINE \\"
    echo "    --use_lmc False \\"
    echo "    --apply_baseline_contract False \\"
    echo "    --vanilla_iterations 28 \\"
    echo "    --use_half False \\"
    echo "    --training_buffer_size 2560000 \\"
    echo "    --samples_per_image 384 \\"
    echo "    --batch_size 5120 \\"
    echo "    --image_resolution 518 \\"
    echo "    --epochs 24 \\"
    echo "    --learning_rate_max 8e-4 \\"
    echo "    --output_layout hierarchical \\"
    echo "    --eval_each_iteration True \\"
    echo "    --keep_best_only True \\"
    echo "    --best_metric pct5 \\"
    echo "    --eval_after_train True"
    echo ""
    echo "# ACE-G full refill — 建议 cuda:$GPU_ACEG"
    echo "python train_ace_dinov2_lmc.py \\"
    echo "    $DATA_ROOT/$scene \\"
    echo "    ${scene}_aceg_full_refill.pt \\"
    echo "    --device cuda:$GPU_ACEG \\"
    echo "    --run_name ${scene}_aceg_full_refill \\"
    echo "    --use_lmc True \\"
    echo "    --memory_path $memory_pt \\"
    echo "    --dinov2_path $DINOV2_PATH \\"
    echo "    --lmc_mode global \\"
    echo "    --num_latent_tokens 64 \\"
    echo "    --lmc_iterations 28 \\"
    echo "    --lmc_warmup_steps 2000 \\"
    echo "    --lmc_train_steps 600 \\"
    echo "    --s1_learning_rate_max 1e-4 \\"
    echo "    --s2_learning_rate_max 1e-3 \\"
    echo "    --training_buffer_size 2560000 \\"
    echo "    --buffer_size_final 7680000 \\"
    echo "    --epochs 24 \\"
    echo "    --batch_size 5120 \\"
    echo "    --samples_per_image 384 \\"
    echo "    --buffer_batch_size 1 \\"
    echo "    --buffer_on_cpu True \\"
    echo "    --image_resolution 518 \\"
    echo "    --s1_batch_size 16 \\"
    echo "    --eval_each_iteration True \\"
    echo "    --eval_after_train True \\"
    echo "    --keep_best_only True \\"
    echo "    --best_metric pct5 \\"
    echo "    --use_half True \\"
    echo "    --lmc_memory_preflight True \\"
    echo "    --lmc_memory_preflight_strict True \\"
    echo "    --lmc_flow ace_g \\"
    echo "    --ace_g_fusion_in_s2 True \\"
    echo "    --ace_g_fusion_lr_ratio 0.01 \\"
    echo "    --ace_g_cross_iter_eval True \\"
    echo "    --s1_use_buffer True \\"
    echo "    --s1_loss_mode sample_per_image \\"
    echo "    --s1_buffer_refill_mode full"
    echo ""
  done
  echo "[OK] 以上命令可直接复制到终端执行；baseline 与 ACE-G 可分别用不同 GPU 并行跑不同场景。"
}

usage() {
  echo "用法: $0 [extract] [train-cmds]"
  echo "  extract     执行 Phase 1：在 map-anything 下对 SCENES 逐场景运行 fps_memory.sh"
  echo "  train-cmds  执行 Phase 2：根据最新 memory 生成 baseline 与 ACE-G 训练命令并打印"
  echo "  可同时指定: $0 extract train-cmds"
  echo ""
  echo "环境变量: SCENES, GPU_EXTRACT, GPU_BASELINE, GPU_ACEG, MEM_ROOT, MAPANYTHING_ROOT, N_VIEWS, DATA_ROOT, DINOV2_PATH"
  exit 0
}

if [ $# -eq 0 ]; then
  usage
fi

while [ $# -gt 0 ]; do
  case "$1" in
    extract)    do_extract ;;
    train-cmds) do_train_cmds ;;
    -h|--help)  usage ;;
    *)          echo "未知参数: $1" >&2; usage ;;
  esac
  shift
done
