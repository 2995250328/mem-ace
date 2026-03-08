#!/usr/bin/env bash
set -euo pipefail

# =========================
# 默认路径（可通过环境变量覆盖：SCENE=... MEM=... DINO=... ./run_smoke_4gpu.sh）
# 从 ace_depth/ 运行时 ROOT 为项目根目录
# =========================
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
TRAIN_PY="$SCRIPT_DIR/train_ace_dinov2_lmc.py"

SCENE="${SCENE:-/data/xwh/7Scenes/pgt_7scenes_heads}"
MEM="${MEM:-/home/xwh/project/map-anything-experiments/memory_extract/heads_train/20views/20260127_235757/7Scenes_heads_train_pooled_GT_patch.pt}"
DINO="${DINO:-/data/xwh/checkpoints/dinov2_vitl14_pretrain.pth}"
ITER_S1BUF_REFILL_MODE="${ITER_S1BUF_REFILL_MODE:-partial}"
ITER_S1BUF_KEEP_RATIO="${ITER_S1BUF_KEEP_RATIO:-0.5}"

TS="$(date +%m%d_%H%M%S)"
EXP="${EXP:-$ROOT/output/smoke_$TS}"
mkdir -p "$EXP"

echo "[INFO] EXP=$EXP"
echo "[INFO] SCENE=$SCENE"
echo "[INFO] MEM=$MEM"
echo "[INFO] DINO=$DINO"

[[ -f "$TRAIN_PY" ]] || { echo "[ERR] train script not found: $TRAIN_PY"; exit 1; }
[[ -e "$SCENE" ]] || { echo "[ERR] scene path not found: $SCENE"; exit 1; }
[[ -f "$MEM" ]] || { echo "[ERR] memory file not found: $MEM"; exit 1; }
[[ -f "$DINO" ]] || { echo "[ERR] dinov2 ckpt not found: $DINO"; exit 1; }

COMMON_ARGS=(
  --dinov2_path "$DINO"
  --use_lmc True
  --memory_path "$MEM"
  --lmc_iterations 2
  --lmc_warmup_steps 20
  --lmc_train_steps 10
  --epochs 1
  --training_buffer_size 32768
  --batch_size 1024
  --samples_per_image 64
  --buffer_batch_size 2
  --buffer_on_cpu True
  --image_resolution 224
  --eval_each_iteration False
  --eval_after_train False
  --use_half False
)

run_job() {
  local gpu="$1"; shift
  local name="$1"; shift
  local outfile="$1"; shift
  echo "[LAUNCH] gpu=$gpu name=$name" >&2
  # 传 --device cuda:$gpu，脚本内 setup_cuda_environment() 会据此设置 CUDA_VISIBLE_DEVICES，每进程占一卡
  python "$TRAIN_PY" "$SCENE" "$name.pt" \
    --device "cuda:$gpu" \
    "${COMMON_ARGS[@]}" \
    "$@" \
    --run_name "$name" \
    > "$outfile" 2>&1 &
}

# 4种模式并行（必须在当前 shell 后台运行，否则 wait 无法等待）
# gpu0: online S1 每步跑 encoder，显存大，单独减小 s1_batch_size
run_job 0 "smoke_iter_online" "$EXP/gpu0_iter_online.log" \
  --lmc_flow iterative \
  --s1_use_buffer False \
  --s1_batch_size 2
PID0=$!

run_job 1 "smoke_iter_s1buf_${ITER_S1BUF_REFILL_MODE}" "$EXP/gpu1_iter_s1buf_${ITER_S1BUF_REFILL_MODE}.log" \
  --lmc_flow iterative \
  --s1_use_buffer True \
  --s1_loss_mode sample_per_image \
  --s1_buffer_refill_mode "$ITER_S1BUF_REFILL_MODE" \
  --s1_buffer_keep_ratio "$ITER_S1BUF_KEEP_RATIO"
PID1=$!

run_job 2 "smoke_aceg_r1" "$EXP/gpu2_aceg_r1.log" \
  --lmc_flow ace_g \
  --ace_g_fusion_in_s2 False \
  --s1_use_buffer True \
  --s1_loss_mode sample_per_image \
  --s1_buffer_refill_mode full
PID2=$!

run_job 3 "smoke_aceg_r2" "$EXP/gpu3_aceg_r2.log" \
  --lmc_flow ace_g \
  --ace_g_fusion_in_s2 True \
  --ace_g_fusion_lr_ratio 0.01 \
  --ace_g_cross_iter_eval True \
  --s1_use_buffer True \
  --s1_loss_mode sample_per_image \
  --s1_buffer_refill_mode full \
  --batch_size 512
PID3=$!

echo "[INFO] PIDs: $PID0 $PID1 $PID2 $PID3"
wait "$PID0" "$PID1" "$PID2" "$PID3" || true
echo "[INFO] all jobs finished"

check_log() {
  local f="$1"
  echo "------------------------------"
  echo "[CHECK] $f"
  [[ -f "$f" ]] || { echo "[MISS] log not found"; return; }

  grep -E -n "Memory check|\[LMC preflight\]|S1 partial refill|S1 data source:|\[S1-Buffer\]|\[S2-G\]|Cross-iter eval|Done ACE-G|Done\." "$f" 2>/dev/null || true
  grep -E -n "Traceback|RuntimeError|ValueError|CUDA out of memory|mat1 and mat2 must have the same dtype|not implemented" "$f" 2>/dev/null || true
}

check_log "$EXP/gpu0_iter_online.log"
check_log "$EXP/gpu1_iter_s1buf_${ITER_S1BUF_REFILL_MODE}.log"
check_log "$EXP/gpu2_aceg_r1.log"
check_log "$EXP/gpu3_aceg_r2.log"

echo "=============================="
echo "[DONE] logs dir: $EXP"
echo "检查点："
echo "1) 入口日志包含 Memory check / [LMC preflight]"
echo "2) iterative s1buf 任务包含 [S1-Buffer]，且当 mode=partial 时出现 partial refill/fallback_full"
echo "3) ace_g 任务包含 [S2-G]"
echo "4) 末尾有 Done / Done ACE-G"
echo "5) 无 Traceback/OOM/dtype mismatch"
