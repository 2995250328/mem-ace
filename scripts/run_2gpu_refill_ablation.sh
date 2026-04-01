#!/usr/bin/env bash
#
# 两卡对照实验脚本：系统验证 S1 raw-buffer refill 策略。
# 每一轮只改一个变量：full refill vs partial refill。
#
# 默认顺序：
#   round1: iterative  + full    vs iterative  + partial
#   round2: ace_g R1   + full    vs ace_g R1   + partial
#   round3: ace_g R2   + full    vs ace_g R2   + partial
#
# 用法（在 ace_depth 根目录）:
#   bash scripts/run_2gpu_refill_ablation.sh
#
# 可选环境变量：
#   GPU0=cuda:0
#   GPU1=cuda:1
#   SCENE_PATH=/mnt/storage/xwh/7Scenes/pgt_7scenes_heads
#   MEMORY_PATH=/home/xwh/project/map-anything-experiments/memory_extract/heads_train/20views/20260127_235757/7Scenes_heads_train_pooled_GT_patch.pt
#   DINO_PATH=/mnt/storage/xwh/checkpoints/dinov2_vitl14_pretrain.pth
#   RUN_PREFIX=refill_ablation
#   LMC_MODE=global
#   LMC_ITERS=28
#   S1_WARMUP_STEPS=2000
#   S1_TRAIN_STEPS=600
#   S1_LR_MAX=1e-4
#   S2_LR_MAX=1e-3
#   BUFFER_SIZE=2560000
#   BUFFER_SIZE_FINAL=7680000
#   EPOCHS=24
#   BATCH_SIZE=5120
#   SAMPLES_PER_IMAGE=384
#   BUFFER_BATCH_SIZE=2
#   S1_BATCH_SIZE=2
#   PARTIAL_KEEP_RATIO=0.5
#   EVAL_EACH_ITER=True
#   EVAL_AFTER_TRAIN=True
#   USE_HALF=True
#   RUN_ROUNDS=all              # all / round1 / round2 / round3 / "round1 round3"

set -euo pipefail

SCRIPT_PATH=$(dirname "$(realpath -s "$0")")
REPO_PATH=$(realpath -s "${SCRIPT_PATH}/..")
cd "$REPO_PATH" || exit 1

GPU0="${GPU0:-cuda:2}"
GPU1="${GPU1:-cuda:3}"
SCENE_PATH="${SCENE_PATH:-/mnt/storage/xwh/7Scenes/pgt_7scenes_heads}"
MEMORY_PATH="${MEMORY_PATH:-/home/xwh/project/map-anything-experiments/memory_extract/heads_train/20views/20260127_235757/7Scenes_heads_train_pooled_GT_patch.pt}"
DINO_PATH="${DINO_PATH:-/mnt/storage/xwh/checkpoints/dinov2_vitl14_pretrain.pth}"
RUN_PREFIX="${RUN_PREFIX:-refill_ablation}"

LMC_MODE="${LMC_MODE:-global}"
LMC_ITERS="${LMC_ITERS:-28}"
S1_WARMUP_STEPS="${S1_WARMUP_STEPS:-2000}"
S1_TRAIN_STEPS="${S1_TRAIN_STEPS:-600}"
S1_LR_MAX="${S1_LR_MAX:-1e-4}"
S2_LR_MAX="${S2_LR_MAX:-1e-3}"
BUFFER_SIZE="${BUFFER_SIZE:-2560000}"
BUFFER_SIZE_FINAL="${BUFFER_SIZE_FINAL:-7680000}"
EPOCHS="${EPOCHS:-24}"
BATCH_SIZE="${BATCH_SIZE:-5120}"
SAMPLES_PER_IMAGE="${SAMPLES_PER_IMAGE:-384}"
BUFFER_BATCH_SIZE="${BUFFER_BATCH_SIZE:-2}"
S1_BATCH_SIZE="${S1_BATCH_SIZE:-16}"
PARTIAL_KEEP_RATIO="${PARTIAL_KEEP_RATIO:-0.5}"
EVAL_EACH_ITER="${EVAL_EACH_ITER:-True}"
EVAL_AFTER_TRAIN="${EVAL_AFTER_TRAIN:-True}"
USE_HALF="${USE_HALF:-True}"
RUN_ROUNDS="${RUN_ROUNDS:-all}"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="${REPO_PATH}/output/${RUN_PREFIX}_${STAMP}"
mkdir -p "${LOG_ROOT}"

# 脚本退出或收到 INT/TERM 时清理本轮启动的两个训练子进程，避免遗留占用 GPU
CLEANUP_PID0=""
CLEANUP_PID1=""
cleanup_bg() {
  [[ -n "$CLEANUP_PID0" ]] && kill "$CLEANUP_PID0" 2>/dev/null || true
  [[ -n "$CLEANUP_PID1" ]] && kill "$CLEANUP_PID1" 2>/dev/null || true
}
trap 'cleanup_bg; exit 130' INT
trap 'cleanup_bg' EXIT TERM

COMMON_ARGS=(
  --use_lmc True
  --memory_path "${MEMORY_PATH}"
  --dinov2_path "${DINO_PATH}"
  --lmc_mode "${LMC_MODE}"
  --num_latent_tokens 64
  --lmc_iterations "${LMC_ITERS}"
  --lmc_warmup_steps "${S1_WARMUP_STEPS}"
  --lmc_train_steps "${S1_TRAIN_STEPS}"
  --s1_learning_rate_max "${S1_LR_MAX}"
  --s2_learning_rate_max "${S2_LR_MAX}"
  --training_buffer_size "${BUFFER_SIZE}"
  --buffer_size_final "${BUFFER_SIZE_FINAL}"
  --epochs "${EPOCHS}"
  --batch_size "${BATCH_SIZE}"
  --samples_per_image "${SAMPLES_PER_IMAGE}"
  --buffer_batch_size "${BUFFER_BATCH_SIZE}"
  --buffer_on_cpu True
  --image_resolution 518
  --s1_batch_size "${S1_BATCH_SIZE}"
  --eval_each_iteration "${EVAL_EACH_ITER}"
  --eval_after_train "${EVAL_AFTER_TRAIN}"
  --keep_best_only True
  --best_metric pct5
  --use_half "${USE_HALF}"
  --lmc_memory_preflight True
  --lmc_memory_preflight_strict True
)

print_header() {
  echo "============================================================"
  echo "2-GPU refill ablation suite"
  echo "GPU0=${GPU0} | GPU1=${GPU1}"
  echo "Scene=${SCENE_PATH}"
  echo "Memory=${MEMORY_PATH}"
  echo "Mode=${LMC_MODE} | Iters=${LMC_ITERS}"
  echo "S1 warmup=${S1_WARMUP_STEPS} | S1 later=${S1_TRAIN_STEPS}"
  echo "S1 LR max=${S1_LR_MAX} | S2 LR max=${S2_LR_MAX}"
  echo "Buffer base=${BUFFER_SIZE} | final=${BUFFER_SIZE_FINAL}"
  echo "Epochs=${EPOCHS} | Batch=${BATCH_SIZE} | SamplesPerImage=${SAMPLES_PER_IMAGE}"
  echo "S1 batch=${S1_BATCH_SIZE} | partial keep_ratio=${PARTIAL_KEEP_RATIO}"
  echo "Run rounds=${RUN_ROUNDS}"
  echo "Logs=${LOG_ROOT}"
  echo "============================================================"
}

should_run_round() {
  local round="$1"
  if [[ "${RUN_ROUNDS}" == "all" ]]; then
    return 0
  fi
  [[ " ${RUN_ROUNDS} " == *" ${round} "* ]]
}

launch_train() {
  local device="$1"; shift
  local run_name="$1"; shift
  local log_file="$1"; shift
  python train_ace_dinov2_lmc.py \
    "${SCENE_PATH}" \
    "${run_name}.pt" \
    --device "${device}" \
    --run_name "${run_name}" \
    "${COMMON_ARGS[@]}" \
    "$@" \
    > "${log_file}" 2>&1 &
  # PID 由调用方通过 $! 获取，避免在子 shell 中启动导致 wait 报 "not a child of this shell"
}

check_log() {
  local log_file="$1"
  echo "------------------------------"
  echo "[CHECK] ${log_file}"
  [[ -f "${log_file}" ]] || { echo "[MISS] log not found"; return; }
  # 使用 grep -E（兼容无 rg 环境），-n 显示行号
  grep -n -E "Memory check|\[LMC preflight\]|S1 partial refill|S1 data source:|\[S1-Buffer\]|\[S2-G\]|Cross-iter eval|Done ACE-G|Done\." "${log_file}" || true
  grep -n -E "Traceback|RuntimeError|ValueError|CUDA out of memory|mat1 and mat2 must have the same dtype|not implemented" "${log_file}" || true
}

run_round_pair() {
  local round_tag="$1"; shift
  local base_name="$1"; shift
  local variant_name="$1"; shift
  local round_dir="${LOG_ROOT}/${round_tag}"
  mkdir -p "${round_dir}"

  echo
  echo "============================================================"
  echo "[ROUND] ${round_tag}"
  echo "[GPU0] ${base_name}"
  echo "[GPU1] ${variant_name}"
  echo "============================================================"

  local args0=()
  while [[ "$1" != "--SPLIT--" ]]; do
    args0+=("$1")
    shift
  done
  shift
  local args1=("$@")

  local pid0
  local pid1
  launch_train "${GPU0}" "${base_name}" "${round_dir}/gpu0_${base_name}.log" "${args0[@]}"
  pid0=$!
  launch_train "${GPU1}" "${variant_name}" "${round_dir}/gpu1_${variant_name}.log" "${args1[@]}"
  pid1=$!
  CLEANUP_PID0=$pid0
  CLEANUP_PID1=$pid1

  echo "[INFO] round=${round_tag} pids: ${pid0} ${pid1}"
  local round_failed=0
  wait "${pid0}" || round_failed=1
  wait "${pid1}" || round_failed=1
  echo "[INFO] round=${round_tag} finished (failed=${round_failed})"

  check_log "${round_dir}/gpu0_${base_name}.log"
  check_log "${round_dir}/gpu1_${variant_name}.log"
  if [[ "${round_failed}" -ne 0 ]]; then
    echo "[ERR] round=${round_tag} failed; stop remaining rounds."
    exit 1
  fi
  # 本轮正常结束，不再在 EXIT 时杀这两个进程（已结束）
  CLEANUP_PID0=""
  CLEANUP_PID1=""
}

print_header

if should_run_round round1; then
  run_round_pair \
    "round1_iterative_refill" \
    "${RUN_PREFIX}_r1_iter_full" \
    "${RUN_PREFIX}_r1_iter_partial" \
    --lmc_flow iterative \
    --s1_use_buffer True \
    --s1_loss_mode sample_per_image \
    --s1_buffer_refill_mode full \
    --SPLIT-- \
    --lmc_flow iterative \
    --s1_use_buffer True \
    --s1_loss_mode sample_per_image \
    --s1_buffer_refill_mode partial \
    --s1_buffer_keep_ratio "${PARTIAL_KEEP_RATIO}"
fi

if should_run_round round2; then
  run_round_pair \
    "round2_aceg_r1_refill" \
    "${RUN_PREFIX}_r2_aceg_r1_full" \
    "${RUN_PREFIX}_r2_aceg_r1_partial" \
    --lmc_flow ace_g \
    --ace_g_fusion_in_s2 False \
    --ace_g_cross_iter_eval False \
    --s1_use_buffer True \
    --s1_loss_mode sample_per_image \
    --s1_buffer_refill_mode full \
    --SPLIT-- \
    --lmc_flow ace_g \
    --ace_g_fusion_in_s2 False \
    --ace_g_cross_iter_eval False \
    --s1_use_buffer True \
    --s1_loss_mode sample_per_image \
    --s1_buffer_refill_mode partial \
    --s1_buffer_keep_ratio "${PARTIAL_KEEP_RATIO}"
fi

if should_run_round round3; then
  run_round_pair \
    "round3_aceg_r2_refill" \
    "${RUN_PREFIX}_r3_aceg_r2_full" \
    "${RUN_PREFIX}_r3_aceg_r2_partial" \
    --lmc_flow ace_g \
    --ace_g_fusion_in_s2 True \
    --ace_g_fusion_lr_ratio 0.01 \
    --ace_g_cross_iter_eval True \
    --s1_use_buffer True \
    --s1_loss_mode sample_per_image \
    --s1_buffer_refill_mode full \
    --batch_size 512 \
    --SPLIT-- \
    --lmc_flow ace_g \
    --ace_g_fusion_in_s2 True \
    --ace_g_fusion_lr_ratio 0.01 \
    --ace_g_cross_iter_eval True \
    --s1_use_buffer True \
    --s1_loss_mode sample_per_image \
    --s1_buffer_refill_mode partial \
    --s1_buffer_keep_ratio "${PARTIAL_KEEP_RATIO}" \
    --batch_size 512
fi

echo
echo "============================================================"
echo "[DONE] 2-GPU refill ablation suite finished"
echo "Logs root: ${LOG_ROOT}"
echo "建议比较："
echo "  1) 各 round 的 best_*_eval_log.txt"
echo "  2) training_full_log.txt 中 S1-Buffer rebuild/partial refill 耗时"
echo "  3) round3 中 Cross-iter eval drop 是否因 partial 变差"
echo "============================================================"
