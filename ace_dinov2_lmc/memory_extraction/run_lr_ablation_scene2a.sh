#!/bin/bash
# ==============================================================================
# LR Ablation + Efficiency Experiment Matrix — scene2a c0_p4
# Generated: 2026-05-12
#
# 基于第一轮 Indoor6 全量基线分析的结果，验证以下优化假设：
#   - LR Group (A1-A3): 降低后期振荡，提升收敛稳定性
#   - Efficiency Group (B1-B3): 减少训练时间，验证精度不降
#   - Combined (C1): 最优LR + 最优效率的组合
#
# 所有实验使用相同的 c0_p4 memory 和 scene2a 数据。
# 修改 --device 和 GPU_ID 以适配可用 GPU。
# ==============================================================================

set -euo pipefail

# ---- Common Parameters ----
readonly SCRIPT="/home/xwh/project/ace_depth/ace_dinov2_lmc/train_ace_dinov2_lmc.py"
readonly SCENE="/home/xwh/data/indoor6_ace/scene2a"
readonly MEM_PATH="/home/xwh/project/ace_depth/ace_dinov2_lmc/memory_extraction/04_evaluation/memory_extract/scene2a/40v_v0.05_bilinear_bse_ut0.02_noaug_asb_adaptive_pool1.0_rgate4_prepair8_sor_gm_l2/20260508_103423/memory_bse.pt"
readonly DEPTH_ROOT="/home/xwh/data/mapanything-dataset/wai_data/indoor6/scene2a_train"
readonly EXP_ROOT="indoor6_lr_ablation"

# Common flags (shared by all experiments)
COMMON_FLAGS=(
    --train_preset memory_compare_ace_g_v1
    --data_backend ace
    --use_lmc True
    --memory_path "$MEM_PATH"
    --lmc_mode global
    --lmc_fps_start_policy farthest_from_center
    --training_buffer_size 2560000
    --buffer_size_final 7680000
    --buffer_on_cpu false
    --buffer_on_cpu_final true
    --buffer_sample_valid_coords True
    --buffer_valid_coord_sample_ratio 1.0
    --buffer_valid_coord_neighbor_radius 1
    --buffer_valid_coord_neighbor_mode cross
    --c1_aux_ref_loss_weight 0.0
    --c1_aux_depth_root "$DEPTH_ROOT"
    --c1_aux_depth_kind gt_depth
    --batch_size 10240
    --post_train_eval_seeds 1305 2026 4242 7777 9001
    --post_train_hypotheses 256
)

# ---- GPU Assignment (modify as needed) ----
GPU_B0=${1:-cuda:0}
GPU_A1=${2:-cuda:0}
GPU_A2=${3:-cuda:0}
GPU_A3=${4:-cuda:0}
GPU_B1=${5:-cuda:0}
GPU_B2=${6:-cuda:0}
GPU_B3=${7:-cuda:0}
GPU_C1=${8:-cuda:0}

# ---- Helper ----
run_exp() {
    local exp_id=$1
    local gpu=$2
    shift 2
    local extra_args=("$@")

    local output="scene2a_${exp_id}.pt"
    local subdir="${EXP_ROOT}/${exp_id}"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting ${exp_id} on ${gpu}..."
    echo "  Output: ${output}"
    echo "  Subdir: ${subdir}"

    python "$SCRIPT" "$SCENE" "$output" \
        --device "$gpu" \
        --post_train_eval_device "$gpu" \
        --experiment_subdir "$subdir" \
        "${COMMON_FLAGS[@]}" \
        "${extra_args[@]}"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished ${exp_id}"
}

# ==============================================================================
# B0: Baseline (legacy profile defaults)
# 预估耗时: ~9.7h
# ==============================================================================
run_B0() {
    run_exp "B0_baseline" "$GPU_B0" \
        --lmc_iterations 28 \
        --epochs 24 \
        --lmc_warmup_steps 2000
}

# ==============================================================================
# A1: LR — 降低 boost (first=1.0, later=0.7)
# 假设: 减少后续迭代的LR峰值，降低振荡幅度
# 预估耗时: ~9.7h
# ==============================================================================
run_A1() {
    run_exp "A1_lr_boost_down" "$GPU_A1" \
        --lmc_iterations 28 \
        --epochs 24 \
        --lmc_warmup_steps 2000 \
        --s2_lr_boost_first 1.0 \
        --s2_lr_boost_later 0.7
}

# ==============================================================================
# A2: LR — 延长 S2 warmup (1800步, ~30%)
# 假设: 更平缓的warmup减少对已收敛参数的扰动
# 预估耗时: ~9.7h
# ==============================================================================
run_A2() {
    run_exp "A2_lr_warmup1800" "$GPU_A2" \
        --lmc_iterations 28 \
        --epochs 24 \
        --lmc_warmup_steps 2000 \
        --s2_lr_warmup_steps 1800
}

# ==============================================================================
# A3: LR — 降低 boost + 延长 warmup 组合
# 假设: 两种LR优化叠加效果
# 预估耗时: ~9.7h
# ==============================================================================
run_A3() {
    run_exp "A3_lr_combined" "$GPU_A3" \
        --lmc_iterations 28 \
        --epochs 24 \
        --lmc_warmup_steps 2000 \
        --s2_lr_boost_first 1.0 \
        --s2_lr_boost_later 0.7 \
        --s2_lr_warmup_steps 1800
}

# ==============================================================================
# B1: 效率 — 减少迭代数 28→16
# 假设: 前14轮贡献85.8%改善，16轮应足以收敛
# 预估耗时: ~6.0h (节省38%)
# ==============================================================================
run_B1() {
    run_exp "B1_iter16" "$GPU_B1" \
        --lmc_iterations 16 \
        --epochs 24 \
        --lmc_warmup_steps 2000
}

# ==============================================================================
# B2: 效率 — 减少 epochs 24→16
# 假设: 每轮S2收敛较快，16 epoch足够
# 预估耗时: ~7.0h (节省28%)
# ==============================================================================
run_B2() {
    run_exp "B2_epoch16" "$GPU_B2" \
        --lmc_iterations 28 \
        --epochs 16 \
        --lmc_warmup_steps 2000
}

# ==============================================================================
# B3: 效率 — 双减 (iter=16, epoch=16)
# 假设: 同时减少迭代和epoch，仍能达标
# 预估耗时: ~4.5h (节省54%)
# ==============================================================================
run_B3() {
    run_exp "B3_iter16_ep16" "$GPU_B3" \
        --lmc_iterations 16 \
        --epochs 16 \
        --lmc_warmup_steps 2000
}

# ==============================================================================
# C1: 组合最优 — 最佳LR + 最佳效率
# 假设: A3的LR配置 + B3的效率配置 + S1 warmup缩减
# 预估耗时: ~4.5h (节省54%)
# ==============================================================================
run_C1() {
    run_exp "C1_combined_opt" "$GPU_C1" \
        --lmc_iterations 16 \
        --epochs 16 \
        --lmc_warmup_steps 1200 \
        --s2_lr_boost_first 1.0 \
        --s2_lr_boost_later 0.7 \
        --s2_lr_warmup_steps 1800
}

# ==============================================================================
# 主入口
# ==============================================================================
usage() {
    cat <<EOF
Usage: $0 <experiment_id> [gpu_id]

Experiments:
  all       Run all experiments sequentially
  B0        Baseline (legacy defaults, 28 iter × 24 epoch)
  A1        LR: boost down (first=1.0, later=0.7)
  A2        LR: warmup extended (1800 steps)
  A3        LR: combined (boost down + warmup extended)
  B1        Efficiency: 16 iterations
  B2        Efficiency: 16 epochs
  B3        Efficiency: 16 iter + 16 epoch
  C1        Combined: best LR + best efficiency
  group_A   Run A1 + A2 + A3 in parallel (needs 3 GPUs)
  group_B   Run B1 + B2 + B3 in parallel (needs 3 GPUs)
  phase1    Run B0 + group_A in parallel (needs 4 GPUs)
  phase2    Run group_B in parallel (needs 3 GPUs)
  phase3    Run C1 (needs 1 GPU)

Parallel example (4 GPUs):
  # Phase 1: LR ablation (run on separate terminals or use nohup)
  $0 B0 cuda:0 &
  $0 A1 cuda:1 &
  $0 A2 cuda:2 &
  $0 A3 cuda:3 &
  wait

  # Phase 2: Efficiency tests
  $0 B1 cuda:0 &
  $0 B2 cuda:1 &
  $0 B3 cuda:2 &
  wait

  # Phase 3: Combined best
  $0 C1 cuda:0
EOF
}

case "${1:-help}" in
    B0) run_B0 ;;
    A1) run_A1 ;;
    A2) run_A2 ;;
    A3) run_A3 ;;
    B1) run_B1 ;;
    B2) run_B2 ;;
    B3) run_B3 ;;
    C1) run_C1 ;;
    all)
        echo "Running all 8 experiments sequentially..."
        run_B0
        run_A1
        run_A2
        run_A3
        run_B1
        run_B2
        run_B3
        run_C1
        ;;
    group_A)
        run_A1 &
        run_A2 &
        run_A3 &
        wait
        ;;
    group_B)
        run_B1 &
        run_B2 &
        run_B3 &
        wait
        ;;
    phase1)
        run_B0 &
        run_A1 &
        run_A2 &
        run_A3 &
        wait
        ;;
    phase2)
        run_B1 &
        run_B2 &
        run_B3 &
        wait
        ;;
    phase3)
        run_C1
        ;;
    *)
        usage
        ;;
esac
