#!/bin/bash
# ==============================================================================
# Focused Ablation — scene2a c0_p4, 2 GPUs (cuda:0, cuda:1)
# Updated: 2026-05-12
#
# 目标: 找到训练时间更短、精度不降的最优配置
# 6个实验, 每对相邻仅差一个维度, 3轮, ~23h wall time
#
# 归因链:
#   B0 → X1: LR优化 + S1缩短
#   X1 → X2: 减少迭代 (28→16)
#   X2 → X3: 减少epoch  (24→16)
#   X3 → X4: 减小buffer (2.56M→2.0M)
#   X3 → X5: 极限迭代   (16→12)
#
# 调度:
#   Round 1 (~12h):  GPU0=B0,  GPU1=X1
#   Round 2 (~7h):   GPU0=X2,  GPU1=X3
#   Round 3 (~5h):   GPU0=X4,  GPU1=X5
#
# 用法:
#   bash run_lr_ablation_scene2a.sh              # 全部3轮自动跑
#   bash run_lr_ablation_scene2a.sh round1       # 只跑第1轮
#   bash run_lr_ablation_scene2a.sh round2       # 只跑第2轮
#   bash run_lr_ablation_scene2a.sh round3       # 只跑第3轮
#   bash run_lr_ablation_scene2a.sh B0           # 只跑单个实验
#   bash run_lr_ablation_scene2a.sh quick        # 快速验证: B0+X3
#   bash run_lr_ablation_scene2a.sh status       # 查看已有结果
# ==============================================================================

set -euo pipefail

# ========================== 固定配置 ==========================
readonly SCRIPT="/home/xwh/project/ace_depth/ace_dinov2_lmc/train_ace_dinov2_lmc.py"
readonly SCENE="/home/xwh/data/indoor6_ace/scene2a"
readonly MEM_PATH="/home/xwh/project/ace_depth/ace_dinov2_lmc/memory_extraction/04_evaluation/memory_extract/scene2a/40v_v0.05_bilinear_bse_ut0.02_noaug_asb_adaptive_pool1.0_rgate4_prepair8_sor_gm_l2/20260508_103423/memory_bse.pt"
readonly DEPTH_ROOT="/home/xwh/data/mapanything-dataset/wai_data/indoor6/scene2a_train"
readonly EXP_ROOT="indoor6_focused_ablation"
readonly GPU0="cuda:0"
readonly GPU1="cuda:1"
readonly LOG_DIR="/tmp/ablation_$(date +%Y%m%d_%H%M%S)"

# ========================== 前置检查 ==========================
preflight() {
    echo "======================================"
    echo "  Pre-flight Check"
    echo "======================================"

    # 检查训练脚本
    if [ ! -f "$SCRIPT" ]; then
        echo "[FAIL] 训练脚本不存在: $SCRIPT"; exit 1
    fi
    echo "[OK]   训练脚本: $SCRIPT"

    # 检查 memory 文件
    if [ ! -f "$MEM_PATH" ]; then
        echo "[FAIL] Memory 文件不存在: $MEM_PATH"; exit 1
    fi
    echo "[OK]   Memory: $(du -h "$MEM_PATH" | cut -f1)"

    # 检查场景数据
    if [ ! -d "$SCENE/train" ]; then
        echo "[FAIL] 场景数据不存在: $SCENE/train"; exit 1
    fi
    echo "[OK]   场景: $SCENE"

    # 检查 GPU
    local gpu_ok=true
    for g in 0 1; do
        local free=$(nvidia-smi -i $g --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null || echo "0")
        free=$(echo "$free" | tr -d ' ')
        if [ "$free" -lt 8000 ] 2>/dev/null; then
            echo "[WARN] GPU $g 空闲显存仅 ${free}MB, 可能不够 (需 ~16GB)"
            gpu_ok=false
        else
            echo "[OK]   GPU $g: ${free}MB 空闲"
        fi
    done

    if [ "$gpu_ok" = false ]; then
        echo ""
        echo "GPU 显存不足，是否继续? (y/N)"
        read -r ans
        [ "$ans" = "y" ] || exit 1
    fi

    # 创建日志目录
    mkdir -p "$LOG_DIR"
    echo "[OK]   日志目录: $LOG_DIR"
    echo ""
}

# ========================== 公共参数 ==========================
COMMON_FLAGS=(
    --train_preset memory_compare_ace_g_v1
    --data_backend ace
    --use_lmc True
    --memory_path "$MEM_PATH"
    --lmc_mode global
    --lmc_fps_start_policy farthest_from_center
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

# ========================== 运行单个实验 ==========================
run_exp() {
    local exp_id=$1
    local gpu=$2
    shift 2

    local output="scene2a_${exp_id}.pt"
    local subdir="${EXP_ROOT}/${exp_id}"
    local log_file="${LOG_DIR}/${exp_id}.log"

    echo "============================================================"
    echo "[$(date '+%H:%M:%S')] START ${exp_id} on ${gpu} | log: ${log_file}"
    echo "============================================================"

    local rc=0
    python "$SCRIPT" "$SCENE" "$output" \
        --device "$gpu" \
        --post_train_eval_device "$gpu" \
        --experiment_subdir "$subdir" \
        "${COMMON_FLAGS[@]}" \
        "$@" \
        > "$log_file" 2>&1 || rc=$?

    if [ $rc -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] DONE  ${exp_id} ✓" | tee -a "$LOG_DIR/_summary.log"
    else
        echo "[$(date '+%H:%M:%S')] FAIL  ${exp_id} ✗ (exit=$rc)" | tee -a "$LOG_DIR/_summary.log"
    fi
    echo ""
    return $rc
}

# ========================== 6个实验定义 ==========================

# B0: 基线 (legacy, 28/24/buf2.56M/s1=2000/600) ~12h
run_B0()  { run_exp "B0_baseline"     "$GPU0" --lmc_iterations 28 --epochs 24 --lmc_warmup_steps 2000 --lmc_train_steps 600 --training_buffer_size 2560000 --buffer_size_final 7680000; }

# X1: LR+S1 (28/24, boost=1.0/0.7, warmup=1800, s1=1200/400) ~11h
run_X1()  { run_exp "X1_lr_s1"        "$GPU1" --lmc_iterations 28 --epochs 24 --lmc_warmup_steps 1200 --lmc_train_steps 400 --training_buffer_size 2560000 --buffer_size_final 7680000 --s2_lr_boost_first 1.0 --s2_lr_boost_later 0.7 --s2_lr_warmup_steps 1800; }

# X2: 减迭代 (16/24 + LR+S1) ~7h
run_X2()  { run_exp "X2_iter16"       "$GPU0" --lmc_iterations 16 --epochs 24 --lmc_warmup_steps 1200 --lmc_train_steps 400 --training_buffer_size 2560000 --buffer_size_final 7680000 --s2_lr_boost_first 1.0 --s2_lr_boost_later 0.7 --s2_lr_warmup_steps 1800; }

# X3: 减迭代+epoch (16/16 + LR+S1) ~5h — 主力候选
run_X3()  { run_exp "X3_iter16_ep16"  "$GPU1" --lmc_iterations 16 --epochs 16 --lmc_warmup_steps 1200 --lmc_train_steps 400 --training_buffer_size 2560000 --buffer_size_final 7680000 --s2_lr_boost_first 1.0 --s2_lr_boost_later 0.7 --s2_lr_warmup_steps 1800; }

# X4: 减buffer (16/16/buf2.0M + LR+S1) ~5h
run_X4()  { run_exp "X4_buf2M"        "$GPU0" --lmc_iterations 16 --epochs 16 --lmc_warmup_steps 1200 --lmc_train_steps 400 --training_buffer_size 2000000 --buffer_size_final 6000000 --s2_lr_boost_first 1.0 --s2_lr_boost_later 0.7 --s2_lr_warmup_steps 1800; }

# X5: 极限12轮 (12/16 + LR+S1) ~4h
run_X5()  { run_exp "X5_iter12"       "$GPU1" --lmc_iterations 12 --epochs 16 --lmc_warmup_steps 1200 --lmc_train_steps 400 --training_buffer_size 2560000 --buffer_size_final 7680000 --s2_lr_boost_first 1.0 --s2_lr_boost_later 0.7 --s2_lr_warmup_steps 1800; }

# ========================== 轮次调度 ==========================

round1() {
    echo ""
    echo "╔══════════════════════════════════════════════════╗"
    echo "║  Round 1: B0 (GPU0) + X1 (GPU1)   预计 ~12h     ║"
    echo "╚══════════════════════════════════════════════════╝"
    local t0=$(date +%s)
    run_B0 &
    run_X1 &
    wait
    local dt=$(( ($(date +%s) - t0) / 60 ))
    echo "Round 1 完成, 耗时 ${dt} 分钟"
}

round2() {
    echo ""
    echo "╔══════════════════════════════════════════════════╗"
    echo "║  Round 2: X2 (GPU0) + X3 (GPU1)   预计 ~7h      ║"
    echo "╚══════════════════════════════════════════════════╝"
    local t0=$(date +%s)
    run_X2 &
    run_X3 &
    wait
    local dt=$(( ($(date +%s) - t0) / 60 ))
    echo "Round 2 完成, 耗时 ${dt} 分钟"
}

round3() {
    echo ""
    echo "╔══════════════════════════════════════════════════╗"
    echo "║  Round 3: X4 (GPU0) + X5 (GPU1)   预计 ~5h      ║"
    echo "╚══════════════════════════════════════════════════╝"
    local t0=$(date +%s)
    run_X4 &
    run_X5 &
    wait
    local dt=$(( ($(date +%s) - t0) / 60 ))
    echo "Round 3 完成, 耗时 ${dt} 分钟"
}

# ========================== 结果汇总 ==========================

collect_results() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║                    实验结果汇总                              ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""

    # 搜索所有实验输出目录
    local eval_root="/home/xwh/project/ace_depth/ace_dinov2_lmc/04_evaluation/train_compare"

    printf "%-20s  %6s  %8s  %8s  %6s  %6s  %s\n" \
        "Experiment" "pct5" "med_rot" "med_trans" "iter" "time" "Status"
    echo "─────────────────────────────────────────────────────────────────────"

    for exp_id in B0_baseline X1_lr_s1 X2_iter16 X3_iter16_ep16 X4_buf2M X5_iter12; do
        local subdir="${EXP_ROOT}/${exp_id}"
        local found=false

        # 在 eval_root 下搜索对应的实验子目录
        while IFS= read -r -d '' exp_dir; do
            local meta="$exp_dir/best_checkpoint_meta.json"
            local eval_log=""
            # 找最后一条 eval 记录
            eval_log=$(find "$exp_dir" -name "*_eval_log.txt" -print -quit 2>/dev/null)

            if [ -f "$meta" ]; then
                local best_iter=$(python3 -c "import json; d=json.load(open('$meta')); print(d.get('best_iter','?'))" 2>/dev/null)
                local best_score=$(python3 -c "import json; d=json.load(open('$meta')); print(f\"{d.get('best_score',0):.2f}\")" 2>/dev/null)
                local med_rot=$(python3 -c "import json; d=json.load(open('$meta')); print(f\"{d.get('median_rotation_deg',0):.4f}\")" 2>/dev/null)
                local med_trans=$(python3 -c "import json; d=json.load(open('$meta')); print(f\"{d.get('median_translation_cm',0):.2f}\")" 2>/dev/null)
                printf "%-20s  %6s  %8s  %8s  %6s  %6s  %s\n" \
                    "$exp_id" "$best_score" "$med_rot" "$med_trans" "$best_iter" "" "BEST"
                found=true
                break
            elif [ -n "$eval_log" ] && [ -f "$eval_log" ]; then
                # 从 eval_log 取最后一行
                local last_line=$(tail -1 "$eval_log" | grep "^iter=")
                if [ -n "$last_line" ]; then
                    local score=$(echo "$last_line" | grep -oP 'acc5=\K[0-9.]+')
                    local med_r=$(echo "$last_line" | grep -oP 'median_rotation_deg=\K[0-9.]+')
                    local med_t=$(echo "$last_line" | grep -oP 'median_translation_cm=\K[0-9.]+')
                    local iter=$(echo "$last_line" | grep -oP 'iter=\K[0-9]+')
                    printf "%-20s  %6s  %8s  %8s  %6s  %6s  %s\n" \
                        "$exp_id" "${score:-?}" "${med_r:-?}" "${med_t:-?}" "${iter:-?}" "" "EVAL"
                    found=true
                    break
                fi
            fi
        done < <(find "$eval_root/$subdir" -mindepth 1 -maxdepth 3 -type d 2>/dev/null | sort -z | tail -z -1)

        if [ "$found" = false ]; then
            # 也检查日志目录
            local log_file="${LOG_DIR}/${exp_id}.log"
            if [ -f "$log_file" ]; then
                printf "%-20s  %6s  %8s  %8s  %6s  %6s  %s\n" \
                    "$exp_id" "-" "-" "-" "-" "-" "LOG EXISTS"
            else
                printf "%-20s  %6s  %8s  %8s  %6s  %6s  %s\n" \
                    "$exp_id" "-" "-" "-" "-" "-" "NOT RUN"
            fi
        fi
    done

    echo ""
    echo "归因分析 (相邻实验对比):"
    echo "  B0 → X1: LR优化 + S1缩短 效果"
    echo "  X1 → X2: 减少迭代 28→16"
    echo "  X2 → X3: 减少epoch 24→16"
    echo "  X3 → X4: 减小buffer 2.56M→2.0M"
    echo "  X3 → X5: 极限迭代 16→12"
}

# ========================== 主入口 ==========================

usage() {
    cat <<EOF
Usage: $0 <command>

Commands:
  all       全部3轮自动运行 (~23h)
  round1    Round 1: B0 + X1 (~12h)
  round2    Round 2: X2 + X3 (~7h)
  round3    Round 3: X4 + X5 (~5h)
  quick     快速验证: B0 + X3 并行 (~12h)

  B0        单独运行 B0 (基线)
  X1        单独运行 X1 (LR+S1)
  X2        单独运行 X2 (16iter)
  X3        单独运行 X3 (16iter+16ep)
  X4        单独运行 X4 (小buffer)
  X5        单独运行 X5 (12iter)

  status    查看已有实验结果
  check     只做前置检查，不启动训练

Experiment Matrix:
  ┌──────┬───────┬───────┬─────────────┬──────────────┬───────┐
  │  ID  │ iter  │ epoch │   buffer    │   vs Prev    │ Time  │
  ├──────┼───────┼───────┼─────────────┼──────────────┼───────┤
  │  B0  │  28   │  24   │ 2.56/7.68M  │   baseline   │ ~12h  │
  │  X1  │  28   │  24   │ 2.56/7.68M  │  B0→LR+S1    │ ~11h  │
  │  X2  │  16   │  24   │ 2.56/7.68M  │  X1→iter-    │ ~7h   │
  │  X3  │  16   │  16   │ 2.56/7.68M  │  X2→epoch-   │ ~5h   │
  │  X4  │  16   │  16   │ 2.00/6.00M  │  X3→buf-     │ ~5h   │
  │  X5  │  12   │  16   │ 2.56/7.68M  │  X3→iter--   │ ~4h   │
  └──────┴───────┴───────┴─────────────┴──────────────┴───────┘

Schedule (GPU0 + GPU1):
  Round 1: B0(GPU0) + X1(GPU1)  ~12h
  Round 2: X2(GPU0) + X3(GPU1)  ~7h
  Round 3: X4(GPU0) + X5(GPU1)  ~5h
EOF
}

case "${1:-help}" in
    check)
        preflight
        ;;
    status)
        collect_results
        ;;
    B0)  preflight; run_B0  ;;
    X1)  preflight; run_X1  ;;
    X2)  preflight; run_X2  ;;
    X3)  preflight; run_X3  ;;
    X4)  preflight; run_X4  ;;
    X5)  preflight; run_X5  ;;
    round1)
        preflight
        round1
        collect_results
        ;;
    round2)
        round2
        collect_results
        ;;
    round3)
        round3
        collect_results
        ;;
    all)
        preflight
        echo "================================================================"
        echo "  全部 3 轮, 预计 wall time ~23h"
        echo "  开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "================================================================"
        local t_start=$(date +%s)

        round1
        collect_results

        echo ""
        echo "按 Enter 继续 Round 2, 或 Ctrl+C 停止..."
        read -r

        round2
        collect_results

        echo ""
        echo "按 Enter 继续 Round 3, 或 Ctrl+C 停止..."
        read -r

        round3

        local t_end=$(date +%s)
        local total_dt=$(( (t_end - t_start) / 60 ))
        echo ""
        echo "================================================================"
        echo "  全部完成! 总耗时 ${total_dt} 分钟"
        echo "================================================================"
        collect_results
        ;;
    quick)
        preflight
        echo "================================================================"
        echo "  Quick: B0(GPU0) + X3(GPU1) — 主力候选直接对比"
        echo "================================================================"
        run_B0 &
        run_X3 &
        wait
        collect_results
        ;;
    *)
        usage
        ;;
esac
