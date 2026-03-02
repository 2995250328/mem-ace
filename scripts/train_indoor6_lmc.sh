#!/usr/bin/env bash
#
# indoor6 数据集批量训练脚本（ACE DINOv2 LMC）
#
# 前置步骤：
#   1. 运行 adapt_indoor6_to_ace.py 将 indoor6 转为 ACE 格式
#      python scripts/adapt_indoor6_to_ace.py /data/xwh/indoor6 /data/xwh/indoor6_ace
#   2. 校验各场景：python check_dataset_dinov2.py /data/xwh/indoor6_ace/<scene>
#
# ─── Vanilla 训练（不启用 LMC）─────────────────────────────────────────────────
#   ./scripts/train_indoor6_lmc.sh
#
# ─── LMC 训练（每个场景必须有独立 memory 文件）─────────────────────────────────
#   方式 A：逐场景直接指定（推荐，最安全）
#     SCENE_MEMORIES='scene1:/path/to/scene1.pt,scene2a:/path/to/scene2a.pt,...' \
#       USE_LMC=True ./scripts/train_indoor6_lmc.sh
#
#   方式 B：指定 memory 根目录（自动按 <MEMORY_ROOT>/<scene>_pooled.pt 规则查找）
#     MEMORY_ROOT=/path/to/memories USE_LMC=True ./scripts/train_indoor6_lmc.sh
#     可选后缀（默认 _pooled.pt）：MEMORY_SUFFIX=_train_pooled_GT.pt
#
#   【重要】LMC 模式下若某场景无法找到对应 memory 文件，脚本会报错并跳过该场景，
#          而不会静默地把其他场景的 memory 用于当前场景。
#
set -euo pipefail

SCRIPT_PATH=$(dirname "$(realpath -s "$0")")
REPO_PATH=$(realpath -s "${SCRIPT_PATH}/..")

scenes=("scene1" "scene2a" "scene3" "scene4a" "scene5" "scene6")

training_exe="${REPO_PATH}/train_ace_dinov2_lmc.py"

# 转换后的 ACE 格式数据根目录
ace_root="${ACE_ROOT:-/data/xwh/indoor6_ace}"

# 是否启用 LMC
use_lmc="${USE_LMC:-False}"

# ─── 解析 SCENE_MEMORIES 映射（格式：scene1:/path.pt,scene2a:/path.pt,...）──────
declare -A scene_memory_map
if [[ -n "${SCENE_MEMORIES:-}" ]]; then
    IFS=',' read -ra _entries <<< "$SCENE_MEMORIES"
    for _entry in "${_entries[@]}"; do
        _scene="${_entry%%:*}"
        _path="${_entry#*:}"
        scene_memory_map["${_scene}"]="${_path}"
    done
fi

# ─── memory 根目录查找配置 ──────────────────────────────────────────────────────
memory_root="${MEMORY_ROOT:-}"
memory_suffix="${MEMORY_SUFFIX:-_pooled.pt}"

# ─── 工具函数：根据场景名解析 memory 路径 ────────────────────────────────────────
resolve_memory_path() {
    local scene="$1"
    # 方式 A：直接映射
    if [[ -n "${scene_memory_map[$scene]+_}" ]]; then
        echo "${scene_memory_map[$scene]}"
        return 0
    fi
    # 方式 B：根目录规则
    if [[ -n "$memory_root" ]]; then
        local candidate="${memory_root}/${scene}${memory_suffix}"
        echo "$candidate"
        return 0
    fi
    # 未配置
    echo ""
    return 1
}

# ─── 主循环 ──────────────────────────────────────────────────────────────────────
for scene in "${scenes[@]}"; do
    scene_dir="${ace_root}/${scene}"
    if [[ ! -d "$scene_dir" ]]; then
        echo "[SKIP] 目录不存在: $scene_dir"
        continue
    fi

    echo ""
    echo "════════════════════════════════════════"
    echo " 场景: $scene"
    echo "════════════════════════════════════════"

    if [[ "$use_lmc" == "True" ]]; then
        # ── LMC 模式：必须找到场景级 memory 文件 ─────────────────────────────
        if ! mem_path=$(resolve_memory_path "$scene"); then
            echo "[ERROR] LMC 模式下未配置场景 '${scene}' 的 memory 路径。"
            echo "        请通过 SCENE_MEMORIES 或 MEMORY_ROOT 指定（见脚本顶部说明）。"
            echo "[SKIP]  跳过场景: $scene"
            continue
        fi

        if [[ -z "$mem_path" ]]; then
            echo "[ERROR] LMC 模式下未找到场景 '${scene}' 的 memory 路径。"
            echo "        SCENE_MEMORIES 与 MEMORY_ROOT 均未设置。"
            echo "[SKIP]  跳过场景: $scene"
            continue
        fi

        if [[ ! -f "$mem_path" ]]; then
            echo "[ERROR] memory 文件不存在: $mem_path"
            echo "        (场景: ${scene}，请先运行 memory 提取步骤)"
            echo "[SKIP]  跳过场景: $scene"
            continue
        fi

        echo "[INFO]  memory 路径: $mem_path"
        echo ">>> LMC 训练: $scene"
        python "$training_exe" \
            "$scene_dir" "output/${scene}_lmc.pt" \
            --use_lmc True \
            --memory_path "$mem_path" \
            "${EXTRA_ARGS[@]:---device cuda:0}"
    else
        # ── Vanilla 模式 ────────────────────────────────────────────────────────
        echo ">>> Vanilla 训练: $scene"
        python "$training_exe" \
            "$scene_dir" "output/${scene}_vanilla.pt" \
            --use_lmc False \
            "${EXTRA_ARGS[@]:---device cuda:0}"
    fi
done

echo ""
echo "[INFO] 所有场景处理完毕。"
