#!/bin/bash
# =============================================================================
# BSE Memory Extraction — ace_depth 版本
# =============================================================================
# 从数据集中提取 memory 特征，使用 BSE pooling + Welford 归一化。
# 参考 map-anything/bash_scripts/ace/fps_memory.sh 的参数设计。
#
# 用法：
#   修改下方「配置参数」，然后执行本脚本。
#   也可通过环境变量覆盖：SCENE_TRAIN=chess_train N_VIEWS=20 bash extract_memory.sh
#   使用ACE数据集加载器：DATASET_LOADER=ace SCENE_TRAIN=chess bash extract_memory.sh
#
# 输出目录结构：
#   <OUTPUT_ROOT>/<scene_name>/<N_VIEWS>views_<config_tag>/<timestamp>/
#     ├── memory_bse.pt          (主输出)
#     ├── extraction_config.json (配置快照)
#     └── extraction_log.txt     (运行日志)
# =============================================================================

set -e

# ace_depth 根目录（用于默认 checkpoint 软链接：checkpoints/dinov2_vitl14_pretrain.pth）
_EXTRACT_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_ACE_DEPTH_ROOT="$(cd "$_EXTRACT_SCRIPT_DIR/../.." && pwd)"
ACE_DATA_ROOT="${ACE_DATA_ROOT:-/home/xwh/data}"

# =============================================================================
# 配置参数（与 run_memory_extraction.py 的 ExtractionConfig / parse_args 对应）
# =============================================================================
# 说明：下方「环境变量」可通过命令行覆盖，例如：
#   N_VIEWS=50 GPU_ID=0 USE_MODEL=dinov2 bash extract_memory.sh
# 未传入 Python 的参数见各变量注释（如 USE_L2_NORMALIZATION 仅用于目录标签与 JSON 快照）。

# --- 数据集与场景 ---
# DATASET_TYPE — 传给 --dataset_type：7scenes / indoor6 / custom（影响默认深度范围与采样默认值）
# SCENE_TRAIN  — 传给 --scene_name：WAI 完整场景名，如 chess_train（含 _train 后缀）
# SCENE_TEST   — 仅写入 extraction_config.json，供实验记录；Python 提取训练 memory 不直接使用
# DATASET_LOADER — 传给 --dataset_loader：wai=WAI 目录结构；ace=ACE CamLocDatasetDINOv2
# 约定：严格分流，不做自动检测。
# - DATASET_LOADER=ace  => 仅走 ACE loader 逻辑
# - DATASET_LOADER=wai  => 仅走 MapAnything/WAI loader 逻辑
DATASET_TYPE="${DATASET_TYPE:-7scenes}"
SCENE_TRAIN="${SCENE_TRAIN:-chess_train}"
SCENE_TEST="${SCENE_TEST:-}"
DATASET_LOADER="${DATASET_LOADER:-wai}"

# 清理场景名中的不可见字符（如复制粘贴带入的 zero-width space）
SCENE_TRAIN="${SCENE_TRAIN//$'\u200b'/}"
SCENE_TRAIN="${SCENE_TRAIN//$'\u200c'/}"
SCENE_TRAIN="${SCENE_TRAIN//$'\u200d'/}"
SCENE_TRAIN="${SCENE_TRAIN//$'\ufeff'/}"

# --- 数据集根目录（根据 DATASET_TYPE 在未设置 DATASET_ROOT 时自动选择） ---
# DATASET_PATH 将作为 Python 第一个位置参数：WAI 时为数据集 ROOT（其下含各 scene 子目录）
# SevenScenesWAI(ROOT=DATASET_PATH, specific_scene_name=SCENE_TRAIN, ...)
if [ -z "$DATASET_ROOT" ]; then
    case "$DATASET_TYPE" in
        7scenes)  DATASET_ROOT="$ACE_DATA_ROOT/mapanything-dataset/wai_data/7scenes" ;;
        indoor6)  DATASET_ROOT="$ACE_DATA_ROOT/mapanything-dataset/wai_data/indoor6" ;;
        mushroom) DATASET_ROOT="${MUSHROOM_WAI_ROOT:-/data/xwh/MuSHRoom_wai/kinect}" ;;
        rio10)    DATASET_ROOT="${RIO10_WAI_ROOT:-/data/xwh/RIO10_wai/mapanything_wai}" ;;
        *)        DATASET_ROOT="$ACE_DATA_ROOT" ;;
    esac
fi

# DATASET_PATH — 与 Python 的 dataset_path 一致，默认等于 DATASET_ROOT
# 当 DATASET_LOADER=ace 时，DATASET_PATH 需要直接指向含 rgb/ 的目录（如 scene/train/），
# 脚本会在下方自动拼接 SCENE_TRAIN（若路径尚不含 rgb/ 子目录）
DATASET_PATH="$DATASET_ROOT"

# --- ACE 数据集路径修正 ---
# ACE 加载器 (CamLocDatasetDINOv2) 要求 dataset_path 直接指向含 rgb/ 的目录。
# 若用户传入的是 scene 根目录（如 pgt_7scenes_chess/），自动拼接 /${SCENE_TRAIN}
if [ "$DATASET_LOADER" = "ace" ]; then
    if [ ! -d "$DATASET_PATH/rgb" ] && [ -d "$DATASET_PATH/${SCENE_TRAIN}/rgb" ]; then
        DATASET_PATH="$DATASET_PATH/${SCENE_TRAIN}"
        echo "[ACE loader] Auto-appended /${SCENE_TRAIN} → $DATASET_PATH"
    fi
fi

# --- 严格分流校验（避免 wai/ace 路径混用） ---
if [ "$DATASET_LOADER" = "ace" ]; then
    if [ ! -d "$DATASET_PATH/rgb" ]; then
        echo "ERROR: DATASET_LOADER=ace 但未找到 '$DATASET_PATH/rgb'"
        echo "  请将 DATASET_ROOT/DATASET_PATH 指向 ACE 场景目录（包含 rgb/, poses/, depth/）。"
        echo "  示例: DATASET_LOADER=ace DATASET_ROOT=/path/to/pgt_7scenes_chess SCENE_TRAIN=train ..."
        exit 1
    fi
fi

if [ "$DATASET_LOADER" = "wai" ]; then
    if [ -d "$DATASET_PATH/rgb" ] || [ -d "$DATASET_PATH/train/rgb" ]; then
        echo "ERROR: DATASET_LOADER=wai 但检测到 ACE 风格目录（rgb/）。"
        echo "  当前路径更像 ACE 数据集，请改用 DATASET_LOADER=ace。"
        echo "  示例: DATASET_LOADER=ace DATASET_ROOT=$DATASET_ROOT SCENE_TRAIN=$SCENE_TRAIN ..."
        exit 1
    fi
    if [ ! -d "$DATASET_PATH/$SCENE_TRAIN" ]; then
        echo "ERROR: DATASET_LOADER=wai 但场景目录不存在: $DATASET_PATH/$SCENE_TRAIN"
        echo "  请检查 SCENE_TRAIN 是否拼写正确，且不包含不可见字符。"
        echo "  常见值示例: chess_train, stairs_train, fire_train ..."
        exit 1
    fi
fi

# --- 自动推导 SCENE_TEST（仅文档/JSON，不参与 Python CLI）---
if [ -z "$SCENE_TEST" ]; then
    case "$SCENE_TRAIN" in
        *_train) SCENE_TEST="${SCENE_TRAIN%_train}_test" ;;
        *_val)   SCENE_TEST="${SCENE_TRAIN%_val}_test"   ;;
        *_test)  SCENE_TEST="$SCENE_TRAIN"               ;;
        *)       SCENE_TEST="${SCENE_TRAIN}_test"         ;;
    esac
fi

# SCENE_NAME — 输出目录的“场景名”层级
# - WAI：通常直接用 SCENE_TRAIN 去掉 _train（chess_train → chess）
# - ACE：很多数据集用 train/val/test 作为 split 目录名；此时用 SCENE_TRAIN 会导致输出落到 .../train/，
#        不利于按真实场景归档。因此默认改为从 DATASET_ROOT 推导（例如 pgt_7scenes_chess → chess）。
# 你也可以显式设置 OUTPUT_SCENE_NAME 来覆盖该逻辑。
if [ -n "$OUTPUT_SCENE_NAME" ]; then
    SCENE_NAME="$OUTPUT_SCENE_NAME"
else
    if [ "$DATASET_LOADER" = "ace" ] && { [ "$SCENE_TRAIN" = "train" ] || [ "$SCENE_TRAIN" = "val" ] || [ "$SCENE_TRAIN" = "test" ]; }; then
        _base="$(basename "$DATASET_ROOT")"
        # 常见前缀清理：pgt_7scenes_chess → chess
        SCENE_NAME="${_base#pgt_7scenes_}"
        SCENE_NAME="${SCENE_NAME#pgt_indoor6_}"
        SCENE_NAME="${SCENE_NAME#pgt_}"
    else
        SCENE_NAME="${SCENE_TRAIN%_train}"
    fi
fi

# N_VIEWS — 传给 --n_memory：参与 memory 提取的帧数
N_VIEWS="${N_VIEWS:-20}"

# Reference contract. C0 keeps current world-frame training/eval behavior;
# C1 is reserved for reference-coordinate learning experiments.
CONTRACT_MODE="${CONTRACT_MODE:-C0}"

# WAI memory extraction preprocessing.
# Keep this deterministic by default: no random resize/crop/color augmentation,
# only model-required normalization. This avoids processed intrinsics changing
# between runs with the same selected views.
WAI_TRANSFORM="${WAI_TRANSFORM:-imgnorm}"
WAI_DATA_NORM_TYPE="${WAI_DATA_NORM_TYPE:-dinov2}"
WAI_AUG_CROP="${WAI_AUG_CROP:-0}"
WAI_RESOLUTION="${WAI_RESOLUTION:-518}"

# WAI_VIEW_MODE — WAI 视图加载协议：
#   fps_flat / fps_strict=每个 FPS index 只加载 1 张图，实际推理输入严格等于 FPS list
#   original_multiview / first_fps_covis=复现 map-anything fps_memory.sh，首个 FPS anchor 返回 n_views 个 covisibility views
#   anchor_support / asb=reference-aware anchor/support 选帧，anchor 保覆盖、support 保局部共视
WAI_VIEW_MODE="${WAI_VIEW_MODE:-fps_flat}"
case "${WAI_VIEW_MODE,,}" in
    fps_strict|strict_fps)
        WAI_VIEW_MODE="fps_flat"
        ;;
    first_fps_covis|first_fps_covis40|mapanything_original)
        WAI_VIEW_MODE="original_multiview"
        ;;
    asb)
        WAI_VIEW_MODE="anchor_support"
        ;;
esac
ASB_ALPHA="${ASB_ALPHA:-${COVIS_ALPHA:-1.0}}"
ASB_EPS="${ASB_EPS:-${COVIS_EPS:-1e-6}}"
ASB_TAU="${ASB_TAU:-${COVIS_TAU:--1.0}}"
ASB_REF_LAMBDA="${ASB_REF_LAMBDA:-${COVIS_REF_LAMBDA:-0.0}}"
ASB_ANCHOR_COUNT="${ASB_ANCHOR_COUNT:-${COVIS_ANCHOR_COUNT:-0}}"
ASB_SUPPORT_PER_ANCHOR="${ASB_SUPPORT_PER_ANCHOR:-${COVIS_SUPPORT_PER_ANCHOR:-3}}"
ASB_SUPPORT_TAU="${ASB_SUPPORT_TAU:-${COVIS_SUPPORT_TAU:-1e-4}}"
ASB_SUPPORT_MIN_NEIGHBORS="${ASB_SUPPORT_MIN_NEIGHBORS:-${COVIS_SUPPORT_MIN_NEIGHBORS:-3}}"
ASB_SAFE_DIST_TO_REF="${ASB_SAFE_DIST_TO_REF:-${COVIS_SAFE_DIST_TO_REF:-3.0}}"
ASB_FAR_VIEW_BUDGET="${ASB_FAR_VIEW_BUDGET:-${COVIS_FAR_VIEW_BUDGET:-8}}"
ASB_FAR_ANCHOR_BUDGET="${ASB_FAR_ANCHOR_BUDGET:-${COVIS_FAR_ANCHOR_BUDGET:-2}}"
ASB_COVERAGE_BETA="${ASB_COVERAGE_BETA:-${COVIS_COVERAGE_BETA:-1.0}}"
ASB_CANDIDATE_POOL_RATIO="${ASB_CANDIDATE_POOL_RATIO:-${COVIS_CANDIDATE_POOL_RATIO:-1.0}}"
ASB_MA_SAFE="${ASB_MA_SAFE:-${COVIS_MA_SAFE_ASB:-false}}"
ASB_MA_SAFE_POOL_RATIO="${ASB_MA_SAFE_POOL_RATIO:-${COVIS_MA_SAFE_POOL_RATIO:-3.0}}"
ASB_NATIVE_GROUP="${ASB_NATIVE_GROUP:-${COVIS_NATIVE_GROUP_ASB:-false}}"
ASB_NATIVE_ANCHOR_CANDIDATES="${ASB_NATIVE_ANCHOR_CANDIDATES:-${COVIS_NATIVE_ANCHOR_CANDIDATES:-64}}"
ASB_ADAPTIVE="${ASB_ADAPTIVE:-${COVIS_ADAPTIVE_ASB:-false}}"
ASB_SCENE_STATS="${ASB_SCENE_STATS:-${COVIS_ADAPTIVE_SCENE_STATS:-true}}"
if [ -z "${ASB_ADAPTIVE_VIEW_COUNT+x}" ]; then
    if [ -n "${COVIS_ADAPTIVE_VIEW_COUNT+x}" ]; then
        ASB_ADAPTIVE_VIEW_COUNT="$COVIS_ADAPTIVE_VIEW_COUNT"
    else
        case "${ASB_ADAPTIVE,,}" in
            true|1|yes|y|on)
                ASB_ADAPTIVE_VIEW_COUNT="true"
                ;;
            *)
                ASB_ADAPTIVE_VIEW_COUNT="false"
                ;;
        esac
    fi
fi
ASB_MIN_VIEWS="${ASB_MIN_VIEWS:-${COVIS_ADAPTIVE_MIN_VIEWS:-28}}"
ASB_REF_DIST_MAX_LIMIT="${ASB_REF_DIST_MAX_LIMIT:-${COVIS_REF_DIST_MAX_LIMIT:-4.2}}"
ASB_REF_DIST_MEAN_LIMIT="${ASB_REF_DIST_MEAN_LIMIT:-${COVIS_REF_DIST_MEAN_LIMIT:-2.7}}"
ASB_MIN_COVERAGE_GAIN="${ASB_MIN_COVERAGE_GAIN:-${COVIS_MIN_COVERAGE_GAIN:-0.01}}"
ASB_GAIN_PATIENCE="${ASB_GAIN_PATIENCE:-${COVIS_GAIN_PATIENCE:-2}}"
ASB_TARGET_COVERAGE_MEAN="${ASB_TARGET_COVERAGE_MEAN:-${COVIS_TARGET_COVERAGE_MEAN:-0.55}}"
ASB_TARGET_COVERAGE_MAX="${ASB_TARGET_COVERAGE_MAX:-${COVIS_TARGET_COVERAGE_MAX:-2.0}}"
ASB_POST_REPAIR="${ASB_POST_REPAIR:-${COVIS_POST_REPAIR_ASB:-false}}"
ASB_POST_REPAIR_MAX_SWAPS="${ASB_POST_REPAIR_MAX_SWAPS:-${COVIS_POST_REPAIR_MAX_SWAPS:-4}}"
ASB_POST_REPAIR_TAIL_PERCENTILE="${ASB_POST_REPAIR_TAIL_PERCENTILE:-${COVIS_POST_REPAIR_TAIL_PERCENTILE:-95.0}}"
ASB_POST_REPAIR_MIN_TAIL_IMPROVEMENT_M="${ASB_POST_REPAIR_MIN_TAIL_IMPROVEMENT_M:-${COVIS_POST_REPAIR_MIN_TAIL_IMPROVEMENT_M:-0.05}}"
ASB_POST_REPAIR_CLUSTER_TOP_K="${ASB_POST_REPAIR_CLUSTER_TOP_K:-${COVIS_POST_REPAIR_CLUSTER_TOP_K:-3}}"
ASB_POSE_PRUNE="${ASB_POSE_PRUNE:-${COVIS_POSE_PRUNE:-false}}"
ASB_POSE_PRUNE_MIN_VIEWS="${ASB_POSE_PRUNE_MIN_VIEWS:-${COVIS_POSE_PRUNE_MIN_VIEWS:-20}}"
ASB_POSE_PRUNE_KEEP_RATIO="${ASB_POSE_PRUNE_KEEP_RATIO:-${COVIS_POSE_PRUNE_KEEP_RATIO:-0.70}}"
ASB_POSE_PRUNE_MAD_K="${ASB_POSE_PRUNE_MAD_K:-${COVIS_POSE_PRUNE_MAD_K:-2.5}}"
ENABLE_REFERENCE_POLICY_GATE="${ENABLE_REFERENCE_POLICY_GATE:-false}"
REFERENCE_POLICY_TOP_M="${REFERENCE_POLICY_TOP_M:-4}"
REFERENCE_POLICY_LIGHT_MIN_SELECTED_LINKS="${REFERENCE_POLICY_LIGHT_MIN_SELECTED_LINKS:-2}"
REFERENCE_POLICY_PROBE_Q90_M="${REFERENCE_POLICY_PROBE_Q90_M:-0.18}"
REFERENCE_POLICY_PROBE_MAX_M="${REFERENCE_POLICY_PROBE_MAX_M:-0.25}"
ENABLE_CLUSTER_FALLBACK="${ENABLE_CLUSTER_FALLBACK:-false}"
CLUSTER_FALLBACK_MAX_CLUSTERS="${CLUSTER_FALLBACK_MAX_CLUSTERS:-2}"
CLUSTER_FALLBACK_MAX_SPLIT_DEPTH="${CLUSTER_FALLBACK_MAX_SPLIT_DEPTH:-2}"
CLUSTER_FALLBACK_MIN_CLUSTER_SIZE="${CLUSTER_FALLBACK_MIN_CLUSTER_SIZE:-8}"
CLUSTER_FALLBACK_MIN_VIEWS_PER_CLUSTER="${CLUSTER_FALLBACK_MIN_VIEWS_PER_CLUSTER:-8}"
CLUSTER_FALLBACK_MIN_POSITIVE_RATIO="${CLUSTER_FALLBACK_MIN_POSITIVE_RATIO:-0.02}"

# GPU_ID — 仅设置 CUDA_VISIBLE_DEVICES；Python 仍使用 --device cuda:0（即「可见 GPU 列表中的第 0 块」）
GPU_ID="${GPU_ID:-1}"

# =============================================================================
# 特征提取配置（映射到 Python 的同名/同类参数）
# =============================================================================

# VOXEL_SIZE — --voxel_size：BSE 体素边长（米）
# POOL_MODE — --pool_mode：
#   bse=voxel hash + Otsu split
#   pooled=map-anything 原版 pooled_GT baseline，raw 点全局 SOR 后一次 voxel mean；自动忽略 BSE/Otsu/prepool/global_merge 开关
#   simple=底层简单 voxel mean（保留给消融）
# PREPOOL_MODE — --prepool_mode：per_view=每个 view 先池化；global_only=直接缓存 raw points，Pass 2 再全局池化
# USE_OTSU — --use_otsu：true=使用 Otsu adaptive split；false=固定 tau=0.90
# UNIMODAL_THRESHOLD — --unimodal_threshold：std(sim) 低于该值时跳过 split
POOL_MODE="${POOL_MODE:-bse}"
PREPOOL_MODE="${PREPOOL_MODE:-per_view}"
USE_OTSU="${USE_OTSU:-true}"
UNIMODAL_THRESHOLD="${UNIMODAL_THRESHOLD:-0.02}"
VOXEL_SIZE="${VOXEL_SIZE:-0.05}"
GLOBAL_MERGE="${GLOBAL_MERGE:-true}"
GLOBAL_MERGE_VOXEL_SIZE="${GLOBAL_MERGE_VOXEL_SIZE:-}"
# Hybrid 模式参数：只对高不一致全局 voxel 做选择性 split
HYBRID_SPLIT_MIN_STD="${HYBRID_SPLIT_MIN_STD:-0.05}"
HYBRID_MIN_CLUSTER_SIZE="${HYBRID_MIN_CLUSTER_SIZE:-4}"
HYBRID_MIN_SPLIT_POINTS="${HYBRID_MIN_SPLIT_POINTS:-2}"
# ENABLE_SOR — 为 true 时追加 --enable_sor，并打开 SOR_K/STD（Python 内 sor_k、sor_std_ratio 用默认值）
ENABLE_SOR="${ENABLE_SOR:-true}"

if [ "$POOL_MODE" = "pooled" ]; then
    PREPOOL_MODE="global_only"
    USE_OTSU="false"
    GLOBAL_MERGE="false"
    GLOBAL_MERGE_VOXEL_SIZE=""
fi

# DEPTH_MIN / DEPTH_MAX — 组成 --depth_valid_range：反投影保留的深度区间（米）
# 7Scenes 常用 max=6.0；Indoor6 COLMAP 稀疏深度常用 max=100 避免裁掉远点
if [ "$DATASET_TYPE" = "indoor6" ]; then
    DEPTH_MIN="${DEPTH_MIN:-0.1}"
    DEPTH_MAX="${DEPTH_MAX:-100.0}"
    # PATCH_DEPTH_SAMPLING — --patch_depth_sampling：Indoor6 默认 nearest_valid（邻域最近有效深度）
    PATCH_DEPTH_SAMPLING="${PATCH_DEPTH_SAMPLING:-nearest_valid}"
elif [ "$DATASET_TYPE" = "mushroom" ]; then
    DEPTH_MIN="${DEPTH_MIN:-0.1}"
    DEPTH_MAX="${DEPTH_MAX:-20.0}"
    PATCH_DEPTH_SAMPLING="${PATCH_DEPTH_SAMPLING:-nearest}"
elif [ "$DATASET_TYPE" = "rio10" ]; then
    DEPTH_MIN="${DEPTH_MIN:-0.1}"
    DEPTH_MAX="${DEPTH_MAX:-10.0}"
    # RIO10 sparse depth is sparse/incomplete; Step19 requires nearest_valid for memory geometry.
    PATCH_DEPTH_SAMPLING="${PATCH_DEPTH_SAMPLING:-nearest_valid}"
else
    DEPTH_MIN="${DEPTH_MIN:-0.1}"
    DEPTH_MAX="${DEPTH_MAX:-6.0}"
    PATCH_DEPTH_SAMPLING="${PATCH_DEPTH_SAMPLING:-nearest}"
fi

# USE_PATCH_BASED — --use_patch_based：是否走 patch 网格提取路径
USE_PATCH_BASED="${USE_PATCH_BASED:-false}"
# USE_L2_NORMALIZATION — 仅写入 config_tag 与 extraction_config.json；Python 当前未接 CLI（内部特征融合另有 apply_l2_norm 开关）
USE_L2_NORMALIZATION="${USE_L2_NORMALIZATION:-true}"

# RAY_POOL_STRATEGY — --ray_pool_strategy：体素内视线方向池化（mean/dominant/first/all）
RAY_POOL_STRATEGY="${RAY_POOL_STRATEGY:-mean}"

# DINOV2_CHECKPOINT — --dinov2_checkpoint（mapanything / dinov2 均传入；优先仓库内软链接再回退共享存储）
if [ -z "${DINOV2_CHECKPOINT:-}" ] || [ ! -f "$DINOV2_CHECKPOINT" ]; then
    if [ -f "$_ACE_DEPTH_ROOT/checkpoints/dinov2_vitl14_pretrain.pth" ]; then
        DINOV2_CHECKPOINT="$_ACE_DEPTH_ROOT/checkpoints/dinov2_vitl14_pretrain.pth"
    else
        DINOV2_CHECKPOINT="$ACE_DATA_ROOT/checkpoints/dinov2_vitl14_pretrain.pth"
    fi
fi

# USE_MODEL — --use_model：mapanything | dinov2
USE_MODEL="${USE_MODEL:-mapanything}"

# MODEL_STR / MODEL_CHECKPOINT — --model_str / --model_checkpoint（仅 USE_MODEL=mapanything 时追加）
# MODEL_CONFIG — 可选；若设置则追加 --model_config（Hydra YAML 路径）
MODEL_STR="${MODEL_STR:-mapanything_store_intermediates_ace}"
MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$ACE_DATA_ROOT/checkpoints/facebook_map-anything.pth}"
MAPANYTHING_USE_AMP="${MAPANYTHING_USE_AMP:-false}"
POSTPROCESS_ON_CPU="${POSTPROCESS_ON_CPU:-false}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# DINOV2_INTERMEDIATE_LAYERS — 空格分隔的整数，传给 --dinov2_intermediate_layers；空则 Python 用默认 8 层
DINOV2_INTERMEDIATE_LAYERS="${DINOV2_INTERMEDIATE_LAYERS:-}"

# =============================================================================
# 输出路径（与 Python 第二个位置参数 output_path 对应：memory_bse.pt）
# =============================================================================
# OUTPUT_DIR = OUTPUT_ROOT / SCENE_NAME / ${N_VIEWS}v_${CONFIG_TAG} / TIMESTAMP
# CONFIG_TAG — 由 VOXEL_SIZE、USE_PATCH_BASED、ENABLE_SOR、USE_L2_NORMALIZATION 编码，便于区分实验目录

OUTPUT_ROOT="${OUTPUT_ROOT:-$(cd "$(dirname "$0")" && pwd)/04_evaluation/memory_extract}"

# 构建 config_tag
CONFIG_TAG="v${VOXEL_SIZE}"
if [ "$USE_PATCH_BASED" = "true" ]; then
    CONFIG_TAG="${CONFIG_TAG}_patch"
else
    CONFIG_TAG="${CONFIG_TAG}_bilinear"
fi
if [ "$POOL_MODE" = "pooled" ]; then
    CONFIG_TAG="${CONFIG_TAG}_pooled"
elif [ "$POOL_MODE" = "simple" ]; then
    CONFIG_TAG="${CONFIG_TAG}_simple"
elif [ "$POOL_MODE" = "hybrid" ]; then
    CONFIG_TAG="${CONFIG_TAG}_hybrid_hstd${HYBRID_SPLIT_MIN_STD}_hmin${HYBRID_MIN_CLUSTER_SIZE}_hsplit${HYBRID_MIN_SPLIT_POINTS}"
else
    CONFIG_TAG="${CONFIG_TAG}_bse"
fi
if [ "$POOL_MODE" != "pooled" ] && [ "$USE_OTSU" = "false" ]; then
    CONFIG_TAG="${CONFIG_TAG}_nootsu"
fi
if [ "$POOL_MODE" = "bse" ]; then
    CONFIG_TAG="${CONFIG_TAG}_ut${UNIMODAL_THRESHOLD}"
fi
if [ "$DATASET_LOADER" = "wai" ]; then
    _preproc_tag_transform="$(printf '%s' "$WAI_TRANSFORM" | tr -c '[:alnum:]' '_' | sed 's/_\\+/_/g; s/^_//; s/_$//')"
    if [ "$WAI_AUG_CROP" = "0" ] && [ "$WAI_TRANSFORM" = "imgnorm" ]; then
        CONFIG_TAG="${CONFIG_TAG}_noaug"
    else
        CONFIG_TAG="${CONFIG_TAG}_${_preproc_tag_transform}_aug${WAI_AUG_CROP}"
    fi
    if [ "$WAI_RESOLUTION" != "518" ]; then
        CONFIG_TAG="${CONFIG_TAG}_res${WAI_RESOLUTION}"
    fi
    case "$WAI_VIEW_MODE" in
        original_multiview)
            CONFIG_TAG="${CONFIG_TAG}_original"
            ;;
        fps_flat)
            CONFIG_TAG="${CONFIG_TAG}_fps"
            ;;
        anchor_support)
            CONFIG_TAG="${CONFIG_TAG}_asb"
            case "${ASB_ADAPTIVE,,}" in
                true|1|yes|y|on)
                    CONFIG_TAG="${CONFIG_TAG}_adaptive"
                    ;;
            esac
            if [ "$ASB_CANDIDATE_POOL_RATIO" = "0" ] || [ "$ASB_CANDIDATE_POOL_RATIO" = "0.0" ]; then
                CONFIG_TAG="${CONFIG_TAG}_fullpool"
            else
                CONFIG_TAG="${CONFIG_TAG}_pool${ASB_CANDIDATE_POOL_RATIO}"
            fi
            case "${ASB_MA_SAFE,,}" in
                true|1|yes|y|on)
                    CONFIG_TAG="${CONFIG_TAG}_masafe${ASB_MA_SAFE_POOL_RATIO}"
                    ;;
            esac
            case "${ASB_NATIVE_GROUP,,}" in
                true|1|yes|y|on)
                    CONFIG_TAG="${CONFIG_TAG}_native${ASB_NATIVE_ANCHOR_CANDIDATES}"
                    ;;
            esac
            case "${ENABLE_REFERENCE_POLICY_GATE,,}" in
                true|1|yes|y|on)
                    CONFIG_TAG="${CONFIG_TAG}_rgate${REFERENCE_POLICY_TOP_M}"
                    ;;
            esac
            case "${ASB_POST_REPAIR,,}" in
                true|1|yes|y|on)
                    CONFIG_TAG="${CONFIG_TAG}_prepair${ASB_POST_REPAIR_MAX_SWAPS}"
                    ;;
            esac
            case "${ENABLE_CLUSTER_FALLBACK,,}" in
                true|1|yes|y|on)
                    CONFIG_TAG="${CONFIG_TAG}_cfb${CLUSTER_FALLBACK_MAX_CLUSTERS}"
                    ;;
            esac
            ;;
    esac
fi
if [ "$POOL_MODE" != "pooled" ] && [ "$PREPOOL_MODE" = "global_only" ]; then
    CONFIG_TAG="${CONFIG_TAG}_globalonly"
fi
if [ "$ENABLE_SOR" = "true" ]; then
    CONFIG_TAG="${CONFIG_TAG}_sor"
fi
if [ "$POOL_MODE" = "pooled" ]; then
    :
elif [ "$GLOBAL_MERGE" = "true" ]; then
    CONFIG_TAG="${CONFIG_TAG}_gm"
    if [ -n "$GLOBAL_MERGE_VOXEL_SIZE" ]; then
        CONFIG_TAG="${CONFIG_TAG}v${GLOBAL_MERGE_VOXEL_SIZE}"
    fi
else
    CONFIG_TAG="${CONFIG_TAG}_nogm"
fi
if [ "$USE_L2_NORMALIZATION" = "true" ]; then
    CONFIG_TAG="${CONFIG_TAG}_l2"
fi

if [ "$POOL_MODE" = "hybrid" ]; then
    if [ "$PREPOOL_MODE" != "global_only" ]; then
        echo "ERROR: POOL_MODE=hybrid requires PREPOOL_MODE=global_only, got '$PREPOOL_MODE'"
        exit 1
    fi
    if [ "$GLOBAL_MERGE" = "true" ]; then
        echo "ERROR: POOL_MODE=hybrid requires GLOBAL_MERGE=false"
        exit 1
    fi
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_ROOT}/${SCENE_NAME}/${N_VIEWS}v_${CONFIG_TAG}/${TIMESTAMP}"
case "$POOL_MODE" in
    pooled) OUTPUT_BASENAME="memory_pooled.pt" ;;
    simple) OUTPUT_BASENAME="memory_simple.pt" ;;
    hybrid) OUTPUT_BASENAME="memory_hybrid.pt" ;;
    *)      OUTPUT_BASENAME="memory_bse.pt" ;;
esac
OUTPUT_FILE="${OUTPUT_DIR}/${OUTPUT_BASENAME}"
TEMP_DIR="/dev/shm/ace_bse_${SCENE_NAME}_${N_VIEWS}v_${CONFIG_TAG}_${TIMESTAMP}_gpu${GPU_ID}_pid$$"

# =============================================================================
# PYTHONPATH 设置
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# ace_depth 的上一级（如 ~/project），其下可有软链 uniception/ 覆盖 site-packages 便于改源码
WORKSPACE_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"
MAP_ANYTHING_PATH="${MAP_ANYTHING_PATH:-$ACE_DATA_ROOT/map-anything}"

PYTHONPATH_ADDITIONS=""
if [ -d "$MAP_ANYTHING_PATH" ]; then
    PYTHONPATH_ADDITIONS="${MAP_ANYTHING_PATH}"
fi
# 确保 ace_depth 根目录也在 path 中
PYTHONPATH_ADDITIONS="${PROJECT_ROOT}:${PYTHONPATH_ADDITIONS}"
# 优先使用 WORKSPACE_ROOT/uniception（软链到 conda 里的包），便于本地改 UniCeption
if [ -e "$WORKSPACE_ROOT/uniception/__init__.py" ]; then
    PYTHONPATH_ADDITIONS="${WORKSPACE_ROOT}:${PYTHONPATH_ADDITIONS}"
fi
export PYTHONPATH="${PYTHONPATH_ADDITIONS}:${PYTHONPATH:-}"

# =============================================================================
# 打印配置
# =============================================================================
echo "============================================================"
echo "  BSE Memory Extraction"
echo "============================================================"
echo ""
echo "  Dataset:"
echo "    type:       $DATASET_TYPE"
echo "    loader:     $DATASET_LOADER"
echo "    root:       $DATASET_ROOT"
echo "    path:       $DATASET_PATH"
echo "    scene:      $SCENE_TRAIN → $SCENE_TEST"
echo "    n_views:    $N_VIEWS"
echo "    contract:   $CONTRACT_MODE"
if [ "$DATASET_LOADER" = "wai" ]; then
echo "    preprocessing: transform=$WAI_TRANSFORM data_norm=$WAI_DATA_NORM_TYPE aug_crop=$WAI_AUG_CROP resolution=$WAI_RESOLUTION"
fi
echo "    wai_view_mode: $WAI_VIEW_MODE"
if [ "$WAI_VIEW_MODE" = "anchor_support" ]; then
echo "    anchor_support: alpha=$ASB_ALPHA eps=$ASB_EPS tau=$ASB_TAU ref_lambda=$ASB_REF_LAMBDA anchor_count=$ASB_ANCHOR_COUNT support_per_anchor=$ASB_SUPPORT_PER_ANCHOR support_tau=$ASB_SUPPORT_TAU support_min_neighbors=$ASB_SUPPORT_MIN_NEIGHBORS safe_dist=$ASB_SAFE_DIST_TO_REF far_views=$ASB_FAR_VIEW_BUDGET far_anchors=$ASB_FAR_ANCHOR_BUDGET coverage_beta=$ASB_COVERAGE_BETA candidate_pool_ratio=$ASB_CANDIDATE_POOL_RATIO ma_safe=$ASB_MA_SAFE ma_safe_pool_ratio=$ASB_MA_SAFE_POOL_RATIO native_group=$ASB_NATIVE_GROUP native_anchor_candidates=$ASB_NATIVE_ANCHOR_CANDIDATES"
echo "    adaptive_asb: enabled=$ASB_ADAPTIVE scene_stats=$ASB_SCENE_STATS adaptive_view_count=$ASB_ADAPTIVE_VIEW_COUNT min_views=$ASB_MIN_VIEWS ref_max=$ASB_REF_DIST_MAX_LIMIT ref_mean=$ASB_REF_DIST_MEAN_LIMIT target_cov_mean=$ASB_TARGET_COVERAGE_MEAN target_cov_max=$ASB_TARGET_COVERAGE_MAX min_gain=$ASB_MIN_COVERAGE_GAIN patience=$ASB_GAIN_PATIENCE"
echo "    post_repair: enabled=$ASB_POST_REPAIR max_swaps=$ASB_POST_REPAIR_MAX_SWAPS tail_percentile=$ASB_POST_REPAIR_TAIL_PERCENTILE min_tail_improvement_m=$ASB_POST_REPAIR_MIN_TAIL_IMPROVEMENT_M cluster_top_k=$ASB_POST_REPAIR_CLUSTER_TOP_K"
echo "    pose_prune: enabled=$ASB_POSE_PRUNE min_views=$ASB_POSE_PRUNE_MIN_VIEWS keep_ratio=$ASB_POSE_PRUNE_KEEP_RATIO mad_k=$ASB_POSE_PRUNE_MAD_K"
echo "    reference_policy_gate: enabled=$ENABLE_REFERENCE_POLICY_GATE top_m=$REFERENCE_POLICY_TOP_M min_links=$REFERENCE_POLICY_LIGHT_MIN_SELECTED_LINKS probe_q90=$REFERENCE_POLICY_PROBE_Q90_M probe_max=$REFERENCE_POLICY_PROBE_MAX_M"
echo "    cluster_fallback: enabled=$ENABLE_CLUSTER_FALLBACK max_clusters=$CLUSTER_FALLBACK_MAX_CLUSTERS max_depth=$CLUSTER_FALLBACK_MAX_SPLIT_DEPTH min_cluster_size=$CLUSTER_FALLBACK_MIN_CLUSTER_SIZE min_views=$CLUSTER_FALLBACK_MIN_VIEWS_PER_CLUSTER min_pos_ratio=$CLUSTER_FALLBACK_MIN_POSITIVE_RATIO"
fi
echo "    temp_dir:   $TEMP_DIR"
echo ""
echo "  BSE Config:"
echo "    voxel_size: $VOXEL_SIZE"
echo "    pool_mode:  $POOL_MODE"
echo "    prepool_mode: $PREPOOL_MODE"
echo "    otsu:       $USE_OTSU"
echo "    unimodal_threshold: $UNIMODAL_THRESHOLD"
if [ "$POOL_MODE" = "hybrid" ]; then
echo "    hybrid_split_min_std: $HYBRID_SPLIT_MIN_STD"
echo "    hybrid_min_cluster_size: $HYBRID_MIN_CLUSTER_SIZE"
echo "    hybrid_min_split_points: $HYBRID_MIN_SPLIT_POINTS"
fi
echo "    sor:        $ENABLE_SOR"
echo "    global_merge: $GLOBAL_MERGE"
echo "    global_merge_voxel_size: ${GLOBAL_MERGE_VOXEL_SIZE:-$VOXEL_SIZE}"
echo "    ray_pool:   $RAY_POOL_STRATEGY"
echo ""
echo "  Depth:"
echo "    valid_range: [$DEPTH_MIN, $DEPTH_MAX]"
echo "    grid_sampling: $PATCH_DEPTH_SAMPLING  (fps_memory: indoor6/rio10 sparse→nearest_valid, 7scenes→nearest)"
echo ""
echo "  Feature:"
echo "    use_model:   $USE_MODEL"
echo "    patch_based: $USE_PATCH_BASED"
echo "    l2_norm:     $USE_L2_NORMALIZATION"
echo "    dinov2:      $DINOV2_CHECKPOINT"
if [ "$USE_MODEL" = "mapanything" ]; then
echo "    model_str:   $MODEL_STR"
echo "    model_ckpt:  $MODEL_CHECKPOINT"
fi
if [ "$USE_MODEL" = "dinov2" ] && [ -n "$DINOV2_INTERMEDIATE_LAYERS" ]; then
echo "    intermediates: $DINOV2_INTERMEDIATE_LAYERS"
fi
echo ""
echo "  Output:"
echo "    dir:  $OUTPUT_DIR"
echo "    file: $OUTPUT_FILE"
echo ""
echo "  GPU: $GPU_ID"
echo "============================================================"

# 检查数据集路径
if [ ! -d "$DATASET_PATH" ]; then
    echo "ERROR: Dataset path not found: $DATASET_PATH"
    echo "  DATASET_TYPE=$DATASET_TYPE SCENE_TRAIN=$SCENE_TRAIN"
    echo "  Expected: $DATASET_PATH"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# =============================================================================
# 构建 Python 参数（与 python -m ... run_memory_extraction 一致）
# =============================================================================
# 未在此列出的 ExtractionConfig 项将使用 Python 默认值，例如：
#   --temp_dir /dev/shm, --save_all_ray_strategies, pose_eval_*, --use_bse/--use_otsu 等
PYTHON_ARGS=""
PYTHON_ARGS="$PYTHON_ARGS $DATASET_PATH"
PYTHON_ARGS="$PYTHON_ARGS $OUTPUT_FILE"
PYTHON_ARGS="$PYTHON_ARGS --n_memory $N_VIEWS"
# 固定 cuda:0：配合上方 CUDA_VISIBLE_DEVICES=$GPU_ID，即使用物理 GPU_ID 对应的那块卡
PYTHON_ARGS="$PYTHON_ARGS --device cuda:0"
PYTHON_ARGS="$PYTHON_ARGS --temp_dir $TEMP_DIR"
PYTHON_ARGS="$PYTHON_ARGS --voxel_size $VOXEL_SIZE"
PYTHON_ARGS="$PYTHON_ARGS --pool_mode $POOL_MODE"
PYTHON_ARGS="$PYTHON_ARGS --prepool_mode $PREPOOL_MODE"
PYTHON_ARGS="$PYTHON_ARGS --use_otsu $USE_OTSU"
PYTHON_ARGS="$PYTHON_ARGS --unimodal_threshold $UNIMODAL_THRESHOLD"
PYTHON_ARGS="$PYTHON_ARGS --global_merge $GLOBAL_MERGE"
if [ -n "$GLOBAL_MERGE_VOXEL_SIZE" ]; then
    PYTHON_ARGS="$PYTHON_ARGS --global_merge_voxel_size $GLOBAL_MERGE_VOXEL_SIZE"
fi
if [ "$POOL_MODE" = "hybrid" ]; then
    PYTHON_ARGS="$PYTHON_ARGS --hybrid_split_min_std $HYBRID_SPLIT_MIN_STD"
    PYTHON_ARGS="$PYTHON_ARGS --hybrid_min_cluster_size $HYBRID_MIN_CLUSTER_SIZE"
    PYTHON_ARGS="$PYTHON_ARGS --hybrid_min_split_points $HYBRID_MIN_SPLIT_POINTS"
fi
PYTHON_ARGS="$PYTHON_ARGS --dataset_type $DATASET_TYPE"
PYTHON_ARGS="$PYTHON_ARGS --dataset_loader $DATASET_LOADER"
PYTHON_ARGS="$PYTHON_ARGS --contract_mode $CONTRACT_MODE"
PYTHON_ARGS="$PYTHON_ARGS --dataset_transform $WAI_TRANSFORM"
PYTHON_ARGS="$PYTHON_ARGS --dataset_data_norm_type $WAI_DATA_NORM_TYPE"
PYTHON_ARGS="$PYTHON_ARGS --dataset_aug_crop $WAI_AUG_CROP"
PYTHON_ARGS="$PYTHON_ARGS --dataset_resolution $WAI_RESOLUTION"
PYTHON_ARGS="$PYTHON_ARGS --wai_view_mode $WAI_VIEW_MODE"
if [ "$WAI_VIEW_MODE" = "anchor_support" ]; then
    PYTHON_ARGS="$PYTHON_ARGS --anchor_support_alpha $ASB_ALPHA"
    PYTHON_ARGS="$PYTHON_ARGS --anchor_support_eps $ASB_EPS"
    PYTHON_ARGS="$PYTHON_ARGS --anchor_support_tau $ASB_TAU"
    PYTHON_ARGS="$PYTHON_ARGS --covis_ref_lambda $ASB_REF_LAMBDA"
    PYTHON_ARGS="$PYTHON_ARGS --covis_anchor_count $ASB_ANCHOR_COUNT"
    PYTHON_ARGS="$PYTHON_ARGS --covis_support_per_anchor $ASB_SUPPORT_PER_ANCHOR"
    PYTHON_ARGS="$PYTHON_ARGS --covis_support_tau $ASB_SUPPORT_TAU"
    PYTHON_ARGS="$PYTHON_ARGS --covis_support_min_neighbors $ASB_SUPPORT_MIN_NEIGHBORS"
    PYTHON_ARGS="$PYTHON_ARGS --covis_safe_dist_to_ref $ASB_SAFE_DIST_TO_REF"
    PYTHON_ARGS="$PYTHON_ARGS --covis_far_view_budget $ASB_FAR_VIEW_BUDGET"
    PYTHON_ARGS="$PYTHON_ARGS --covis_far_anchor_budget $ASB_FAR_ANCHOR_BUDGET"
    PYTHON_ARGS="$PYTHON_ARGS --covis_coverage_beta $ASB_COVERAGE_BETA"
    PYTHON_ARGS="$PYTHON_ARGS --covis_candidate_pool_ratio $ASB_CANDIDATE_POOL_RATIO"
    case "${ASB_MA_SAFE,,}" in
        true|1|yes|y|on)
            PYTHON_ARGS="$PYTHON_ARGS --covis_ma_safe_asb"
            ;;
    esac
    PYTHON_ARGS="$PYTHON_ARGS --covis_ma_safe_pool_ratio $ASB_MA_SAFE_POOL_RATIO"
    case "${ASB_NATIVE_GROUP,,}" in
        true|1|yes|y|on)
            PYTHON_ARGS="$PYTHON_ARGS --covis_native_group_asb"
            ;;
    esac
    PYTHON_ARGS="$PYTHON_ARGS --covis_native_anchor_candidates $ASB_NATIVE_ANCHOR_CANDIDATES"
    case "${ASB_ADAPTIVE,,}" in
        true|1|yes|y|on)
            PYTHON_ARGS="$PYTHON_ARGS --covis_adaptive_asb"
            ;;
    esac
    case "${ASB_ADAPTIVE_VIEW_COUNT,,}" in
        true|1|yes|y|on)
            PYTHON_ARGS="$PYTHON_ARGS --covis_adaptive_view_count"
            ;;
    esac
    case "${ASB_SCENE_STATS,,}" in
        false|0|no|n|off)
            PYTHON_ARGS="$PYTHON_ARGS --covis_disable_adaptive_scene_stats"
            ;;
    esac
    PYTHON_ARGS="$PYTHON_ARGS --covis_adaptive_min_views $ASB_MIN_VIEWS"
    PYTHON_ARGS="$PYTHON_ARGS --covis_ref_dist_max_limit $ASB_REF_DIST_MAX_LIMIT"
    PYTHON_ARGS="$PYTHON_ARGS --covis_ref_dist_mean_limit $ASB_REF_DIST_MEAN_LIMIT"
    PYTHON_ARGS="$PYTHON_ARGS --covis_min_coverage_gain $ASB_MIN_COVERAGE_GAIN"
    PYTHON_ARGS="$PYTHON_ARGS --covis_gain_patience $ASB_GAIN_PATIENCE"
    PYTHON_ARGS="$PYTHON_ARGS --covis_target_coverage_mean $ASB_TARGET_COVERAGE_MEAN"
    PYTHON_ARGS="$PYTHON_ARGS --covis_target_coverage_max $ASB_TARGET_COVERAGE_MAX"
    case "${ASB_POST_REPAIR,,}" in
        true|1|yes|y|on)
            PYTHON_ARGS="$PYTHON_ARGS --covis_post_repair_asb"
            ;;
    esac
    PYTHON_ARGS="$PYTHON_ARGS --covis_post_repair_max_swaps $ASB_POST_REPAIR_MAX_SWAPS"
    PYTHON_ARGS="$PYTHON_ARGS --covis_post_repair_tail_percentile $ASB_POST_REPAIR_TAIL_PERCENTILE"
    PYTHON_ARGS="$PYTHON_ARGS --covis_post_repair_min_tail_improvement_m $ASB_POST_REPAIR_MIN_TAIL_IMPROVEMENT_M"
    PYTHON_ARGS="$PYTHON_ARGS --covis_post_repair_cluster_top_k $ASB_POST_REPAIR_CLUSTER_TOP_K"
    case "${ASB_POSE_PRUNE,,}" in
        true|1|yes|y|on)
            PYTHON_ARGS="$PYTHON_ARGS --pose_prune_after_infer"
            ;;
    esac
    PYTHON_ARGS="$PYTHON_ARGS --pose_prune_min_views $ASB_POSE_PRUNE_MIN_VIEWS"
    PYTHON_ARGS="$PYTHON_ARGS --pose_prune_min_keep_ratio $ASB_POSE_PRUNE_KEEP_RATIO"
    PYTHON_ARGS="$PYTHON_ARGS --pose_prune_mad_k $ASB_POSE_PRUNE_MAD_K"
    case "${ENABLE_REFERENCE_POLICY_GATE,,}" in
        true|1|yes|y|on)
            PYTHON_ARGS="$PYTHON_ARGS --enable_reference_policy_gate"
            ;;
    esac
    PYTHON_ARGS="$PYTHON_ARGS --reference_policy_top_m $REFERENCE_POLICY_TOP_M"
    PYTHON_ARGS="$PYTHON_ARGS --reference_policy_light_min_selected_links $REFERENCE_POLICY_LIGHT_MIN_SELECTED_LINKS"
    PYTHON_ARGS="$PYTHON_ARGS --reference_policy_probe_q90_m $REFERENCE_POLICY_PROBE_Q90_M"
    PYTHON_ARGS="$PYTHON_ARGS --reference_policy_probe_max_m $REFERENCE_POLICY_PROBE_MAX_M"
    case "${ENABLE_CLUSTER_FALLBACK,,}" in
        true|1|yes|y|on)
            PYTHON_ARGS="$PYTHON_ARGS --enable_cluster_fallback"
            ;;
    esac
    PYTHON_ARGS="$PYTHON_ARGS --cluster_fallback_max_clusters $CLUSTER_FALLBACK_MAX_CLUSTERS"
    PYTHON_ARGS="$PYTHON_ARGS --cluster_fallback_max_split_depth $CLUSTER_FALLBACK_MAX_SPLIT_DEPTH"
    PYTHON_ARGS="$PYTHON_ARGS --cluster_fallback_min_cluster_size $CLUSTER_FALLBACK_MIN_CLUSTER_SIZE"
    PYTHON_ARGS="$PYTHON_ARGS --cluster_fallback_min_views_per_cluster $CLUSTER_FALLBACK_MIN_VIEWS_PER_CLUSTER"
    PYTHON_ARGS="$PYTHON_ARGS --cluster_fallback_min_positive_ratio $CLUSTER_FALLBACK_MIN_POSITIVE_RATIO"
fi
PYTHON_ARGS="$PYTHON_ARGS --scene_name $SCENE_TRAIN"
PYTHON_ARGS="$PYTHON_ARGS --depth_valid_range $DEPTH_MIN $DEPTH_MAX"
PYTHON_ARGS="$PYTHON_ARGS --patch_depth_sampling $PATCH_DEPTH_SAMPLING"
PYTHON_ARGS="$PYTHON_ARGS --dinov2_checkpoint $DINOV2_CHECKPOINT"
PYTHON_ARGS="$PYTHON_ARGS --ray_pool_strategy $RAY_POOL_STRATEGY"
PYTHON_ARGS="$PYTHON_ARGS --use_model $USE_MODEL"

# Debug dump (optional)
if [ "$DEBUG_DUMP" = "true" ]; then
    PYTHON_ARGS="$PYTHON_ARGS --debug_dump"
    if [ -n "$DEBUG_DUMP_MAX_VIEWS" ]; then
        PYTHON_ARGS="$PYTHON_ARGS --debug_dump_max_views $DEBUG_DUMP_MAX_VIEWS"
    fi
    if [ "$DEBUG_DUMP_SAVE_NPZ" = "true" ]; then
        PYTHON_ARGS="$PYTHON_ARGS --debug_dump_save_npz"
    fi
fi

if [ "$ENABLE_SOR" = "true" ]; then
    PYTHON_ARGS="$PYTHON_ARGS --enable_sor"
fi
case "${MAPANYTHING_USE_AMP,,}" in
    true|1|yes|y|on)
        PYTHON_ARGS="$PYTHON_ARGS --mapanything_use_amp"
        ;;
esac
case "${POSTPROCESS_ON_CPU,,}" in
    true|1|yes|y|on)
        PYTHON_ARGS="$PYTHON_ARGS --postprocess_on_cpu"
        ;;
esac

# MapAnything model config
if [ "$USE_MODEL" = "mapanything" ]; then
    PYTHON_ARGS="$PYTHON_ARGS --model_str $MODEL_STR"
    PYTHON_ARGS="$PYTHON_ARGS --model_checkpoint $MODEL_CHECKPOINT"
    if [ -n "$MODEL_CONFIG" ]; then
        PYTHON_ARGS="$PYTHON_ARGS --model_config $MODEL_CONFIG"
    fi
fi

# DINOv2 multi-scale config
if [ "$USE_MODEL" = "dinov2" ] && [ -n "$DINOV2_INTERMEDIATE_LAYERS" ]; then
    PYTHON_ARGS="$PYTHON_ARGS --dinov2_intermediate_layers $DINOV2_INTERMEDIATE_LAYERS"
fi

# =============================================================================
# 保存配置快照
# =============================================================================
if [ -n "$GLOBAL_MERGE_VOXEL_SIZE" ]; then
    GLOBAL_MERGE_VOXEL_SIZE_JSON="$GLOBAL_MERGE_VOXEL_SIZE"
else
    GLOBAL_MERGE_VOXEL_SIZE_JSON="null"
fi

cat > "${OUTPUT_DIR}/extraction_config.json" <<EOF
{
    "dataset_type": "$DATASET_TYPE",
    "dataset_loader": "$DATASET_LOADER",
    "dataset_root": "$DATASET_ROOT",
    "dataset_path": "$DATASET_PATH",
    "scene_train": "$SCENE_TRAIN",
    "scene_test": "$SCENE_TEST",
    "n_views": $N_VIEWS,
    "contract_mode": "$CONTRACT_MODE",
    "wai_transform": "$WAI_TRANSFORM",
    "wai_data_norm_type": "$WAI_DATA_NORM_TYPE",
    "wai_aug_crop": $WAI_AUG_CROP,
    "wai_resolution": $WAI_RESOLUTION,
    "wai_view_mode": "$WAI_VIEW_MODE",
    "anchor_support_alpha": $ASB_ALPHA,
    "anchor_support_eps": $ASB_EPS,
    "anchor_support_tau": $ASB_TAU,
    "anchor_support_ref_lambda": $ASB_REF_LAMBDA,
    "anchor_support_anchor_count": $ASB_ANCHOR_COUNT,
    "anchor_support_support_per_anchor": $ASB_SUPPORT_PER_ANCHOR,
    "anchor_support_support_tau": $ASB_SUPPORT_TAU,
    "anchor_support_support_min_neighbors": $ASB_SUPPORT_MIN_NEIGHBORS,
    "anchor_support_safe_dist_to_ref": $ASB_SAFE_DIST_TO_REF,
    "anchor_support_far_view_budget": $ASB_FAR_VIEW_BUDGET,
    "anchor_support_far_anchor_budget": $ASB_FAR_ANCHOR_BUDGET,
    "anchor_support_coverage_beta": $ASB_COVERAGE_BETA,
    "anchor_support_candidate_pool_ratio": $ASB_CANDIDATE_POOL_RATIO,
    "anchor_support_ma_safe": "$ASB_MA_SAFE",
    "anchor_support_ma_safe_pool_ratio": $ASB_MA_SAFE_POOL_RATIO,
    "anchor_support_native_group": "$ASB_NATIVE_GROUP",
    "anchor_support_native_anchor_candidates": $ASB_NATIVE_ANCHOR_CANDIDATES,
    "anchor_support_adaptive": "$ASB_ADAPTIVE",
    "anchor_support_adaptive_scene_stats": "$ASB_SCENE_STATS",
    "anchor_support_adaptive_view_count": "$ASB_ADAPTIVE_VIEW_COUNT",
    "anchor_support_min_views": $ASB_MIN_VIEWS,
    "anchor_support_ref_dist_max_limit": $ASB_REF_DIST_MAX_LIMIT,
    "anchor_support_ref_dist_mean_limit": $ASB_REF_DIST_MEAN_LIMIT,
    "anchor_support_min_coverage_gain": $ASB_MIN_COVERAGE_GAIN,
    "anchor_support_gain_patience": $ASB_GAIN_PATIENCE,
    "anchor_support_target_coverage_mean": $ASB_TARGET_COVERAGE_MEAN,
    "anchor_support_target_coverage_max": $ASB_TARGET_COVERAGE_MAX,
    "anchor_support_post_repair": "$ASB_POST_REPAIR",
    "anchor_support_post_repair_max_swaps": $ASB_POST_REPAIR_MAX_SWAPS,
    "anchor_support_post_repair_tail_percentile": $ASB_POST_REPAIR_TAIL_PERCENTILE,
    "anchor_support_post_repair_min_tail_improvement_m": $ASB_POST_REPAIR_MIN_TAIL_IMPROVEMENT_M,
    "anchor_support_post_repair_cluster_top_k": $ASB_POST_REPAIR_CLUSTER_TOP_K,
    "anchor_support_pose_prune": "$ASB_POSE_PRUNE",
    "anchor_support_pose_prune_min_views": $ASB_POSE_PRUNE_MIN_VIEWS,
    "anchor_support_pose_prune_keep_ratio": $ASB_POSE_PRUNE_KEEP_RATIO,
    "anchor_support_pose_prune_mad_k": $ASB_POSE_PRUNE_MAD_K,
    "enable_reference_policy_gate": "$ENABLE_REFERENCE_POLICY_GATE",
    "reference_policy_top_m": $REFERENCE_POLICY_TOP_M,
    "reference_policy_light_min_selected_links": $REFERENCE_POLICY_LIGHT_MIN_SELECTED_LINKS,
    "reference_policy_probe_q90_m": $REFERENCE_POLICY_PROBE_Q90_M,
    "reference_policy_probe_max_m": $REFERENCE_POLICY_PROBE_MAX_M,
    "enable_cluster_fallback": "$ENABLE_CLUSTER_FALLBACK",
    "cluster_fallback_max_clusters": $CLUSTER_FALLBACK_MAX_CLUSTERS,
    "cluster_fallback_max_split_depth": $CLUSTER_FALLBACK_MAX_SPLIT_DEPTH,
    "cluster_fallback_min_cluster_size": $CLUSTER_FALLBACK_MIN_CLUSTER_SIZE,
    "cluster_fallback_min_views_per_cluster": $CLUSTER_FALLBACK_MIN_VIEWS_PER_CLUSTER,
    "cluster_fallback_min_positive_ratio": $CLUSTER_FALLBACK_MIN_POSITIVE_RATIO,
    "voxel_size": $VOXEL_SIZE,
    "pool_mode": "$POOL_MODE",
    "prepool_mode": "$PREPOOL_MODE",
    "use_otsu": "$USE_OTSU",
    "unimodal_threshold": $UNIMODAL_THRESHOLD,
    "hybrid_split_min_std": $HYBRID_SPLIT_MIN_STD,
    "hybrid_min_cluster_size": $HYBRID_MIN_CLUSTER_SIZE,
    "hybrid_min_split_points": $HYBRID_MIN_SPLIT_POINTS,
    "enable_sor": "$ENABLE_SOR",
    "global_merge": "$GLOBAL_MERGE",
    "global_merge_voxel_size": $GLOBAL_MERGE_VOXEL_SIZE_JSON,
    "depth_valid_range": [$DEPTH_MIN, $DEPTH_MAX],
    "patch_depth_sampling": "$PATCH_DEPTH_SAMPLING",
    "use_patch_based": "$USE_PATCH_BASED",
    "use_l2_normalization": "$USE_L2_NORMALIZATION",
    "ray_pool_strategy": "$RAY_POOL_STRATEGY",
    "dinov2_checkpoint": "$DINOV2_CHECKPOINT",
    "use_model": "$USE_MODEL",
    "model_str": "$MODEL_STR",
    "model_checkpoint": "$MODEL_CHECKPOINT",
    "mapanything_use_amp": "$MAPANYTHING_USE_AMP",
    "postprocess_on_cpu": "$POSTPROCESS_ON_CPU",
    "dinov2_intermediate_layers": "$DINOV2_INTERMEDIATE_LAYERS",
    "gpu_id": $GPU_ID,
    "temp_dir": "$TEMP_DIR",
    "config_tag": "$CONFIG_TAG",
    "timestamp": "$TIMESTAMP"
}
EOF

# =============================================================================
# 运行
# =============================================================================
echo ""
echo "--- Running extraction ---"
CUDA_VISIBLE_DEVICES=$GPU_ID PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF" python -u -m ace_dinov2_lmc.memory_extraction.run_memory_extraction \
    $PYTHON_ARGS \
    2>&1 | tee "${OUTPUT_DIR}/extraction_log.txt"

EXIT_CODE=${PIPESTATUS[0]}

# =============================================================================
# 结果
# =============================================================================
echo ""
if [ $EXIT_CODE -eq 0 ] && [ -f "$OUTPUT_FILE" ]; then
    FILE_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
    echo "============================================================"
    echo "  Success!"
    echo "============================================================"
    echo "  Output: $OUTPUT_FILE"
    echo "  Size:   $FILE_SIZE"
    echo "  Config: ${OUTPUT_DIR}/extraction_config.json"
    echo "  Log:    ${OUTPUT_DIR}/extraction_log.txt"
    echo ""
    echo "  Output tree:"
    echo "    ${OUTPUT_ROOT}/"
    echo "    └── ${SCENE_NAME}/"
    echo "        └── ${N_VIEWS}v_${CONFIG_TAG}/"
    echo "            └── ${TIMESTAMP}/"
    echo "                ├── ${OUTPUT_BASENAME}"
    echo "                ├── extraction_config.json"
    echo "                └── extraction_log.txt"
else
    echo "============================================================"
    echo "  ERROR: Extraction failed (exit code: $EXIT_CODE)"
    echo "============================================================"
    exit 1
fi
