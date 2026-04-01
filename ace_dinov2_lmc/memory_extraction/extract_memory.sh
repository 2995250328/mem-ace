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
        7scenes)  DATASET_ROOT="/mnt/storage/xwh/mapanything-dataset/wai_data/7scenes" ;;
        indoor6)  DATASET_ROOT="/mnt/storage/xwh/mapanything-dataset/wai_data/indoor6" ;;
        *)        DATASET_ROOT="/mnt/storage/xwh" ;;
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

# GPU_ID — 仅设置 CUDA_VISIBLE_DEVICES；Python 仍使用 --device cuda:0（即「可见 GPU 列表中的第 0 块」）
GPU_ID="${GPU_ID:-1}"

# =============================================================================
# 特征提取配置（映射到 Python 的同名/同类参数）
# =============================================================================

# VOXEL_SIZE — --voxel_size：BSE 体素边长（米）
# USE_OTSU — 记录在 JSON；当前脚本未传 --use_otsu/--no-use_otsu，Python 默认 use_otsu=True
USE_OTSU="${USE_OTSU:-true}"
VOXEL_SIZE="${VOXEL_SIZE:-0.05}"
# ENABLE_SOR — 为 true 时追加 --enable_sor，并打开 SOR_K/STD（Python 内 sor_k、sor_std_ratio 用默认值）
ENABLE_SOR="${ENABLE_SOR:-true}"

# DEPTH_MIN / DEPTH_MAX — 组成 --depth_valid_range：反投影保留的深度区间（米）
# 7Scenes 常用 max=6.0；Indoor6 COLMAP 稀疏深度常用 max=100 避免裁掉远点
if [ "$DATASET_TYPE" = "indoor6" ]; then
    DEPTH_MIN="${DEPTH_MIN:-0.02}"
    DEPTH_MAX="${DEPTH_MAX:-100.0}"
    # PATCH_DEPTH_SAMPLING — --patch_depth_sampling：Indoor6 默认 nearest_valid（邻域最近有效深度）
    PATCH_DEPTH_SAMPLING="${PATCH_DEPTH_SAMPLING:-nearest_valid}"
else
    DEPTH_MIN="${DEPTH_MIN:-0.02}"
    DEPTH_MAX="${DEPTH_MAX:-6.0}"
    PATCH_DEPTH_SAMPLING="${PATCH_DEPTH_SAMPLING:-nearest}"
fi

# USE_PATCH_BASED — --use_patch_based：是否走 patch 网格提取路径
USE_PATCH_BASED="${USE_PATCH_BASED:-false}"
# USE_L2_NORMALIZATION — 仅写入 config_tag 与 extraction_config.json；Python 当前未接 CLI（内部特征融合另有 apply_l2_norm 开关）
USE_L2_NORMALIZATION="${USE_L2_NORMALIZATION:-true}"

# RAY_POOL_STRATEGY — --ray_pool_strategy：体素内视线方向池化（mean/dominant/first/all）
RAY_POOL_STRATEGY="${RAY_POOL_STRATEGY:-mean}"

# DINOV2_CHECKPOINT — --dinov2_checkpoint（脚本始终传入；mapanything 模式也会带上路径，Python 仅在 dinov2 模式使用）
DINOV2_CHECKPOINT="${DINOV2_CHECKPOINT:-/mnt/storage/xwh/checkpoints/dinov2_vitl14_pretrain.pth}"

# USE_MODEL — --use_model：mapanything | dinov2
USE_MODEL="${USE_MODEL:-mapanything}"

# MODEL_STR / MODEL_CHECKPOINT — --model_str / --model_checkpoint（仅 USE_MODEL=mapanything 时追加）
# MODEL_CONFIG — 可选；若设置则追加 --model_config（Hydra YAML 路径）
MODEL_STR="${MODEL_STR:-mapanything_store_intermediates_ace}"
MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-/mnt/storage/xwh/checkpoints/facebook_map-anything.pth}"

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
if [ "$ENABLE_SOR" = "true" ]; then
    CONFIG_TAG="${CONFIG_TAG}_sor"
fi
if [ "$USE_L2_NORMALIZATION" = "true" ]; then
    CONFIG_TAG="${CONFIG_TAG}_l2"
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_ROOT}/${SCENE_NAME}/${N_VIEWS}v_${CONFIG_TAG}/${TIMESTAMP}"
OUTPUT_FILE="${OUTPUT_DIR}/memory_bse.pt"

# =============================================================================
# PYTHONPATH 设置
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MAP_ANYTHING_PATH="/home/xwh/project/map-anything"

PYTHONPATH_ADDITIONS=""
if [ -d "$MAP_ANYTHING_PATH" ]; then
    PYTHONPATH_ADDITIONS="${MAP_ANYTHING_PATH}"
fi
# 确保 ace_depth 根目录也在 path 中
PYTHONPATH_ADDITIONS="${PROJECT_ROOT}:${PYTHONPATH_ADDITIONS}"
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
echo ""
echo "  BSE Config:"
echo "    voxel_size: $VOXEL_SIZE"
echo "    otsu:       $USE_OTSU"
echo "    sor:        $ENABLE_SOR"
echo "    ray_pool:   $RAY_POOL_STRATEGY"
echo ""
echo "  Depth:"
echo "    valid_range: [$DEPTH_MIN, $DEPTH_MAX]"
echo "    grid_sampling: $PATCH_DEPTH_SAMPLING  (fps_memory: indoor6→nearest_valid, 7scenes→nearest)"
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
PYTHON_ARGS="$PYTHON_ARGS --voxel_size $VOXEL_SIZE"
PYTHON_ARGS="$PYTHON_ARGS --dataset_type $DATASET_TYPE"
PYTHON_ARGS="$PYTHON_ARGS --dataset_loader $DATASET_LOADER"
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
cat > "${OUTPUT_DIR}/extraction_config.json" <<EOF
{
    "dataset_type": "$DATASET_TYPE",
    "dataset_loader": "$DATASET_LOADER",
    "dataset_root": "$DATASET_ROOT",
    "dataset_path": "$DATASET_PATH",
    "scene_train": "$SCENE_TRAIN",
    "scene_test": "$SCENE_TEST",
    "n_views": $N_VIEWS,
    "voxel_size": $VOXEL_SIZE,
    "use_otsu": "$USE_OTSU",
    "enable_sor": "$ENABLE_SOR",
    "depth_valid_range": [$DEPTH_MIN, $DEPTH_MAX],
    "patch_depth_sampling": "$PATCH_DEPTH_SAMPLING",
    "use_patch_based": "$USE_PATCH_BASED",
    "use_l2_normalization": "$USE_L2_NORMALIZATION",
    "ray_pool_strategy": "$RAY_POOL_STRATEGY",
    "dinov2_checkpoint": "$DINOV2_CHECKPOINT",
    "use_model": "$USE_MODEL",
    "model_str": "$MODEL_STR",
    "model_checkpoint": "$MODEL_CHECKPOINT",
    "dinov2_intermediate_layers": "$DINOV2_INTERMEDIATE_LAYERS",
    "gpu_id": $GPU_ID,
    "config_tag": "$CONFIG_TAG",
    "timestamp": "$TIMESTAMP"
}
EOF

# =============================================================================
# 运行
# =============================================================================
echo ""
echo "--- Running extraction ---"
CUDA_VISIBLE_DEVICES=$GPU_ID PYTHONUNBUFFERED=1 python -u -m ace_dinov2_lmc.memory_extraction.run_memory_extraction \
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
    echo "                ├── memory_bse.pt"
    echo "                ├── extraction_config.json"
    echo "                └── extraction_log.txt"
else
    echo "============================================================"
    echo "  ERROR: Extraction failed (exit code: $EXIT_CODE)"
    echo "============================================================"
    exit 1
fi
