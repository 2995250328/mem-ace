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
# 配置参数
# =============================================================================

# --- 数据集与场景 ---
# DATASET_TYPE: 7scenes | indoor6 | custom
# SCENE_TRAIN:  场景名，如 chess_train, scene2a_train
# SCENE_TEST:   测试场景名；留空则自动推断（chess_train → chess_test）
# DATASET_LOADER: wai | ace  (数据集加载器：WAI格式或ACE格式)
DATASET_TYPE="${DATASET_TYPE:-7scenes}"
SCENE_TRAIN="${SCENE_TRAIN:-chess_train}"
SCENE_TEST="${SCENE_TEST:-}"
DATASET_LOADER="${DATASET_LOADER:-wai}"

# --- 数据集根目录（根据 DATASET_TYPE 自动选择） ---
# WAI 格式数据集的 ROOT 目录（包含 <scene>_train/ 子目录）
# SevenScenesWAI(ROOT=..., specific_scene_name=chess_train, ...)
if [ -z "$DATASET_ROOT" ]; then
    case "$DATASET_TYPE" in
        7scenes)  DATASET_ROOT="/data/xwh/mapanything-dataset/wai_data/7scenes" ;;
        indoor6)  DATASET_ROOT="/data/xwh/mapanything-dataset/wai_data/indoor6" ;;
        *)        DATASET_ROOT="/data/xwh" ;;
    esac
fi

# --- DATASET_PATH = DATASET_ROOT（WAI dataset 的 ROOT） ---
DATASET_PATH="$DATASET_ROOT"

# --- 自动推导 SCENE_TEST ---
if [ -z "$SCENE_TEST" ]; then
    case "$SCENE_TRAIN" in
        *_train) SCENE_TEST="${SCENE_TRAIN%_train}_test" ;;
        *_val)   SCENE_TEST="${SCENE_TRAIN%_val}_test"   ;;
        *_test)  SCENE_TEST="$SCENE_TRAIN"               ;;
        *)       SCENE_TEST="${SCENE_TRAIN}_test"         ;;
    esac
fi

# --- 推导场景短名（用于输出目录） ---
SCENE_NAME="${SCENE_TRAIN%_train}"

# --- Memory 视角数 ---
N_VIEWS="${N_VIEWS:-20}"

# --- GPU ---
GPU_ID="${GPU_ID:-3}"

# =============================================================================
# 特征提取配置
# =============================================================================

# BSE pooling
VOXEL_SIZE="${VOXEL_SIZE:-0.05}"
USE_OTSU="${USE_OTSU:-true}"
ENABLE_SOR="${ENABLE_SOR:-false}"

# 深度有效范围（米）
# 7Scenes: max=6.0 | Indoor6: max=100.0（COLMAP 稀疏深度，远距几何不裁）
if [ "$DATASET_TYPE" = "indoor6" ]; then
    DEPTH_MIN="${DEPTH_MIN:-0.1}"
    DEPTH_MAX="${DEPTH_MAX:-100.0}"
    # 与 map-anything fps_memory.sh 一致：稀疏深度在网格邻域内取有效深度中位数
    PATCH_DEPTH_SAMPLING="${PATCH_DEPTH_SAMPLING:-nearest_valid}"
else
    DEPTH_MIN="${DEPTH_MIN:-0.1}"
    DEPTH_MAX="${DEPTH_MAX:-6.0}"
    PATCH_DEPTH_SAMPLING="${PATCH_DEPTH_SAMPLING:-nearest}"
fi

# 特征提取方式
USE_PATCH_BASED="${USE_PATCH_BASED:-false}"
USE_L2_NORMALIZATION="${USE_L2_NORMALIZATION:-true}"

# Ray pooling策略
RAY_POOL_STRATEGY="${RAY_POOL_STRATEGY:-mean}"

# DINOv2 权重
DINOV2_CHECKPOINT="${DINOV2_CHECKPOINT:-/data/xwh/checkpoints/dinov2_vitl14_pretrain.pth}"

# 特征提取模型选择
# use_model: mapanything (默认，与 fps_memory.sh 对齐) | dinov2 (DPT-style 多尺度)
USE_MODEL="${USE_MODEL:-mapanything}"

# MapAnything 模型配置（use_model=mapanything 时使用）
MODEL_STR="${MODEL_STR:-mapanything_store_intermediates_ace}"
MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-/data/xwh/checkpoints/facebook_map-anything.pth}"

# DINOv2 多尺度配置（use_model=dinov2 时使用）
# 中间层 block 索引，默认 DPT-style 8 层；留空使用默认值
DINOV2_INTERMEDIATE_LAYERS="${DINOV2_INTERMEDIATE_LAYERS:-}"

# =============================================================================
# 输出路径
# =============================================================================
# 结构: <OUTPUT_ROOT>/<scene>/<Nviews>_<config_tag>/<timestamp>/
#   config_tag 编码关键配置：voxel size, patch/bilinear, sor
# 例: memory_extraction/04_evaluation/memory_extract/chess/20v_v0.05_bilinear/<timestamp>/
# 默认根目录为本脚本所在目录 memory_extraction/ 下的 04_evaluation/memory_extract

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
# 构建 Python 参数
# =============================================================================
PYTHON_ARGS=""
PYTHON_ARGS="$PYTHON_ARGS $DATASET_PATH"
PYTHON_ARGS="$PYTHON_ARGS $OUTPUT_FILE"
PYTHON_ARGS="$PYTHON_ARGS --n_memory $N_VIEWS"
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
