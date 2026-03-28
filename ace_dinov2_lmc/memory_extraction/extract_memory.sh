#!/bin/bash
# BSE-enhanced memory extraction launcher
# Adapted from map-anything/bash_scripts/ace/fps_memory.sh
#
# Usage:
#   ./extract_memory.sh [DATASET] [OUTPUT] [N_MEMORY] [GPU_ID]
#
# Examples:
#   ./extract_memory.sh datasets/7scenes_chess output/memory.pt 100 0
#   ./extract_memory.sh datasets/indoor6_scene3 output/indoor6.pt 50 1

# Default values
DATASET=${1:-"datasets/7scenes_chess"}
OUTPUT=${2:-"output/memory.pt"}
N_MEMORY=${3:-100}
GPU_ID=${4:-0}

# BSE configuration
VOXEL_SIZE=${VOXEL_SIZE:-0.05}
USE_OTSU=${USE_OTSU:-true}
ENABLE_SOR=${ENABLE_SOR:-false}

# Depth configuration
DEPTH_MIN=${DEPTH_MIN:-0.1}
DEPTH_MAX=${DEPTH_MAX:-6.0}

# Model configuration (optional - uses DINOv2 standalone if not specified)
MODEL_STR=${MODEL_STR:-""}
MODEL_CONFIG=${MODEL_CONFIG:-""}
MODEL_CHECKPOINT=${MODEL_CHECKPOINT:-""}
DINOV2_CHECKPOINT=${DINOV2_CHECKPOINT:-"/data/xwh/checkpoints/dinov2_vitl14_pretrain.pth"}

echo "============================================"
echo "BSE Memory Extraction"
echo "============================================"
echo "Dataset: $DATASET"
echo "Output: $OUTPUT"
echo "N_MEMORY: $N_MEMORY"
echo "GPU: $GPU_ID"
echo "Voxel size: $VOXEL_SIZE"
echo "Use Otsu: $USE_OTSU"
echo "Enable SOR: $ENABLE_SOR"
echo "Depth range: [$DEPTH_MIN, $DEPTH_MAX]"
echo "============================================"

# Build optional arguments
OPTIONAL_ARGS=""

if [ -n "$MODEL_STR" ]; then
    OPTIONAL_ARGS="$OPTIONAL_ARGS --model_str $MODEL_STR"
fi

if [ -n "$MODEL_CONFIG" ]; then
    OPTIONAL_ARGS="$OPTIONAL_ARGS --model_config $MODEL_CONFIG"
fi

if [ -n "$MODEL_CHECKPOINT" ]; then
    OPTIONAL_ARGS="$OPTIONAL_ARGS --model_checkpoint $MODEL_CHECKPOINT"
fi

if [ "$ENABLE_SOR" = "true" ]; then
    OPTIONAL_ARGS="$OPTIONAL_ARGS --enable_sor"
fi

# Run extraction
CUDA_VISIBLE_DEVICES=$GPU_ID python -m ace_dinov2_lmc.memory_extraction.run_memory_extraction \
    "$DATASET" \
    "$OUTPUT" \
    --n_memory $N_MEMORY \
    --device cuda:0 \
    --voxel_size $VOXEL_SIZE \
    --depth_valid_range $DEPTH_MIN $DEPTH_MAX \
    --dinov2_checkpoint $DINOV2_CHECKPOINT \
    $OPTIONAL_ARGS

# Check result
if [ $? -eq 0 ]; then
    echo "============================================"
    echo "Success! Memory saved to $OUTPUT"
    echo "============================================"

    # Print file info
    if [ -f "$OUTPUT" ]; then
        FILE_SIZE=$(du -h "$OUTPUT" | cut -f1)
        echo "File size: $FILE_SIZE"
    fi
else
    echo "============================================"
    echo "Error: Extraction failed"
    echo "============================================"
    exit 1
fi
