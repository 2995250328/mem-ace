#!/bin/bash
# Ablation study: Multi-modal features
# Tests different combinations of depth, SuperPoint, and intrinsics

SCENE=$1
OUTPUT_DIR=$2
DEVICE=${3:-cuda:0}

if [ -z "$SCENE" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: $0 <scene_path> <output_dir> [device]"
    echo "Example: $0 datasets/7scenes_chess output/ablation cuda:0"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

SCENE_NAME=$(basename "$SCENE")

echo "=========================================="
echo "Multi-Modal Feature Ablation Study"
echo "Scene: $SCENE"
echo "Output: $OUTPUT_DIR"
echo "Device: $DEVICE"
echo "=========================================="

# Define configurations
declare -a configs=(
    "rgb:False:False:False"
    "rgb_depth:True:False:False"
    "rgb_sp:False:True:False"
    "rgb_intr:False:False:True"
    "rgb_depth_sp:True:True:False"
    "rgb_depth_intr:True:False:True"
    "rgb_sp_intr:False:True:True"
    "full:True:True:True"
)

# Train all configurations
for config in "${configs[@]}"; do
    IFS=':' read -r name depth sp intr <<< "$config"
    
    echo ""
    echo "Training configuration: $name"
    echo "  Depth: $depth, SuperPoint: $sp, Intrinsics: $intr"
    
    ./train_ace_dinov2_lmc.py \
        "$SCENE" \
        "$OUTPUT_DIR/${SCENE_NAME}_${name}.pt" \
        --use_depth "$depth" \
        --use_superpoint "$sp" \
        --use_intrinsics "$intr" \
        --enable_compression True \
        --compression_ratio 0.5 \
        --device "$DEVICE" \
        --epochs 16 \
        --batch_size 3840
done

# Test all configurations
echo ""
echo "=========================================="
echo "Testing all configurations"
echo "=========================================="

for config in "${configs[@]}"; do
    IFS=':' read -r name depth sp intr <<< "$config"
    
    echo ""
    echo "Testing configuration: $name"
    
    ./test_ace_dinov2_lmc.py \
        "$SCENE" \
        "$OUTPUT_DIR/${SCENE_NAME}_${name}.pt" \
        --session "$name" \
        --device "$DEVICE"
done

echo ""
echo "=========================================="
echo "Multi-modal ablation study completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="
