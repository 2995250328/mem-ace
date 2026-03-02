#!/bin/bash
# Ablation study: Compression ratio
# Tests different compression ratios: 0.25, 0.5, 0.75, 1.0 (no compression)

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
echo "Compression Ratio Ablation Study"
echo "Scene: $SCENE"
echo "Output: $OUTPUT_DIR"
echo "Device: $DEVICE"
echo "=========================================="

# Train with different compression ratios
for ratio in 0.25 0.5 0.75 1.0; do
    echo ""
    echo "Training with compression ratio: $ratio"
    ./train_ace_dinov2_lmc.py \
        "$SCENE" \
        "$OUTPUT_DIR/${SCENE_NAME}_compress_${ratio}.pt" \
        --compression_ratio "$ratio" \
        --enable_compression True \
        --device "$DEVICE" \
        --epochs 16 \
        --batch_size 3840
done

# Test all models
echo ""
echo "=========================================="
echo "Testing all models"
echo "=========================================="

for ratio in 0.25 0.5 0.75 1.0; do
    echo ""
    echo "Testing compression ratio: $ratio"
    ./test_ace_dinov2_lmc.py \
        "$SCENE" \
        "$OUTPUT_DIR/${SCENE_NAME}_compress_${ratio}.pt" \
        --session "compress_${ratio}" \
        --device "$DEVICE"
done

echo ""
echo "=========================================="
echo "Ablation study completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="
