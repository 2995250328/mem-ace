#!/bin/bash
# Ablation study: Compression on/off
# Compares performance with and without latent memory compression

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
echo "Compression On/Off Ablation Study"
echo "Scene: $SCENE"
echo "Output: $OUTPUT_DIR"
echo "Device: $DEVICE"
echo "=========================================="

# Train without compression
echo ""
echo "Training WITHOUT compression"
./train_ace_dinov2_lmc.py \
    "$SCENE" \
    "$OUTPUT_DIR/${SCENE_NAME}_no_compression.pt" \
    --enable_compression False \
    --device "$DEVICE" \
    --epochs 16 \
    --batch_size 3840

# Train with compression (ratio 0.5)
echo ""
echo "Training WITH compression (ratio=0.5)"
./train_ace_dinov2_lmc.py \
    "$SCENE" \
    "$OUTPUT_DIR/${SCENE_NAME}_with_compression.pt" \
    --enable_compression True \
    --compression_ratio 0.5 \
    --device "$DEVICE" \
    --epochs 16 \
    --batch_size 3840

# Test both models
echo ""
echo "=========================================="
echo "Testing both models"
echo "=========================================="

echo ""
echo "Testing WITHOUT compression"
./test_ace_dinov2_lmc.py \
    "$SCENE" \
    "$OUTPUT_DIR/${SCENE_NAME}_no_compression.pt" \
    --session "no_compression" \
    --device "$DEVICE"

echo ""
echo "Testing WITH compression"
./test_ace_dinov2_lmc.py \
    "$SCENE" \
    "$OUTPUT_DIR/${SCENE_NAME}_with_compression.pt" \
    --session "with_compression" \
    --device "$DEVICE"

echo ""
echo "=========================================="
echo "Compression on/off ablation completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="
