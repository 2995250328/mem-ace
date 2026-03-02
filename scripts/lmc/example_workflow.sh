#!/bin/bash
# Example workflow: Complete LMC training and testing pipeline

SCENE=$1
OUTPUT_DIR=$2
DEVICE=${3:-cuda:0}

if [ -z "$SCENE" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: $0 <scene_path> <output_dir> [device]"
    echo "Example: $0 datasets/7scenes_chess output/lmc_example cuda:0"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

SCENE_NAME=$(basename "$SCENE")

echo "=========================================="
echo "ACE DINOv2 LMC - Example Workflow"
echo "Scene: $SCENE"
echo "Output: $OUTPUT_DIR"
echo "Device: $DEVICE"
echo "=========================================="

# 1. Baseline: RGB-only with compression
echo ""
echo "Step 1: Training baseline (RGB-only with compression)"
./train_ace_dinov2_lmc.py \
    "$SCENE" \
    "$OUTPUT_DIR/${SCENE_NAME}_baseline.pt" \
    --enable_compression True \
    --compression_ratio 0.5 \
    --use_depth False \
    --use_superpoint False \
    --use_intrinsics False \
    --device "$DEVICE" \
    --epochs 16 \
    --batch_size 3840

# 2. Full model: All features enabled
echo ""
echo "Step 2: Training full model (all features)"
./train_ace_dinov2_lmc.py \
    "$SCENE" \
    "$OUTPUT_DIR/${SCENE_NAME}_full.pt" \
    --enable_compression True \
    --compression_ratio 0.5 \
    --use_depth True \
    --use_superpoint True \
    --use_intrinsics True \
    --device "$DEVICE" \
    --epochs 16 \
    --batch_size 3840

# 3. High compression model
echo ""
echo "Step 3: Training high compression model (ratio=0.25)"
./train_ace_dinov2_lmc.py \
    "$SCENE" \
    "$OUTPUT_DIR/${SCENE_NAME}_high_compress.pt" \
    --enable_compression True \
    --compression_ratio 0.25 \
    --use_depth False \
    --use_superpoint False \
    --use_intrinsics False \
    --device "$DEVICE" \
    --epochs 16 \
    --batch_size 3840

# Test all models
echo ""
echo "=========================================="
echo "Testing all models"
echo "=========================================="

echo ""
echo "Testing baseline model"
./test_ace_dinov2_lmc.py \
    "$SCENE" \
    "$OUTPUT_DIR/${SCENE_NAME}_baseline.pt" \
    --session "baseline" \
    --device "$DEVICE" \
    --hypotheses 64 \
    --threshold 10

echo ""
echo "Testing full model"
./test_ace_dinov2_lmc.py \
    "$SCENE" \
    "$OUTPUT_DIR/${SCENE_NAME}_full.pt" \
    --session "full" \
    --device "$DEVICE" \
    --hypotheses 64 \
    --threshold 10

echo ""
echo "Testing high compression model"
./test_ace_dinov2_lmc.py \
    "$SCENE" \
    "$OUTPUT_DIR/${SCENE_NAME}_high_compress.pt" \
    --session "high_compress" \
    --device "$DEVICE" \
    --hypotheses 64 \
    --threshold 10

# Summary
echo ""
echo "=========================================="
echo "Workflow completed!"
echo "=========================================="
echo ""
echo "Models trained:"
echo "  1. Baseline (RGB + compression 0.5): ${OUTPUT_DIR}/${SCENE_NAME}_baseline.pt"
echo "  2. Full (all features + compression 0.5): ${OUTPUT_DIR}/${SCENE_NAME}_full.pt"
echo "  3. High compression (RGB + compression 0.25): ${OUTPUT_DIR}/${SCENE_NAME}_high_compress.pt"
echo ""
echo "Pose files:"
echo "  1. ${SCENE}/poses_${SCENE_NAME}_baseline_baseline.txt"
echo "  2. ${SCENE}/poses_${SCENE_NAME}_full_full.txt"
echo "  3. ${SCENE}/poses_${SCENE_NAME}_high_compress_high_compress.txt"
echo ""
echo "To evaluate results, use:"
echo "  ./eval_poses.py $SCENE ${SCENE}/poses_${SCENE_NAME}_baseline_baseline.txt"
echo "=========================================="
