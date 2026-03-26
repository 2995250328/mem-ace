#!/bin/bash
# BSE-enhanced memory extraction launcher
# Adapted from map-anything/bash_scripts/ace/fps_memory.sh

DATASET=${1:-"datasets/7scenes_chess"}
OUTPUT=${2:-"output/memory.pt"}
N_MEMORY=${3:-100}
GPU_ID=${4:-0}

echo "Dataset: $DATASET"
echo "Output: $OUTPUT"
echo "N_MEMORY: $N_MEMORY"
echo "GPU: $GPU_ID"

CUDA_VISIBLE_DEVICES=$GPU_ID python -m ace_dinov2_lmc.memory_extraction.run_memory_extraction \
    "$DATASET" \
    "$OUTPUT" \
    --n_memory $N_MEMORY \
    --use_bse \
    --use_otsu \
    --device cuda:0 \
    --model_config configs/mapanything_default.yaml

echo "Done. Memory saved to $OUTPUT"
