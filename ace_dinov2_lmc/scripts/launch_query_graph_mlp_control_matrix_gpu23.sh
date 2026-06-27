#!/usr/bin/env bash
set -euo pipefail

# First query-graph infrastructure matrix.
# Scope: Wayspots ACE-FCN-LMC stage1 only. No Cambridge stage2 / GLACE stage2.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${RUN_ROOT:-/data/xwh/ace_dinov2_lmc/04_evaluation/query_graph_mlp_control_${STAMP}_gpu23}"
SCENES="${SCENES:-wayspots_bears wayspots_squarebench wayspots_cubes wayspots_tendrils}"

COMMON_ENV=(
  CONDA_ENV=mapanything
  WAYSPOTS_ROOT=/data/xwh/Wayspots
  RUN_ROOT="${RUN_ROOT}"
  SCENES="${SCENES}"
  GPU_0=2
  GPU_1=3
  METHODS=stage1
  LMC_ITERATIONS=6
  LMC_FUSION_REFINEMENT_MODE=single
  S1_LOSS_STEP_MODE=fixed_zero
  S1_EARLY_STOP=False
  LMC_LOG_RUNTIME_STATS=True
  LMC_RUNTIME_STATS_INTERVAL=50
  LMC_RUNTIME_STATS_MAX_PIXELS=4096
  NUM_DATA_LOADER_WORKERS=8
  EVAL_NUM_WORKERS=4
  EVAL_DETERMINISTIC=True
  EVAL_DSACSTAR_SEED=1305
  EVAL_DSACSTAR_SEED_PER_FRAME=True
  ITERATION_EVAL_SEED=1305
  ITERATION_EVAL_HYPOTHESES=256
  POST_TRAIN_EVAL_SEEDS="1305 2026 4242"
  POST_TRAIN_HYPOTHESES=256
  IMAGE_RESOLUTION=512
  BATCH_SIZE=4096
  TRAINING_BUFFER_SIZE=800000
  BUFFER_SIZE_FINAL=1600000
  SAMPLES_PER_IMAGE=512
  SKIP_EXISTING=true
  CONTINUE_ON_ERROR=true
)

echo "[query-graph-matrix] run root: ${RUN_ROOT}"
echo "[query-graph-matrix] scenes  : ${SCENES}"

# Build memory once inside the shared run root.
env "${COMMON_ENV[@]}" \
  METHODS="memory" \
  bash ace_dinov2_lmc/scripts/run_wayspots_ace_fcn_lmc_suite.sh

# Strong single baseline under the same short schedule/buffer.
env "${COMMON_ENV[@]}" \
  STAGE1_SUBDIR=stage1_single_base_it6 \
  LMC_QUERY_GRAPH_REFINE_MODE=none \
  bash ace_dinov2_lmc/scripts/run_wayspots_ace_fcn_lmc_suite.sh

# Query-graph infrastructure path: local-window grouped sampler + per-node MLP control.
env "${COMMON_ENV[@]}" \
  STAGE1_SUBDIR=stage1_qgraph_mlp_local_it6 \
  LMC_QUERY_GRAPH_REFINE_MODE=mlp_control \
  LMC_QUERY_GRAPH_IMAGES_PER_BATCH=4 \
  LMC_QUERY_GRAPH_PATCHES_PER_IMAGE=128 \
  LMC_QUERY_GRAPH_SAMPLER=local_window \
  LMC_QUERY_GRAPH_LAYERSCALE_INIT=0.01 \
  LMC_QUERY_GRAPH_GATE_INIT=-4.0 \
  LMC_QUERY_GRAPH_RESIDUAL_L1_WEIGHT=0.0 \
  LMC_QUERY_GRAPH_FREEZE_BASE=True \
  bash ace_dinov2_lmc/scripts/run_wayspots_ace_fcn_lmc_suite.sh

# Same MLP control with stratified grouped sampler to separate sampler effects.
env "${COMMON_ENV[@]}" \
  STAGE1_SUBDIR=stage1_qgraph_mlp_stratified_it6 \
  LMC_QUERY_GRAPH_REFINE_MODE=mlp_control \
  LMC_QUERY_GRAPH_IMAGES_PER_BATCH=4 \
  LMC_QUERY_GRAPH_PATCHES_PER_IMAGE=128 \
  LMC_QUERY_GRAPH_SAMPLER=stratified \
  LMC_QUERY_GRAPH_LAYERSCALE_INIT=0.01 \
  LMC_QUERY_GRAPH_GATE_INIT=-4.0 \
  LMC_QUERY_GRAPH_RESIDUAL_L1_WEIGHT=0.0 \
  LMC_QUERY_GRAPH_FREEZE_BASE=True \
  bash ace_dinov2_lmc/scripts/run_wayspots_ace_fcn_lmc_suite.sh

echo "[query-graph-matrix] done: ${RUN_ROOT}"
