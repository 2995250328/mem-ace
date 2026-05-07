#!/bin/bash
# Batch extract Indoor6 reference-consistent memories.
#
# Policy:
# - clustered extraction: scene1, scene2a, scene4a, scene5, scene6
# - non-cluster extraction: scene1, scene4a, scene5, scene6
# - scene3: intentionally skipped
#
# GPU scheduling:
# - GPU 0: non-cluster scenes
# - GPU 1: clustered scenes

set -euo pipefail

ROOT_DIR="/home/xwh/project/ace_depth"
EXTRACT_DIR="${ROOT_DIR}/ace_dinov2_lmc/memory_extraction"
OUT_ROOT="${EXTRACT_DIR}/04_evaluation/memory_extract"
DATA_ROOT="${ACE_DATA_ROOT:-/home/xwh/data}"
DATASET_ROOT="${DATA_ROOT}/mapanything-dataset/wai_data/indoor6"

SCENE_BASE_ARGS=(
  ACE_DATA_ROOT="$DATA_ROOT"
  DATASET_ROOT="$DATASET_ROOT"
  DATASET_TYPE=indoor6
  N_VIEWS=40
  OUTPUT_ROOT="$OUT_ROOT"
  CONTRACT_MODE=C1
  WAI_VIEW_MODE=anchor_support
  WAI_TRANSFORM=imgnorm
  WAI_AUG_CROP=0
  POOL_MODE=bse
  VOXEL_SIZE=0.05
  USE_OTSU=true
  UNIMODAL_THRESHOLD=0.02
  PREPOOL_MODE=per_view
  ENABLE_SOR=true
  GLOBAL_MERGE=true
  USE_L2_NORMALIZATION=true
  ASB_ADAPTIVE=true
  ASB_CANDIDATE_POOL_RATIO=1.0
  ENABLE_REFERENCE_POLICY_GATE=true
  REFERENCE_POLICY_TOP_M=4
  REFERENCE_POLICY_LIGHT_MIN_SELECTED_LINKS=2
  REFERENCE_POLICY_PROBE_Q90_M=0.18
  REFERENCE_POLICY_PROBE_MAX_M=0.25
  ASB_POST_REPAIR=true
  ASB_POST_REPAIR_MAX_SWAPS=8
  ASB_POST_REPAIR_TAIL_PERCENTILE=95.0
  ASB_POST_REPAIR_MIN_TAIL_IMPROVEMENT_M=0.02
  ASB_POST_REPAIR_CLUSTER_TOP_K=3
)

run_extract() {
  local gpu_id="$1"
  local scene="$2"
  local enable_cluster_fallback="$3"
  local log_dir="$4"

  local -a env_args=(
    GPU_ID="$gpu_id"
    SCENE_TRAIN="${scene}_train"
    ENABLE_CLUSTER_FALLBACK="$enable_cluster_fallback"
  )

  mkdir -p "$log_dir"
  echo "[start] scene=${scene} gpu=${gpu_id} cluster_fallback=${enable_cluster_fallback}"
  env "${SCENE_BASE_ARGS[@]}" "${env_args[@]}" \
    bash "${EXTRACT_DIR}/extract_memory.sh" \
    > "${log_dir}/${scene}.log" 2>&1
  echo "[done] scene=${scene}"
}

trap 'jobs -pr | xargs -r kill' EXIT

LOG_ROOT="${EXTRACT_DIR}/batch_logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_ROOT"

# Clustered scenes on GPU 1.
(
  for scene in scene1 scene2a scene4a scene5 scene6; do
    run_extract 1 "$scene" true "$LOG_ROOT"
  done
) &
PID_CLUSTER=$!

# Non-cluster scenes on GPU 0.
(
  for scene in scene1 scene4a scene5 scene6; do
    run_extract 0 "$scene" false "$LOG_ROOT"
  done
) &
PID_NON_CLUSTER=$!

wait "$PID_CLUSTER"
wait "$PID_NON_CLUSTER"

echo
echo "============================================================"
echo "Result summary"
echo "============================================================"

find "$OUT_ROOT" \
  \( -path '*/scene1/*' -o -path '*/scene2a/*' -o -path '*/scene4a/*' -o -path '*/scene5/*' -o -path '*/scene6/*' \) \
  \( -name 'memory_bse.pt' -o -name 'memory_bse.clustered.pt' -o -name 'memory_bse.cluster_*.pt' -o -name 'cluster_fallback_plan.json' -o -name 'memory_policy_report.json' -o -name 'extraction_config.json' \) \
  | sort
