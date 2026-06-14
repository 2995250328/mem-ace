#!/usr/bin/env bash
# Diagnostic pair run:
# - scene2a on GPU0
# - scene5  on GPU1
#
# Both runs use force-global ACE-G but keep the old/legacy S1 loss-step behavior
# by intentionally not passing --s1_loss_step_mode per_iter.
#
# Run:
#   cd /home/xwh/project/ace_depth
#   conda activate mapanything
#   bash ace_dinov2_lmc/scripts/run_indoor6_scene2a_scene5_dino_forceglobal_legacy_s1_gpu01.sh

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/xwh/project/ace_depth}"
REPO_ROOT="${REPO_ROOT:-${ROOT_DIR}/ace_dinov2_lmc}"
RUN_ONE="${REPO_ROOT}/scripts/run_indoor6_scene5_dino_forceglobal_legacy_s1_gpu.sh"

EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-/data/xwh/ace_dinov2_lmc/04_evaluation/train_compare}"
EXPERIMENT_SUBDIR="${EXPERIMENT_SUBDIR:-indoor6_full_baselines_4090_forceglobal_legacy_s1_scene2a_scene5}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/04_evaluation/train_compare/run_logs/forceglobal_legacy_s1_scene2a_scene5_$(date +%Y%m%d_%H%M%S)}"
DRY_RUN="${DRY_RUN:-false}"

SCENE2A_MEMORY="${SCENE2A_MEMORY:-${REPO_ROOT}/memory_extraction/04_evaluation/memory_extract/scene2a/40v_v0.05_bilinear_bse_ut0.02_noaug_asb_adaptive_pool1.0_rgate4_prepair8_sor_gm_l2/20260508_103423/memory_bse.pt}"
SCENE5_MEMORY="${SCENE5_MEMORY:-${REPO_ROOT}/memory_extraction/04_evaluation/memory_extract/scene5/40v_v0.05_bilinear_bse_ut0.02_noaug_asb_adaptive_pool1.0_rgate4_prepair8_sor_gm_l2/20260508_105011/memory_bse.pt}"

mkdir -p "${RUN_ROOT}"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

run_scene() {
  local scene="$1"
  local gpu="$2"
  local memory="$3"
  local run_root="${RUN_ROOT}/${scene}_gpu${gpu}"

  log "[start] ${scene} on GPU${gpu}"
  SCENE="$scene" \
  GPU_ID="$gpu" \
  MEMORY_PATH="$memory" \
  EXPERIMENT_ROOT="$EXPERIMENT_ROOT" \
  EXPERIMENT_SUBDIR="$EXPERIMENT_SUBDIR" \
  RUN_ROOT="$run_root" \
  DRY_RUN="$DRY_RUN" \
  bash "$RUN_ONE"
  log "[done] ${scene} on GPU${gpu}"
}

main() {
  if [ ! -x "$RUN_ONE" ]; then
    echo "ERROR: missing executable helper script: ${RUN_ONE}" >&2
    exit 1
  fi

  log "Experiment root   : ${EXPERIMENT_ROOT}"
  log "Experiment subdir : ${EXPERIMENT_SUBDIR}"
  log "Run root          : ${RUN_ROOT}"
  log "Scene2a memory    : ${SCENE2A_MEMORY}"
  log "Scene5 memory     : ${SCENE5_MEMORY}"
  log "Dry run           : ${DRY_RUN}"

  run_scene scene2a 0 "$SCENE2A_MEMORY" &
  local pid_scene2a="$!"
  run_scene scene5 1 "$SCENE5_MEMORY" &
  local pid_scene5="$!"

  local failed=0
  if ! wait "$pid_scene2a"; then
    log "[failed] scene2a on GPU0"
    failed=1
  fi
  if ! wait "$pid_scene5"; then
    log "[failed] scene5 on GPU1"
    failed=1
  fi

  if [ "$failed" -ne 0 ]; then
    exit "$failed"
  fi
  log "Done. Logs under: ${RUN_ROOT}"
}

main "$@"
