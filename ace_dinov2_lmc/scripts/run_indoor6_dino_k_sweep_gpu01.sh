#!/usr/bin/env bash
set -euo pipefail

# Indoor6 DINOv2+MapAnything-memory K sweep wrapper using GPUs 0/1.
# This reuses run_indoor6_dino_forceglobal_s1_periter_remaining_gpu01.sh,
# which now accepts NUM_LATENT_TOKENS, LMC_AUTO_MODE_BY_VISIBILITY, and S1_LOSS_STEP_MODE.
# Run from /home/xwh/project/ace_depth:
#   bash ace_dinov2_lmc/scripts/run_indoor6_dino_k_sweep_gpu01.sh
# Useful:
#   DRY_RUN=true KS_STR="256 1024" SCENES_STR="scene2a scene5 scene6" bash ...

ROOT_DIR="${ROOT_DIR:-/home/xwh/project/ace_depth}"
SCRIPT_DIR="${ROOT_DIR}/ace_dinov2_lmc/scripts"
KS_STR="${KS_STR:-256 1024}"
SCENES_STR="${SCENES_STR:-scene2a scene5 scene6}"
GPUS_STR="${GPUS_STR:-0 1}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${ROOT_DIR}/ace_dinov2_lmc/04_evaluation/train_compare}"
BASE_EXPERIMENT_SUBDIR="${BASE_EXPERIMENT_SUBDIR:-indoor6_dino_mapanything_k_sweep}"
DRY_RUN="${DRY_RUN:-false}"
LMC_AUTO_MODE_BY_VISIBILITY="${LMC_AUTO_MODE_BY_VISIBILITY:-true}"
S1_LOSS_STEP_MODE="${S1_LOSS_STEP_MODE:-per_iter}"
EVAL_EACH_ITERATION="${EVAL_EACH_ITERATION:-true}"
EVAL_AFTER_TRAIN="${EVAL_AFTER_TRAIN:-true}"

read -r -a KS <<< "${KS_STR}"
read -r -a GPUS <<< "${GPUS_STR}"
for gpu in "${GPUS[@]}"; do
  if [[ "${gpu}" != "0" && "${gpu}" != "1" ]]; then
    echo "ERROR: this script is constrained to GPUs 0/1, got ${gpu}" >&2
    exit 2
  fi
done

printf 'Scenes   : %s\n' "${SCENES_STR}"
printf 'K values : %s\n' "${KS_STR}"
printf 'GPUs     : %s\n' "${GPUS_STR}"
printf 'Route    : LMC_AUTO_MODE_BY_VISIBILITY=%s S1_LOSS_STEP_MODE=%s\n' "${LMC_AUTO_MODE_BY_VISIBILITY}" "${S1_LOSS_STEP_MODE}"
printf 'Dry run  : %s\n' "${DRY_RUN}"

for k in "${KS[@]}"; do
  subdir="${BASE_EXPERIMENT_SUBDIR}_K${k}"
  printf '\n============================================================\n'
  printf '[K sweep] K=%s subdir=%s\n' "${k}" "${subdir}"
  printf '============================================================\n'
  env \
    ROOT_DIR="${ROOT_DIR}" \
    SCENES_STR="${SCENES_STR}" \
    GPUS_STR="${GPUS_STR}" \
    EXPERIMENT_ROOT="${EXPERIMENT_ROOT}" \
    EXPERIMENT_SUBDIR="${subdir}" \
    NUM_LATENT_TOKENS="${k}" \
    LMC_AUTO_MODE_BY_VISIBILITY="${LMC_AUTO_MODE_BY_VISIBILITY}" \
    S1_LOSS_STEP_MODE="${S1_LOSS_STEP_MODE}" \
    DRY_RUN="${DRY_RUN}" \
    EVAL_EACH_ITERATION="${EVAL_EACH_ITERATION}" \
    EVAL_AFTER_TRAIN="${EVAL_AFTER_TRAIN}" \
    bash "${SCRIPT_DIR}/run_indoor6_dino_forceglobal_s1_periter_remaining_gpu01.sh"
done
