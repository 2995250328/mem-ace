#!/usr/bin/env bash
set -euo pipefail

# Run official ACE-G25 baseline on Wayspots using only physical GPUs 0/1.
# Run from /home/xwh/project/ace_depth:
#   bash ace_dinov2_lmc/scripts/run_wayspots_aceg25_baseline_gpu01.sh
# Useful:
#   DRY_RUN=true bash ace_dinov2_lmc/scripts/run_wayspots_aceg25_baseline_gpu01.sh
#   SCENES="wayspots_bears wayspots_cubes" bash ...

ROOT_DIR="${ROOT_DIR:-/home/xwh/project/ace_depth}"
SCRIPT_DIR="${ROOT_DIR}/ace_dinov2_lmc/scripts"
CONDA_ENV="${CONDA_ENV:-mapanything}"
WAYSPOTS_ROOT="${WAYSPOTS_ROOT:-/data/xwh/Wayspots}"
RUN_ROOT="${RUN_ROOT:-/data/xwh/ace_dinov2_lmc/04_evaluation/wayspots_aceg25_baseline_gpu01/$(date +%Y%m%d_%H%M%S)}"
SCENES="${SCENES:-wayspots_bears wayspots_cubes wayspots_inscription wayspots_lawn wayspots_map wayspots_squarebench wayspots_statue wayspots_tendrils wayspots_therock wayspots_wintersign}"
GPUS_STR="${GPUS_STR:-0 1}"
DRY_RUN="${DRY_RUN:-false}"
SKIP_EXISTING="${SKIP_EXISTING:-true}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-true}"
HYPOTHESES="${HYPOTHESES:-256}"

ACE_G_ROOT="${ACE_G_ROOT:-/home/xwh/project/ace-g}"
ACEG25_CONFIG="${ACEG25_CONFIG:-${ACE_G_ROOT}/src/ace_g/configs/ace_g_25min.yaml}"
ACEG25_NUM_ITERATIONS="${ACEG25_NUM_ITERATIONS:-4000}"
ACEG25_MAX_DATASET_PASSES="${ACEG25_MAX_DATASET_PASSES:-40}"
ACEG25_MAX_BUFFER_SIZE="${ACEG25_MAX_BUFFER_SIZE:-6000000}"
ACEG25_BATCH_SIZE="${ACEG25_BATCH_SIZE:-32768}"
ACEG25_SAMPLES_PER_IMAGE="${ACEG25_SAMPLES_PER_IMAGE:-1024}"
ACEG25_BUFFER_DEVICE="${ACEG25_BUFFER_DEVICE:-}"
ACEG25_BUFFER_CREATION_BATCH_SIZE="${ACEG25_BUFFER_CREATION_BATCH_SIZE:-4}"
ACEG25_BUFFER_CREATION_WORKERS="${ACEG25_BUFFER_CREATION_WORKERS:-8}"

read -r -a GPUS <<< "${GPUS_STR}"
read -r -a SCENE_ARR <<< "${SCENES}"
for gpu in "${GPUS[@]}"; do
  if [[ "${gpu}" != "0" && "${gpu}" != "1" ]]; then
    echo "ERROR: this script is constrained to GPUs 0/1, got ${gpu}" >&2
    exit 2
  fi
done

mkdir -p "${RUN_ROOT}"
printf 'Run root : %s\n' "${RUN_ROOT}"
printf 'Scenes   : %s\n' "${SCENES}"
printf 'GPUs     : %s\n' "${GPUS_STR}"
printf 'ACE-G   : %s\n' "${ACEG25_CONFIG}"
printf 'Dry run : %s\n' "${DRY_RUN}"

run_group() {
  local gpu="$1"; shift
  local scenes="$*"
  [[ -n "${scenes}" ]] || return 0
  local log_dir="${RUN_ROOT}/_orchestrator"
  mkdir -p "${log_dir}"
  local log_file="${log_dir}/aceg25_gpu${gpu}.log"
  env \
    CONDA_ENV="${CONDA_ENV}" \
    WAYSPOTS_ROOT="${WAYSPOTS_ROOT}" \
    RUN_ROOT="${RUN_ROOT}" \
    SCENES="${scenes}" \
    METHODS="aceg25" \
    GPU_ID="${gpu}" \
    DEVICE="cuda:${gpu}" \
    EVAL_DEVICE="cuda:${gpu}" \
    HYPOTHESES="${HYPOTHESES}" \
    DRY_RUN="${DRY_RUN}" \
    SKIP_EXISTING="${SKIP_EXISTING}" \
    CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR}" \
    ACE_G_ROOT="${ACE_G_ROOT}" \
    ACEG25_CONFIG="${ACEG25_CONFIG}" \
    ACEG25_NUM_ITERATIONS="${ACEG25_NUM_ITERATIONS}" \
    ACEG25_MAX_DATASET_PASSES="${ACEG25_MAX_DATASET_PASSES}" \
    ACEG25_MAX_BUFFER_SIZE="${ACEG25_MAX_BUFFER_SIZE}" \
    ACEG25_BATCH_SIZE="${ACEG25_BATCH_SIZE}" \
    ACEG25_SAMPLES_PER_IMAGE="${ACEG25_SAMPLES_PER_IMAGE}" \
    ACEG25_BUFFER_DEVICE="${ACEG25_BUFFER_DEVICE}" \
    ACEG25_BUFFER_CREATION_BATCH_SIZE="${ACEG25_BUFFER_CREATION_BATCH_SIZE}" \
    ACEG25_BUFFER_CREATION_WORKERS="${ACEG25_BUFFER_CREATION_WORKERS}" \
    bash "${SCRIPT_DIR}/run_wayspots_baselines_all.sh" 2>&1 | tee "${log_file}"
}

queues=("" "")
for i in "${!SCENE_ARR[@]}"; do
  idx=$(( i % ${#GPUS[@]} ))
  queues[$idx]="${queues[$idx]} ${SCENE_ARR[$i]}"
done

pids=()
for i in "${!GPUS[@]}"; do
  run_group "${GPUS[$i]}" ${queues[$i]} &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  wait "${pid}" || failed=1
done

printf 'Done. Status: %s/status.tsv\n' "${RUN_ROOT}"
exit "${failed}"
