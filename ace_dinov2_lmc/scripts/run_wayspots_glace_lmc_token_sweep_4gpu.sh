#!/usr/bin/env bash
set -euo pipefail

# Four-GPU Wayspots GLACE+LMC token-count sweep.
#
# Default plan:
#   scenes: all Wayspots scenes with existing memory in the 20260530 suite
#   K:      256 1024
#   route:  Stage1 local ACE-FCN-LMC -> Stage2 GLACE concat
#
# Run from /home/xwh/project/ace_depth:
#   conda activate mapanything
#   bash ace_dinov2_lmc/scripts/run_wayspots_glace_lmc_token_sweep_4gpu.sh
#
# Optional:
#   DRY_RUN=true bash ace_dinov2_lmc/scripts/run_wayspots_glace_lmc_token_sweep_4gpu.sh
#   K_VALUES="256 1024 2048" bash ace_dinov2_lmc/scripts/run_wayspots_glace_lmc_token_sweep_4gpu.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

SUITE_ROOT="${SUITE_ROOT:-/home/xwh/project/ace_depth/ace_dinov2_lmc/04_evaluation/wayspots_ace_fcn_lmc_suite/20260530_181745}"
RUN_SCRIPT="${RUN_SCRIPT:-${ROOT_DIR}/ace_dinov2_lmc/scripts/run_wayspots_ace_fcn_lmc_suite.sh}"
CONDA_ENV="${CONDA_ENV:-mapanything}"
DRY_RUN="${DRY_RUN:-false}"
SKIP_EXISTING="${SKIP_EXISTING:-true}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-true}"

GPUS=(${GPUS:-0 1 2 3})
K_VALUES=(${K_VALUES:-256 1024})
SCENES=(${SCENES:-wayspots_bears wayspots_cubes wayspots_inscription wayspots_lawn wayspots_map wayspots_squarebench wayspots_tendrils wayspots_therock})

LMC_ITERATIONS="${LMC_ITERATIONS:-12}"
METHODS="${METHODS:-stage1 stage2}"

if [[ ${#GPUS[@]} -ne 4 ]]; then
  echo "ERROR: this scheduler expects exactly 4 GPUs, got: ${GPUS[*]}" >&2
  exit 2
fi

for scene in "${SCENES[@]}"; do
  memory_path="${SUITE_ROOT}/memory/${scene}/memory_ace_fcn_sparse_sp_r4.pt"
  if [[ ! -s "${memory_path}" ]]; then
    echo "ERROR: missing memory for ${scene}: ${memory_path}" >&2
    exit 2
  fi
done

LOG_ROOT="${LOG_ROOT:-${SUITE_ROOT}/token_sweep_logs_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${LOG_ROOT}"

run_one() {
  local gpu="$1"
  local scene="$2"
  local k="$3"
  local stage1_subdir="stage1_local_ace_memory_it${LMC_ITERATIONS}_K${k}"
  local stage2_subdir="stage2_glace_concat_it${LMC_ITERATIONS}_K${k}"
  local log_file="${LOG_ROOT}/gpu${gpu}_${scene}_K${k}.log"

  {
    printf "============================================================\n"
    printf "[token-sweep] scene=%s K=%s GPU=%s\n" "${scene}" "${k}" "${gpu}"
    printf "suite_root : %s\n" "${SUITE_ROOT}"
    printf "stage1     : %s\n" "${stage1_subdir}"
    printf "stage2     : %s\n" "${stage2_subdir}"
    printf "dry_run    : %s\n" "${DRY_RUN}"
    printf "============================================================\n"
  } | tee -a "${log_file}"

  env \
    CONDA_ENV="${CONDA_ENV}" \
    RUN_ROOT="${SUITE_ROOT}" \
    SCENES="${scene}" \
    METHODS="${METHODS}" \
    GPU_0="${gpu}" \
    GPU_1="${gpu}" \
    NUM_LATENT_TOKENS="${k}" \
    LMC_ITERATIONS="${LMC_ITERATIONS}" \
    STAGE1_SUBDIR="${stage1_subdir}" \
    STAGE2_SUBDIR="${stage2_subdir}" \
    SKIP_EXISTING="${SKIP_EXISTING}" \
    CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR}" \
    DRY_RUN="${DRY_RUN}" \
    bash "${RUN_SCRIPT}" 2>&1 | tee -a "${log_file}"
}

worker() {
  local gpu="$1"
  shift
  local item scene k
  for item in "$@"; do
    scene="${item%%:*}"
    k="${item##*:}"
    run_one "${gpu}" "${scene}" "${k}"
  done
}

queues=( "" "" "" "" )
idx=0

# Prioritize diagnostic pairs first, then fill the rest round-robin.
priority_scenes=(wayspots_squarebench wayspots_bears)
for scene in "${priority_scenes[@]}"; do
  scene_found=false
  for configured_scene in "${SCENES[@]}"; do
    [[ "${configured_scene}" == "${scene}" ]] && scene_found=true
  done
  [[ "${scene_found}" == "true" ]] || continue
  for k in "${K_VALUES[@]}"; do
    q=$((idx % 4))
    queues[${q}]="${queues[${q}]} ${scene}:${k}"
    idx=$((idx + 1))
  done
done

for scene in "${SCENES[@]}"; do
  [[ "${scene}" == "wayspots_squarebench" || "${scene}" == "wayspots_bears" ]] && continue
  for k in "${K_VALUES[@]}"; do
    q=$((idx % 4))
    queues[${q}]="${queues[${q}]} ${scene}:${k}"
    idx=$((idx + 1))
  done
done

printf "Wayspots token sweep\n"
printf "Suite root : %s\n" "${SUITE_ROOT}"
printf "Scenes     : %s\n" "${SCENES[*]}"
printf "K values   : %s\n" "${K_VALUES[*]}"
printf "GPUs       : %s\n" "${GPUS[*]}"
printf "Methods    : %s\n" "${METHODS}"
printf "Logs       : %s\n" "${LOG_ROOT}"
printf "Dry run    : %s\n" "${DRY_RUN}"
for i in 0 1 2 3; do
  printf "GPU %s queue:%s\n" "${GPUS[$i]}" "${queues[$i]}"
done

pids=()
for i in 0 1 2 3; do
  read -r -a items <<< "${queues[$i]}"
  worker "${GPUS[$i]}" "${items[@]}" &
  pids+=("$!")
done

exit_code=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    exit_code=1
  fi
done

printf "Done. token sweep logs: %s\n" "${LOG_ROOT}"
exit "${exit_code}"
