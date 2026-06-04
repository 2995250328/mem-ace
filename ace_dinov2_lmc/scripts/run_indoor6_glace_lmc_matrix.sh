#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

SCENES_STR="${SCENES_STR:-scene3 scene4a}"
GPUS_STR="${GPUS_STR:-0 1}"
MEMORY_MANIFEST="${MEMORY_MANIFEST:-}"
RUN_ROOT="${RUN_ROOT:-/data/xwh/ace_dinov2_lmc/04_evaluation/indoor6_glace_lmc}"
DRY_RUN="${DRY_RUN:-false}"
SKIP_MISSING_MEMORY="${SKIP_MISSING_MEMORY:-false}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-false}"

read -r -a SCENES <<< "${SCENES_STR}"
read -r -a GPUS <<< "${GPUS_STR}"

for gpu in "${GPUS[@]}"; do
  if [[ "${gpu}" != "0" && "${gpu}" != "1" ]]; then
    echo "ERROR: this matrix is constrained to GPUs 0/1 only, got ${gpu}." >&2
    exit 2
  fi
done

lookup_memory() {
  local scene="$1"
  if [[ -n "${MEMORY_PATH_PREFIX:-}" ]]; then
    local candidate="${MEMORY_PATH_PREFIX}/${scene}/memory_bse.pt"
    [[ -s "${candidate}" ]] && { echo "${candidate}"; return 0; }
  fi
  if [[ -n "${MEMORY_MANIFEST}" && -s "${MEMORY_MANIFEST}" ]]; then
    awk -F "\t" -v s="${scene}" '$2 == s {p=$3} END {print p}' "${MEMORY_MANIFEST}"
    return 0
  fi
  local env_name="MEMORY_PATH_${scene}"
  echo "${!env_name:-}"
}

pids=()
labels=()
logs=()

for i in "${!SCENES[@]}"; do
  scene="${SCENES[$i]}"
  gpu="${GPUS[$(( i % ${#GPUS[@]} ))]}"
  memory_path="$(lookup_memory "${scene}")"
  if [[ -z "${memory_path}" || ! -s "${memory_path}" ]]; then
    msg="missing memory for ${scene}; set MEMORY_PATH_${scene}, MEMORY_PATH_PREFIX, or MEMORY_MANIFEST"
    if [[ "${SKIP_MISSING_MEMORY}" == "true" ]]; then
      echo "[skip] ${msg}" >&2
      continue
    fi
    echo "ERROR: ${msg}" >&2
    exit 2
  fi

  log_dir="${RUN_ROOT}/matrix_logs"
  mkdir -p "${log_dir}"
  log_file="${log_dir}/${scene}_gpu${gpu}.log"
  labels+=("${scene}:gpu${gpu}")
  logs+=("${log_file}")

  (
    SCENE="${scene}" \
    GPU_ID="${gpu}" \
    MEMORY_PATH="${memory_path}" \
    RUN_ROOT="${RUN_ROOT}" \
    DRY_RUN="${DRY_RUN}" \
    bash ace_dinov2_lmc/scripts/run_indoor6_glace_lmc_scene.sh
  ) > "${log_file}" 2>&1 &
  pids+=("$!")

  if (( ${#pids[@]} >= ${#GPUS[@]} )); then
    for j in "${!pids[@]}"; do
      if wait "${pids[$j]}"; then
        echo "[ok] ${labels[$j]} log=${logs[$j]}"
      else
        echo "[failed] ${labels[$j]} log=${logs[$j]}" >&2
        if [[ "${CONTINUE_ON_ERROR}" != "true" ]]; then
          exit 1
        fi
      fi
    done
    pids=()
    labels=()
    logs=()
  fi
done

for j in "${!pids[@]}"; do
  if wait "${pids[$j]}"; then
    echo "[ok] ${labels[$j]} log=${logs[$j]}"
  else
    echo "[failed] ${labels[$j]} log=${logs[$j]}" >&2
    if [[ "${CONTINUE_ON_ERROR}" != "true" ]]; then
      exit 1
    fi
  fi
done
