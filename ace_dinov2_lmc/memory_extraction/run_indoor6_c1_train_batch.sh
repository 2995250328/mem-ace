#!/bin/bash
# Batch train Indoor6 reference-consistent memories.
#
# Training policy:
# - Phase 1: train remaining single-memory runs first
# - Phase 2: train clustered runs scene by scene
# - scene3 is intentionally skipped
#
# GPU scheduling:
# - Phase 1: use two GPUs to run two single-memory jobs in parallel
# - Phase 2: for each clustered scene, run cluster_01 on GPU 0 and cluster_02 on GPU 1

set -euo pipefail

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

ROOT_DIR="/home/xwh/project/ace_depth"
REPO_ROOT="${ROOT_DIR}/ace_dinov2_lmc"
EXTRACT_ROOT="${REPO_ROOT}/memory_extraction/04_evaluation/memory_extract"
DATA_ROOT="${ACE_DATA_ROOT:-/home/xwh/data}"

TRAIN_PRESET="${TRAIN_PRESET:-memory_compare_ace_g_v2}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-10240}"
BUFFER_ON_CPU="${BUFFER_ON_CPU:-false}"
LMC_MODE="${LMC_MODE:-global}"
EXPERIMENT_SUBDIR="${EXPERIMENT_SUBDIR:-memory_pooled_vs_asb_c1}"
DEVICE_A="${DEVICE_A:-cuda:0}"
DEVICE_B="${DEVICE_B:-cuda:1}"
TRAIN_ROOT="${REPO_ROOT}/04_evaluation/train_compare/${EXPERIMENT_SUBDIR}/indoor6_ace"

SINGLE_TAG="40v_v0.05_bilinear_bse_ut0.02_noaug_asb_adaptive_pool1.0_rgate4_prepair8_sor_gm_l2"
CLUSTER_TAG="40v_v0.05_bilinear_bse_ut0.02_noaug_asb_adaptive_pool1.0_rgate4_prepair8_cfb2_sor_gm_l2"

SINGLE_SCENES=(scene5 scene6)
CLUSTER_SCENES=(scene1 scene2a scene4a scene5 scene6)
FAILED_JOBS=()

log_root="${REPO_ROOT}/memory_extraction/batch_train_logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$log_root"
result_root="${log_root}/results"
mkdir -p "$result_root"

on_error() {
  local status=$?
  local line_no=$1
  log "ERROR: script failed at line ${line_no} with exit=${status}"
  log "Log root: ${log_root}"
  log "Result root: ${result_root}"
  jobs -pr | xargs -r kill
  exit "$status"
}

cleanup_jobs() {
  jobs -pr | xargs -r kill
}

trap 'on_error $LINENO' ERR
trap cleanup_jobs EXIT

print_startup_config() {
  log "Indoor6 C1 batch training started"
  log "Repo root: ${REPO_ROOT}"
  log "Extract root: ${EXTRACT_ROOT}"
  log "Train root: ${TRAIN_ROOT}"
  log "Log root: ${log_root}"
  log "Result root: ${result_root}"
  log "Data root: ${DATA_ROOT}"
  log "Devices: ${DEVICE_A}, ${DEVICE_B}"
  log "Train preset: ${TRAIN_PRESET}"
  log "Batch size: ${TRAIN_BATCH_SIZE}"
  log "Buffer on CPU: ${BUFFER_ON_CPU}"
  log "LMC mode: ${LMC_MODE}"
  log "Single-memory scenes: ${SINGLE_SCENES[*]:-none}"
  log "Cluster scenes: ${CLUSTER_SCENES[*]:-none}"
  log "Note: scene1 and scene4a single-memory runs are skipped because they already finished."
}

latest_single_memory() {
  local scene="$1"
  [ -d "${EXTRACT_ROOT}/${scene}" ] || return 0
  find "${EXTRACT_ROOT}/${scene}" \
    -type f \
    -path "*/${SINGLE_TAG}/*/memory_bse.pt" \
    2>/dev/null \
    | sort \
    | tail -n 1 \
    || true
}

cluster_memory_path() {
  local scene="$1"
  local cluster_id="$2"
  local suffix
  suffix=$(printf 'cluster_%02d.pt' "$cluster_id")
  [ -d "${EXTRACT_ROOT}/${scene}" ] || return 0
  find "${EXTRACT_ROOT}/${scene}" \
    -type f \
    -path "*/${CLUSTER_TAG}/*/memory_bse.${suffix}" \
    2>/dev/null \
    | sort \
    | tail -n 1 \
    || true
}

wait_for_job() {
  local pid="$1"
  local label="$2"
  local log_file="$3"
  local result_file="$4"
  local status=0

  if wait "$pid"; then
    log "[ok] ${label}"
    return 0
  fi

  status=$?
  log "[failed] ${label} exit=${status} log=${log_file}" >&2
  FAILED_JOBS+=("${label} exit=${status} log=${log_file}")
  {
    echo "status=failed"
    echo "label=${label}"
    echo "exit_code=${status}"
    echo "train_log=${log_file}"
  } > "$result_file"
  return 0
}

run_train() {
  local device="$1"
  local scene="$2"
  local output_name="$3"
  local memory_path="$4"
  local scene_root="/home/xwh/data/indoor6_ace/${scene}"
  local log_file="$5"
  local result_file="$6"

  mkdir -p "$(dirname "$log_file")"
  log "[start] scene=${scene} device=${device} memory=${memory_path}"
  log "[log] ${log_file}"

  ACE_DATA_ROOT="$DATA_ROOT" \
  python "${REPO_ROOT}/train_ace_dinov2_lmc.py" \
    "$scene_root" \
    "$output_name" \
    --train_preset "$TRAIN_PRESET" \
    --data_backend ace \
    --device "$device" \
    --post_train_eval_device "$device" \
    --use_lmc True \
    --memory_path "$memory_path" \
    --lmc_mode "$LMC_MODE" \
    --experiment_subdir "$EXPERIMENT_SUBDIR" \
    --buffer_on_cpu "$BUFFER_ON_CPU" \
    --batch_size "$TRAIN_BATCH_SIZE" \
    > "$log_file" 2>&1

  local run_dir
  run_dir="$(find "${TRAIN_ROOT}/${scene}" -type f -name "training_full_log.txt" -printf '%T@ %h\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2- || true)"
  local best_file
  best_file="$(find "${TRAIN_ROOT}/${scene}" -type f -name "best_*.pt" -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2- || true)"
  local eval_file
  eval_file="$(find "${TRAIN_ROOT}/${scene}" -type f -name 'eval_summary_*.txt' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2- || true)"

  {
    echo "scene=${scene}"
    echo "device=${device}"
    echo "output_name=${output_name}"
    echo "memory_path=${memory_path}"
    echo "run_dir=${run_dir:-unknown}"
    echo "best_file=${best_file:-unknown}"
    echo "eval_summary=${eval_file:-unknown}"
    echo "train_log=${log_file}"
  } > "$result_file"

  log "[done] scene=${scene} device=${device}"
}

run_single_phase() {
  echo "============================================================"
  echo "Phase 1: single-memory training"
  echo "============================================================"

  local pids=()
  local labels=()
  local logs=()
  local results=()
  local gpu_index=0
  for scene in "${SINGLE_SCENES[@]}"; do
    local memory_path
    memory_path="$(latest_single_memory "$scene")"
    if [ -z "$memory_path" ]; then
      log "ERROR: cannot find single-memory extraction for ${scene}" >&2
      exit 1
    fi

    local device
    if [ "$gpu_index" -eq 0 ]; then
      device="$DEVICE_A"
    else
      device="$DEVICE_B"
    fi

    local ts
    ts="$(date +%Y%m%d_%H%M%S)"
    local output_name="${scene}_c1_${ts}.pt"
    local log_file="${log_root}/single_${scene}_${ts}.log"
    local result_file="${result_root}/single_${scene}_${ts}.txt"

    run_train "$device" "$scene" "$output_name" "$memory_path" "$log_file" "$result_file" &
    pids+=("$!")
    labels+=("single:${scene}:${device}")
    logs+=("$log_file")
    results+=("$result_file")

    gpu_index=$((1 - gpu_index))
    if [ "${#pids[@]}" -eq 2 ]; then
      wait_for_job "${pids[0]}" "${labels[0]}" "${logs[0]}" "${results[0]}"
      wait_for_job "${pids[1]}" "${labels[1]}" "${logs[1]}" "${results[1]}"
      pids=()
      labels=()
      logs=()
      results=()
    fi
  done

  if [ "${#pids[@]}" -gt 0 ]; then
    wait_for_job "${pids[0]}" "${labels[0]}" "${logs[0]}" "${results[0]}"
  fi
}

run_cluster_phase() {
  echo "============================================================"
  echo "Phase 2: clustered training"
  echo "============================================================"

  for scene in "${CLUSTER_SCENES[@]}"; do
    local cluster_01 cluster_02
    cluster_01="$(cluster_memory_path "$scene" 1)"
    cluster_02="$(cluster_memory_path "$scene" 2)"

    if [ -z "$cluster_01" ] || [ -z "$cluster_02" ]; then
      log "ERROR: cannot find clustered memory files for ${scene}" >&2
      exit 1
    fi

    local ts
    ts="$(date +%Y%m%d_%H%M%S)"

    local result_file_a="${result_root}/cluster_${scene}_01_${ts}.txt"
    local result_file_b="${result_root}/cluster_${scene}_02_${ts}.txt"

    run_train "$DEVICE_A" "$scene" "${scene}_c1_cluster01_${ts}.pt" "$cluster_01" "${log_root}/cluster_${scene}_01_${ts}.log" "$result_file_a" &
    pid_a=$!
    run_train "$DEVICE_B" "$scene" "${scene}_c1_cluster02_${ts}.pt" "$cluster_02" "${log_root}/cluster_${scene}_02_${ts}.log" "$result_file_b" &
    pid_b=$!

    wait_for_job "$pid_a" "cluster:${scene}:01:${DEVICE_A}" "${log_root}/cluster_${scene}_01_${ts}.log" "$result_file_a"
    wait_for_job "$pid_b" "cluster:${scene}:02:${DEVICE_B}" "${log_root}/cluster_${scene}_02_${ts}.log" "$result_file_b"
  done
}

print_startup_config
run_single_phase
run_cluster_phase

echo
echo "============================================================"
echo "Result summary"
echo "============================================================"

summary_file="${log_root}/indoor6_c1_training_results.txt"
{
  echo "# Indoor6 C1 training result summary"
  echo "# generated_at=$(date +%Y%m%d_%H%M%S)"
  echo
  for report in "${result_root}"/*.txt; do
    [ -e "$report" ] || continue
    echo "### $(basename "$report")"
    cat "$report"
    echo
  done
} > "$summary_file"

echo "Summary file: $summary_file"
echo "Results dir:   $result_root"
echo
cat "$summary_file"

if [ "${#FAILED_JOBS[@]}" -gt 0 ]; then
  echo
  echo "Failed jobs:"
  printf '  %s\n' "${FAILED_JOBS[@]}"
  exit 1
fi
