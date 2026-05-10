#!/usr/bin/env bash
# Full Indoor6 baseline matrix for the 4090 machine.
#
# Variants per scene:
# - c0_original: C0 memory with the original ASB-style selector, no reference gate, no post-repair.
# - c0_p4:       C0 memory with reference policy gate + post-repair.
# - c1_p4:       C1 memory with reference policy gate + post-repair.
#
# The script has two phases:
# 1. Extract memories.
# 2. Train ACE-G global models from the extracted memories.
#
# Run from anywhere:
#   cd /home/xwh/project/ace_depth
#   conda activate mapanything_new
#   bash ace_dinov2_lmc/memory_extraction/run_indoor6_full_baselines_4090.sh

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/xwh/project/ace_depth}"
REPO_ROOT="${REPO_ROOT:-${ROOT_DIR}/ace_dinov2_lmc}"
EXTRACT_DIR="${EXTRACT_DIR:-${REPO_ROOT}/memory_extraction}"
OUT_ROOT="${OUT_ROOT:-${EXTRACT_DIR}/04_evaluation/memory_extract}"
DATA_ROOT="${ACE_DATA_ROOT:-/home/xwh/data}"
WAI_ROOT="${WAI_ROOT:-${DATA_ROOT}/mapanything-dataset/wai_data/indoor6}"
ACE_ROOT="${ACE_ROOT:-${DATA_ROOT}/indoor6_ace}"

SCENES_STR="${SCENES_STR:-scene1 scene2a scene3 scene4a scene5 scene6}"
VARIANTS_STR="${VARIANTS_STR:-c0_original c0_p4 c1_p4}"
GPUS_STR="${GPUS_STR:-0 1}"

DO_EXTRACT="${DO_EXTRACT:-true}"
DO_TRAIN="${DO_TRAIN:-true}"
RESUME_EXISTING="${RESUME_EXISTING:-true}"
DRY_RUN="${DRY_RUN:-false}"
LIVE_LOGS="${LIVE_LOGS:-true}"
LIVE_TAIL_LINES="${LIVE_TAIL_LINES:-20}"
HEARTBEAT_INTERVAL="${HEARTBEAT_INTERVAL:-300}"

TRAIN_PRESET="${TRAIN_PRESET:-memory_compare_ace_g_v1}"
EXPERIMENT_SUBDIR="${EXPERIMENT_SUBDIR:-indoor6_full_baselines_4090}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-10240}"
TRAINING_BUFFER_SIZE="${TRAINING_BUFFER_SIZE:-2560000}"
BUFFER_SIZE_FINAL="${BUFFER_SIZE_FINAL:-7680000}"
BUFFER_ON_CPU="${BUFFER_ON_CPU:-false}"
BUFFER_ON_CPU_FINAL="${BUFFER_ON_CPU_FINAL:-true}"
POST_TRAIN_SEEDS_STR="${POST_TRAIN_SEEDS_STR:-1305 2026 4242 7777 9001}"
POST_TRAIN_HYPOTHESES="${POST_TRAIN_HYPOTHESES:-256}"

RUN_ROOT="${RUN_ROOT:-${EXTRACT_DIR}/full_baseline_logs/$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${RUN_ROOT}/logs"
RESULT_ROOT="${RUN_ROOT}/results"
MEMORY_MANIFEST="${MEMORY_MANIFEST:-${RUN_ROOT}/memory_manifest.tsv}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-${RUN_ROOT}/train_manifest.tsv}"

mkdir -p "$LOG_ROOT" "$RESULT_ROOT"

read -r -a SCENES <<< "$SCENES_STR"
read -r -a VARIANTS <<< "$VARIANTS_STR"
read -r -a GPUS <<< "$GPUS_STR"
read -r -a POST_TRAIN_SEEDS <<< "$POST_TRAIN_SEEDS_STR"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

bool_true() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

variant_contract_mode() {
  case "$1" in
    c0_original|c0_p4) echo "C0" ;;
    c1_p4) echo "C1" ;;
    *) echo "ERROR: unknown variant $1" >&2; return 1 ;;
  esac
}

variant_gate_enabled() {
  case "$1" in
    c0_original) echo "false" ;;
    c0_p4|c1_p4) echo "true" ;;
    *) echo "ERROR: unknown variant $1" >&2; return 1 ;;
  esac
}

variant_repair_enabled() {
  case "$1" in
    c0_original) echo "false" ;;
    c0_p4|c1_p4) echo "true" ;;
    *) echo "ERROR: unknown variant $1" >&2; return 1 ;;
  esac
}

require_scene_paths() {
  local scene="$1"
  if [ ! -d "${WAI_ROOT}/${scene}_train" ]; then
    echo "ERROR: missing WAI scene root: ${WAI_ROOT}/${scene}_train" >&2
    return 1
  fi
  if [ ! -d "${ACE_ROOT}/${scene}" ]; then
    echo "ERROR: missing ACE scene root: ${ACE_ROOT}/${scene}" >&2
    return 1
  fi
}

find_manifest_memory() {
  local variant="$1"
  local scene="$2"
  [ -f "$MEMORY_MANIFEST" ] || return 0
  awk -F '\t' -v v="$variant" -v s="$scene" '$1 == v && $2 == s {p=$3} END {print p}' "$MEMORY_MANIFEST"
}

append_memory_manifest() {
  local variant="$1"
  local scene="$2"
  local memory_path="$3"
  local tmp="${MEMORY_MANIFEST}.tmp"
  touch "$MEMORY_MANIFEST"
  awk -F '\t' -v v="$variant" -v s="$scene" '$1 != v || $2 != s' "$MEMORY_MANIFEST" > "$tmp"
  printf '%s\t%s\t%s\n' "$variant" "$scene" "$memory_path" >> "$tmp"
  mv "$tmp" "$MEMORY_MANIFEST"
}

start_live_log_tail() {
  local label="$1"
  local log_file="$2"
  STARTED_MONITOR_PID=""
  if ! bool_true "$LIVE_LOGS"; then
    return 0
  fi
  touch "$log_file"
  tail -n "$LIVE_TAIL_LINES" -F "$log_file" 2>/dev/null \
    | awk -v label="$label" '{ printf("[%s] %s\n", label, $0); fflush(); }' &
  STARTED_MONITOR_PID="$!"
}

start_heartbeat() {
  local label="$1"
  local job_pid="$2"
  local log_file="$3"
  STARTED_MONITOR_PID=""
  if [ "$HEARTBEAT_INTERVAL" -le 0 ]; then
    return 0
  fi
  (
    local start_ts
    start_ts="$(date +%s)"
    while kill -0 "$job_pid" 2>/dev/null; do
      sleep "$HEARTBEAT_INTERVAL" || exit 0
      if kill -0 "$job_pid" 2>/dev/null; then
        local now elapsed
        now="$(date +%s)"
        elapsed=$((now - start_ts))
        log "[running] ${label} elapsed=${elapsed}s log=${log_file}"
      fi
    done
  ) &
  STARTED_MONITOR_PID="$!"
}

stop_monitor_pid() {
  local pid="${1:-}"
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  fi
}

run_extract_job() {
  local gpu="$1"
  local variant="$2"
  local scene="$3"
  local log_file="${LOG_ROOT}/extract_${variant}_${scene}.log"
  local result_file="${RESULT_ROOT}/extract_${variant}_${scene}.txt"
  local marker="${RESULT_ROOT}/extract_${variant}_${scene}.start"

  require_scene_paths "$scene"

  if bool_true "$RESUME_EXISTING"; then
    local existing_memory
    existing_memory="$(find_manifest_memory "$variant" "$scene")"
    if [ -n "$existing_memory" ] && [ -f "$existing_memory" ]; then
      log "[skip extract] ${variant}/${scene}: manifest memory exists: ${existing_memory}"
      return 0
    fi
  fi

  local contract_mode gate_enabled repair_enabled
  contract_mode="$(variant_contract_mode "$variant")"
  gate_enabled="$(variant_gate_enabled "$variant")"
  repair_enabled="$(variant_repair_enabled "$variant")"

  : > "$marker"
  log "[extract start] gpu=${gpu} variant=${variant} scene=${scene}"
  log "[extract log] ${log_file}"

  env \
    ACE_DATA_ROOT="$DATA_ROOT" \
    DATASET_ROOT="$WAI_ROOT" \
    DATASET_TYPE=indoor6 \
    DATASET_LOADER=wai \
    GPU_ID="$gpu" \
    SCENE_TRAIN="${scene}_train" \
    OUTPUT_SCENE_NAME="$scene" \
    OUTPUT_ROOT="$OUT_ROOT" \
    N_VIEWS=40 \
    CONTRACT_MODE="$contract_mode" \
    WAI_VIEW_MODE=anchor_support \
    WAI_TRANSFORM=imgnorm \
    WAI_DATA_NORM_TYPE=dinov2 \
    WAI_AUG_CROP=0 \
    POOL_MODE=bse \
    VOXEL_SIZE=0.05 \
    USE_OTSU=true \
    UNIMODAL_THRESHOLD=0.02 \
    PREPOOL_MODE=per_view \
    ENABLE_SOR=true \
    GLOBAL_MERGE=true \
    USE_L2_NORMALIZATION=true \
    ASB_ADAPTIVE=true \
    ASB_CANDIDATE_POOL_RATIO=1.0 \
    ENABLE_REFERENCE_POLICY_GATE="$gate_enabled" \
    REFERENCE_POLICY_TOP_M=4 \
    REFERENCE_POLICY_LIGHT_MIN_SELECTED_LINKS=2 \
    REFERENCE_POLICY_PROBE_Q90_M=0.18 \
    REFERENCE_POLICY_PROBE_MAX_M=0.25 \
    ASB_POST_REPAIR="$repair_enabled" \
    ASB_POST_REPAIR_MAX_SWAPS=8 \
    ASB_POST_REPAIR_TAIL_PERCENTILE=95.0 \
    ASB_POST_REPAIR_MIN_TAIL_IMPROVEMENT_M=0.02 \
    ASB_POST_REPAIR_CLUSTER_TOP_K=3 \
    ENABLE_CLUSTER_FALLBACK=false \
    bash "${EXTRACT_DIR}/extract_memory.sh" \
    > "$log_file" 2>&1

  local memory_path
  memory_path="$(
    find "${OUT_ROOT}/${scene}" -type f -name memory_bse.pt -newer "$marker" -printf '%T@ %p\n' 2>/dev/null \
      | sort -nr \
      | head -n 1 \
      | cut -d' ' -f2- \
      || true
  )"
  if [ -z "$memory_path" ] || [ ! -f "$memory_path" ]; then
    echo "ERROR: extraction finished but no new memory_bse.pt found for ${variant}/${scene}" >&2
    echo "log=${log_file}" >&2
    return 1
  fi

  append_memory_manifest "$variant" "$scene" "$memory_path"
  {
    echo "status=ok"
    echo "variant=${variant}"
    echo "scene=${scene}"
    echo "gpu=${gpu}"
    echo "contract_mode=${contract_mode}"
    echo "memory_path=${memory_path}"
    echo "log=${log_file}"
  } > "$result_file"
  log "[extract done] ${variant}/${scene}: ${memory_path}"
}

run_train_job() {
  local gpu="$1"
  local variant="$2"
  local scene="$3"
  local memory_path="$4"
  local device="cuda:${gpu}"
  local ts
  ts="$(date +%Y%m%d_%H%M%S)"
  local output_name="${scene}_${variant}_4090_${ts}.pt"
  local log_file="${LOG_ROOT}/train_${variant}_${scene}.log"
  local result_file="${RESULT_ROOT}/train_${variant}_${scene}.txt"

  require_scene_paths "$scene"
  if [ ! -f "$memory_path" ]; then
    echo "ERROR: missing memory for ${variant}/${scene}: ${memory_path}" >&2
    return 1
  fi

  log "[train start] gpu=${gpu} variant=${variant} scene=${scene}"
  log "[train log] ${log_file}"

  ACE_DATA_ROOT="$DATA_ROOT" \
  python "${REPO_ROOT}/train_ace_dinov2_lmc.py" \
    "${ACE_ROOT}/${scene}" \
    "$output_name" \
    --train_preset "$TRAIN_PRESET" \
    --data_backend ace \
    --device "$device" \
    --post_train_eval_device "$device" \
    --use_lmc True \
    --memory_path "$memory_path" \
    --lmc_mode global \
    --lmc_fps_start_policy farthest_from_center \
    --experiment_subdir "$EXPERIMENT_SUBDIR" \
    --training_buffer_size "$TRAINING_BUFFER_SIZE" \
    --buffer_size_final "$BUFFER_SIZE_FINAL" \
    --buffer_on_cpu "$BUFFER_ON_CPU" \
    --buffer_on_cpu_final "$BUFFER_ON_CPU_FINAL" \
    --buffer_sample_valid_coords True \
    --buffer_valid_coord_sample_ratio 1.0 \
    --buffer_valid_coord_neighbor_radius 1 \
    --buffer_valid_coord_neighbor_mode cross \
    --c1_aux_ref_loss_weight 0.0 \
    --c1_aux_depth_root "${WAI_ROOT}/${scene}_train" \
    --c1_aux_depth_kind gt_depth \
    --batch_size "$TRAIN_BATCH_SIZE" \
    --post_train_eval_seeds "${POST_TRAIN_SEEDS[@]}" \
    --post_train_hypotheses "$POST_TRAIN_HYPOTHESES" \
    > "$log_file" 2>&1

  local train_root="${REPO_ROOT}/04_evaluation/train_compare/${EXPERIMENT_SUBDIR}/indoor6_ace/${scene}/dino_ace_lmc_ace_g"
  local run_dir
  run_dir="$(find "$train_root" -type f -name "run_config.json" -printf '%T@ %h\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2- || true)"
  local best_file
  best_file="$(find "$train_root" -type f -name "best_*.pt" -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2- || true)"

  {
    echo "status=ok"
    echo "variant=${variant}"
    echo "scene=${scene}"
    echo "gpu=${gpu}"
    echo "memory_path=${memory_path}"
    echo "run_dir=${run_dir:-unknown}"
    echo "best_file=${best_file:-unknown}"
    echo "log=${log_file}"
  } > "$result_file"
  printf '%s\t%s\t%s\t%s\n' "$variant" "$scene" "${run_dir:-unknown}" "${best_file:-unknown}" >> "$TRAIN_MANIFEST"
  log "[train done] ${variant}/${scene}: ${run_dir:-unknown}"
}

run_matrix_phase() {
  local phase="$1"
  local pids=()
  local labels=()
  local log_files=()
  local tail_pids=()
  local heartbeat_pids=()
  local gpu_index=0

  for variant in "${VARIANTS[@]}"; do
    for scene in "${SCENES[@]}"; do
      local gpu="${GPUS[$gpu_index]}"
      local label="${phase}:${variant}:${scene}:gpu${gpu}"
      local log_file="${LOG_ROOT}/${phase}_${variant}_${scene}.log"

      if [ "$phase" = "extract" ]; then
        run_extract_job "$gpu" "$variant" "$scene" &
      else
        local memory_path
        memory_path="$(find_manifest_memory "$variant" "$scene")"
        if [ -z "$memory_path" ]; then
          echo "ERROR: no memory manifest entry for ${variant}/${scene}; run extraction first." >&2
          return 1
        fi
        run_train_job "$gpu" "$variant" "$scene" "$memory_path" &
      fi

      local job_pid="$!"
      pids+=("$job_pid")
      labels+=("$label")
      log_files+=("$log_file")
      start_live_log_tail "$label" "$log_file"
      tail_pids+=("$STARTED_MONITOR_PID")
      start_heartbeat "$label" "$job_pid" "$log_file"
      heartbeat_pids+=("$STARTED_MONITOR_PID")
      gpu_index=$(((gpu_index + 1) % ${#GPUS[@]}))

      if [ "${#pids[@]}" -ge "${#GPUS[@]}" ]; then
        wait_phase_batch pids labels log_files tail_pids heartbeat_pids
        pids=()
        labels=()
        log_files=()
        tail_pids=()
        heartbeat_pids=()
      fi
    done
  done

  if [ "${#pids[@]}" -gt 0 ]; then
    wait_phase_batch pids labels log_files tail_pids heartbeat_pids
  fi
}

wait_phase_batch() {
  local -n _pids="$1"
  local -n _labels="$2"
  local -n _log_files="$3"
  local -n _tail_pids="$4"
  local -n _heartbeat_pids="$5"
  local failures=0
  for i in "${!_pids[@]}"; do
    if wait "${_pids[$i]}"; then
      log "[ok] ${_labels[$i]}"
    else
      log "[failed] ${_labels[$i]}"
      log "[failed log] ${_log_files[$i]}"
      failures=$((failures + 1))
    fi
    stop_monitor_pid "${_heartbeat_pids[$i]}"
    stop_monitor_pid "${_tail_pids[$i]}"
  done
  if [ "$failures" -ne 0 ]; then
    echo "ERROR: ${failures} ${_labels[*]} job(s) failed. See ${LOG_ROOT}" >&2
    return 1
  fi
}

cleanup_jobs() {
  jobs -pr | xargs -r kill
}

trap cleanup_jobs EXIT

log "Full Indoor6 4090 baseline matrix"
log "Run root: ${RUN_ROOT}"
log "Scenes: ${SCENES[*]}"
log "Variants: ${VARIANTS[*]}"
log "GPUs: ${GPUS[*]}"
log "Do extract: ${DO_EXTRACT}; do train: ${DO_TRAIN}"
log "Dry run: ${DRY_RUN}"
log "Live logs: ${LIVE_LOGS} (tail=${LIVE_TAIL_LINES}, heartbeat=${HEARTBEAT_INTERVAL}s)"
log "Train preset: ${TRAIN_PRESET}"
log "Buffers: training=${TRAINING_BUFFER_SIZE}, final=${BUFFER_SIZE_FINAL}, CPU=${BUFFER_ON_CPU}, final_CPU=${BUFFER_ON_CPU_FINAL}"
log "Sampling: prefer valid depth/scene-coordinate patches for all variants (neighbor=cross,radius=1)"
log "Aux ref loss: disabled (--c1_aux_ref_loss_weight 0.0)"

cd "$ROOT_DIR"

if bool_true "$DRY_RUN"; then
  echo
  echo "Planned extraction/training matrix:"
  for variant in "${VARIANTS[@]}"; do
    for scene in "${SCENES[@]}"; do
      printf '  %s\t%s\tcontract=%s\tgate=%s\trepair=%s\n' \
        "$variant" \
        "$scene" \
        "$(variant_contract_mode "$variant")" \
        "$(variant_gate_enabled "$variant")" \
        "$(variant_repair_enabled "$variant")"
    done
  done
  echo
  echo "No extraction or training launched because DRY_RUN=true."
  exit 0
fi

if bool_true "$DO_EXTRACT"; then
  run_matrix_phase extract
else
  log "Skipping extraction; using MEMORY_MANIFEST=${MEMORY_MANIFEST}"
fi

if bool_true "$DO_TRAIN"; then
  : > "$TRAIN_MANIFEST"
  run_matrix_phase train
else
  log "Skipping training."
fi

log "Done."
log "Memory manifest: ${MEMORY_MANIFEST}"
log "Train manifest: ${TRAIN_MANIFEST}"
log "Logs: ${LOG_ROOT}"
