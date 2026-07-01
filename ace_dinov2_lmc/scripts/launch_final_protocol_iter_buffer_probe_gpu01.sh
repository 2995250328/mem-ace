#!/usr/bin/env bash
set -euo pipefail

# Final-protocol iter/buffer efficiency probe on physical GPUs 0/1.
#
# Goal:
#   Compare big-buffer fewer-iter vs big-buffer more-iter across representative
#   scenes from Wayspots / Indoor6 / Cambridge.
#
# Run from /home/xwh/project/ace_depth:
#   ACTION=preflight bash ace_dinov2_lmc/scripts/launch_final_protocol_iter_buffer_probe_gpu01.sh
#   ACTION=launch    bash ace_dinov2_lmc/scripts/launch_final_protocol_iter_buffer_probe_gpu01.sh
#
# Defaults:
#   Wayspots:  ACE-FCN-LMC ACE-G every-iter S1/S2-G + STGS anchor-v2, Acc50 best.
#   Indoor6:   DINOv2 + MapAnything memory + STGS inter-frame w=0.01 stage2_g, Acc25 best.
#   Cambridge: ACE-FCN-LMC GLACE-concat R2 stage2, no STGS sidecar by default, median best.

ROOT_DIR="${ROOT_DIR:-/home/xwh/project/ace_depth}"
REPO_ROOT="${REPO_ROOT:-${ROOT_DIR}/ace_dinov2_lmc}"
cd "${ROOT_DIR}"

ACTION="${ACTION:-preflight}"  # preflight | launch | worker
WORKER_GPU="${WORKER_GPU:-}"
QUEUE_FILE="${QUEUE_FILE:-}"
CONDA_ENV="${CONDA_ENV:-mapanything}"
PYTHON_BIN="${PYTHON_BIN:-python}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
SESSION="${SESSION:-final_iter_probe_${STAMP}_gpu01}"
RUN_ROOT="${RUN_ROOT:-/data/xwh/ace_dinov2_lmc/04_evaluation/final_protocol/iter_buffer_probe/${STAMP}_3dataset_2scene_bigbuf_it4-6-10_gpu01}"
LOG_DIR="${RUN_ROOT}/logs"
QUEUE_DIR="${RUN_ROOT}/queues"
STATUS_FILE="${RUN_ROOT}/status.tsv"
MATRIX_FILE="${RUN_ROOT}/matrix.tsv"
MANIFEST_FILE="${RUN_ROOT}/manifest.txt"

GPUS_STR="${GPUS_STR:-0 1}"
ITER_VARIANTS="${ITER_VARIANTS:-it4 it6 it10}"
WAYSPOTS_SCENES="${WAYSPOTS_SCENES:-wayspots_bears wayspots_cubes}"
INDOOR6_SCENES="${INDOOR6_SCENES:-scene2a scene5}"
CAMBRIDGE_SCENES="${CAMBRIDGE_SCENES:-Cambridge_GreatCourt Cambridge_ShopFacade}"
DATASETS="${DATASETS:-wayspots indoor6 cambridge}"

# Dataset-specific big-buffer defaults. Keep buffer fixed within each dataset;
# only lmc_iterations changes across variants.
WAYSPOTS_BUFFER="${WAYSPOTS_BUFFER:-10000000}"
WAYSPOTS_FINAL_BUFFER="${WAYSPOTS_FINAL_BUFFER:-10000000}"
INDOOR6_BUFFER="${INDOOR6_BUFFER:-10000000}"
INDOOR6_FINAL_BUFFER="${INDOOR6_FINAL_BUFFER:-10000000}"
CAMBRIDGE_BUFFER_M="${CAMBRIDGE_BUFFER_M:-20}"
CAMBRIDGE_LR_DIGITS="${CAMBRIDGE_LR_DIGITS:-001}"

POST_TRAIN_EVAL_SEEDS="${POST_TRAIN_EVAL_SEEDS:-1305 2026 4242}"
POST_TRAIN_HYPOTHESES="${POST_TRAIN_HYPOTHESES:-256}"
ITERATION_EVAL_HYPOTHESES="${ITERATION_EVAL_HYPOTHESES:-256}"

# Watchdog: per job, not only per full matrix. It terminates processes whose
# command line contains the job RUN_ROOT when no non-watchdog log has changed.
WATCHDOG_ENABLED="${WATCHDOG_ENABLED:-true}"
WATCHDOG_STALE_SEC="${WATCHDOG_STALE_SEC:-3600}"
WATCHDOG_CHECK_SEC="${WATCHDOG_CHECK_SEC:-120}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-true}"
DRY_RUN="${DRY_RUN:-false}"
SKIP_EXISTING="${SKIP_EXISTING:-true}"

# Existing stable sidecar root for Indoor6 DINO STGS scene2a/scene5.
INDOOR6_SIDECAR_ROOT="${INDOOR6_SIDECAR_ROOT:-/data/xwh/ace_dinov2_lmc/04_evaluation/indoor6/dino_stgs_matrix/20260629_203557_it2_buf10M_pair_gpu23/sidecars}"
INDOOR6_STGS_MODE="${INDOOR6_STGS_MODE:-inter_frame}"
INDOOR6_INTER_FRAME_WEIGHT="${INDOOR6_INTER_FRAME_WEIGHT:-0.01}"
INDOOR6_INTER_FRAME_APPLY_TO="${INDOOR6_INTER_FRAME_APPLY_TO:-stage2_g}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OMP_DYNAMIC="${OMP_DYNAMIC:-FALSE}"

mkdir -p "${LOG_DIR}" "${QUEUE_DIR}"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "${LOG_DIR}/supervisor_${STAMP}.log"
}

bool_true() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

iter_for_variant() {
  local variant="$1"
  case "${variant}" in
    it*) echo "${variant#it}" ;;
    *) echo "ERROR"; return 2 ;;
  esac
}

scene_list_for_dataset() {
  case "$1" in
    wayspots) echo "${WAYSPOTS_SCENES}" ;;
    indoor6) echo "${INDOOR6_SCENES}" ;;
    cambridge) echo "${CAMBRIDGE_SCENES}" ;;
    *) return 2 ;;
  esac
}

primary_metric_for_dataset() {
  case "$1" in
    wayspots) echo "Acc50+Acc10; best_metric=pct50_5" ;;
    indoor6) echo "Acc25+median; best_metric=pct25_5" ;;
    cambridge) echo "median only; best_metric=median_error" ;;
  esac
}

ensure_headers() {
  if [[ ! -f "${STATUS_FILE}" ]]; then
    printf 'timestamp\tdataset\tscene\tvariant\tgpu\tstage\tstatus\texit_code\tjob_root\tlog\n' > "${STATUS_FILE}"
  fi
  if [[ ! -f "${MATRIX_FILE}" ]]; then
    printf 'dataset\tscene\tvariant\titers\tgpu_policy\tbuffer\tfinal_buffer\tmethod\tstgs\tprimary_metric\n' > "${MATRIX_FILE}"
  fi
}

append_status() {
  local dataset="$1" scene="$2" variant="$3" gpu="$4" stage="$5" status="$6" exit_code="$7" job_root="$8" log_file="$9"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date +%Y-%m-%dT%H:%M:%S)" "$dataset" "$scene" "$variant" "$gpu" "$stage" "$status" "$exit_code" "$job_root" "$log_file" >> "${STATUS_FILE}"
}

write_manifest() {
  cat > "${MANIFEST_FILE}" <<EOF
purpose: final-protocol iteration-count efficiency probe
run_root: ${RUN_ROOT}
session: ${SESSION}
gpus: ${GPUS_STR}
iter_variants: ${ITER_VARIANTS}
datasets: ${DATASETS}
scenes:
  wayspots: ${WAYSPOTS_SCENES}
  indoor6: ${INDOOR6_SCENES}
  cambridge: ${CAMBRIDGE_SCENES}
metric_policy:
  indoor6: primary Acc25 + median error
  wayspots: primary Acc50 + Acc10
  cambridge: primary median error only
protocol:
  wayspots: ACE-FCN-LMC ACE-G every_iter S1/S2-G, fusion_in_s2=True, STGS anchor-v2, buffer=${WAYSPOTS_BUFFER}/${WAYSPOTS_FINAL_BUFFER}
  indoor6: DINOv2 + MapAnything memory ACE-G, fusion_in_s2=True, STGS mode=${INDOOR6_STGS_MODE}, inter_weight=${INDOOR6_INTER_FRAME_WEIGHT}, apply=${INDOOR6_INTER_FRAME_APPLY_TO}, buffer=${INDOOR6_BUFFER}/${INDOOR6_FINAL_BUFFER}
  cambridge: ACE-FCN-LMC GLACE-concat R2 stage2, no STGS sidecar by default, buffer=${CAMBRIDGE_BUFFER_M}M/${CAMBRIDGE_BUFFER_M}M
watchdog:
  enabled: ${WATCHDOG_ENABLED}
  stale_sec: ${WATCHDOG_STALE_SEC}
  check_sec: ${WATCHDOG_CHECK_SEC}
EOF
}

record_matrix_line() {
  local dataset="$1" scene="$2" variant="$3" gpu="$4"
  local iters buffer final_buffer method stgs
  iters="$(iter_for_variant "$variant")"
  case "${dataset}" in
    wayspots)
      buffer="${WAYSPOTS_BUFFER}"; final_buffer="${WAYSPOTS_FINAL_BUFFER}"
      method="ace_fcn_lmc_aceg_every_iter_alt_r2"
      stgs="anchor_v2"
      ;;
    indoor6)
      buffer="${INDOOR6_BUFFER}"; final_buffer="${INDOOR6_FINAL_BUFFER}"
      method="dinov2_mapanything_memory_aceg"
      stgs="${INDOOR6_STGS_MODE}_w${INDOOR6_INTER_FRAME_WEIGHT}_${INDOOR6_INTER_FRAME_APPLY_TO}"
      ;;
    cambridge)
      buffer="$((CAMBRIDGE_BUFFER_M * 1000000))"; final_buffer="$((CAMBRIDGE_BUFFER_M * 1000000))"
      method="ace_fcn_lmc_glace_concat_r2_stage2"
      stgs="none_no_sidecar"
      ;;
    *) return 2 ;;
  esac
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$dataset" "$scene" "$variant" "$iters" "gpu${gpu}" "$buffer" "$final_buffer" "$method" "$stgs" "$(primary_metric_for_dataset "$dataset")" >> "${MATRIX_FILE}"
}

write_queues() {
  printf 'dataset\tscene\tvariant\titers\tgpu_policy\tbuffer\tfinal_buffer\tmethod\tstgs\tprimary_metric\n' > "${MATRIX_FILE}"
  : > "${QUEUE_DIR}/gpu0.tsv"
  : > "${QUEUE_DIR}/gpu1.tsv"
  local idx=0
  local dataset scene variant gpu queue_file
  for dataset in ${DATASETS}; do
    for scene in $(scene_list_for_dataset "${dataset}"); do
      for variant in ${ITER_VARIANTS}; do
        if (( idx % 2 == 0 )); then
          gpu=0
        else
          gpu=1
        fi
        queue_file="${QUEUE_DIR}/gpu${gpu}.tsv"
        printf '%s\t%s\t%s\t%s\n' "${dataset}" "${scene}" "${variant}" "${gpu}" >> "${queue_file}"
        record_matrix_line "${dataset}" "${scene}" "${variant}" "${gpu}"
        idx=$((idx + 1))
      done
    done
  done
}

job_root_for() {
  local dataset="$1" scene="$2" variant="$3"
  echo "${RUN_ROOT}/${dataset}/${scene}/${variant}"
}

job_log_for() {
  local dataset="$1" scene="$2" variant="$3" gpu="$4"
  echo "${LOG_DIR}/${dataset}_${scene}_${variant}_gpu${gpu}.log"
}

run_with_watchdog() {
  local dataset="$1" scene="$2" variant="$3" gpu="$4" job_root="$5" log_file="$6"
  shift 6
  local watchdog_pid=""
  mkdir -p "${job_root}" "$(dirname "${log_file}")"

  if bool_true "${WATCHDOG_ENABLED}" && ! bool_true "${DRY_RUN}" && [[ "${ACTION}" != "preflight" ]]; then
    WATCHDOG_PATTERN="${job_root}" bash "${REPO_ROOT}/scripts/watch_runroot_stall.sh" "${job_root}" "${WATCHDOG_STALE_SEC}" "${WATCHDOG_CHECK_SEC}" >> "${job_root}/watchdog.outer.log" 2>&1 &
    watchdog_pid="$!"
    log "watchdog start pid=${watchdog_pid} dataset=${dataset} scene=${scene} variant=${variant} job_root=${job_root}"
  fi

  {
    printf '[%s] START dataset=%s scene=%s variant=%s gpu=%s job_root=%s\n' "$(date)" "$dataset" "$scene" "$variant" "$gpu" "$job_root"
    printf 'command:'
    printf ' %q' "$@"
    printf '\n'
  } | tee -a "${log_file}"

  set +e
  "$@" 2>&1 | tee -a "${log_file}"
  local rc=${PIPESTATUS[0]}
  set -e

  if [[ -n "${watchdog_pid}" ]] && kill -0 "${watchdog_pid}" 2>/dev/null; then
    kill "${watchdog_pid}" 2>/dev/null || true
    wait "${watchdog_pid}" 2>/dev/null || true
  fi

  if [[ "${rc}" -eq 0 ]]; then
    append_status "$dataset" "$scene" "$variant" "$gpu" "train" "ok" 0 "$job_root" "$log_file"
    log "job ok dataset=${dataset} scene=${scene} variant=${variant} gpu=${gpu}"
  else
    append_status "$dataset" "$scene" "$variant" "$gpu" "train" "failed" "$rc" "$job_root" "$log_file"
    log "job failed rc=${rc} dataset=${dataset} scene=${scene} variant=${variant} gpu=${gpu}"
    if [[ "${CONTINUE_ON_ERROR}" != "true" ]]; then
      exit "${rc}"
    fi
  fi
  return "${rc}"
}

run_wayspots_job() {
  local scene="$1" variant="$2" gpu="$3" job_root="$4" log_file="$5"
  local iters
  iters="$(iter_for_variant "$variant")"
  local dry="false"
  if [[ "${ACTION}" == "preflight" ]] || bool_true "${DRY_RUN}"; then
    dry="true"
  fi
  run_with_watchdog "wayspots" "$scene" "$variant" "$gpu" "$job_root" "$log_file" \
    env -u CUDA_VISIBLE_DEVICES \
      SCENES="$scene" CONFIGS="alt_r2" GPU0="$gpu" GPU1="$gpu" \
      RUN_ROOT="$job_root" STAMP="${STAMP}_${scene}_${variant}" \
      LMC_ITERATIONS="$iters" TRAINING_BUFFER_SIZE="${WAYSPOTS_BUFFER}" BUFFER_SIZE_FINAL="${WAYSPOTS_FINAL_BUFFER}" \
      BEST_METRIC="pct50_5" POST_TRAIN_EVAL_SEEDS="${POST_TRAIN_EVAL_SEEDS}" POST_TRAIN_HYPOTHESES="${POST_TRAIN_HYPOTHESES}" \
      ITERATION_EVAL_HYPOTHESES="${ITERATION_EVAL_HYPOTHESES}" DRY_RUN="$dry" \
      bash "${REPO_ROOT}/scripts/launch_wayspots_stgs_s2_schedule_small_matrix_gpu01.sh"
}

run_indoor6_job() {
  local scene="$1" variant="$2" gpu="$3" job_root="$4" log_file="$5"
  local iters guided_mode inter_enabled
  iters="$(iter_for_variant "$variant")"
  guided_mode="anchor_only"
  inter_enabled="False"
  if [[ "${INDOOR6_STGS_MODE}" == "inter_frame" ]]; then
    guided_mode="inter_frame"
    inter_enabled="True"
  fi
  local dry="False" prep="False"
  if [[ "${ACTION}" == "preflight" ]]; then
    prep="True"
  fi
  if bool_true "${DRY_RUN}"; then
    dry="True"
  fi
  run_with_watchdog "indoor6" "$scene" "$variant" "$gpu" "$job_root" "$log_file" \
    env \
      PYTHON_BIN="/home/xwh/miniforge3/envs/mapanything/bin/python" \
      SCENES_STR="$scene" GPUS_STR="$gpu" RUN_ROOT="$job_root" SIDECAR_ROOT="${INDOOR6_SIDECAR_ROOT}" \
      VARIANT_LABEL="${variant}_${INDOOR6_STGS_MODE}_w${INDOOR6_INTER_FRAME_WEIGHT}_${INDOOR6_INTER_FRAME_APPLY_TO}" \
      LMC_ITERATIONS="$iters" TRAINING_BUFFER_SIZE="${INDOOR6_BUFFER}" BUFFER_SIZE_FINAL="${INDOOR6_FINAL_BUFFER}" \
      BEST_METRIC="pct25_5" POST_TRAIN_SEEDS_STR="${POST_TRAIN_EVAL_SEEDS}" POST_TRAIN_HYPOTHESES="${POST_TRAIN_HYPOTHESES}" \
      USE_STGS_GUIDED_SAMPLING="True" USE_STGS_INTER_FRAME_LOSS="$inter_enabled" \
      SFM_TRACK_GUIDED_MODE="$guided_mode" SFM_TRACK_INTER_FRAME_WEIGHT="${INDOOR6_INTER_FRAME_WEIGHT}" \
      SFM_TRACK_INTER_FRAME_APPLY_TO="${INDOOR6_INTER_FRAME_APPLY_TO}" \
      DRY_RUN="$dry" PREP_ONLY="$prep" \
      bash "${REPO_ROOT}/scripts/launch_indoor6_dino_stgs_short_gpu01.sh"
}

run_cambridge_job() {
  local scene="$1" variant="$2" gpu="$3" job_root="$4" log_file="$5"
  local iters cam_variant dry
  iters="$(iter_for_variant "$variant")"
  cam_variant="r2_it${iters}_buf${CAMBRIDGE_BUFFER_M}m_lr${CAMBRIDGE_LR_DIGITS}"
  dry="false"
  if [[ "${ACTION}" == "preflight" ]] || bool_true "${DRY_RUN}"; then
    dry="true"
  fi
  run_with_watchdog "cambridge" "$scene" "$variant" "$gpu" "$job_root" "$log_file" \
    env \
      SCENE="$scene" GPU="$gpu" RUN_ROOT="$job_root" VARIANTS="$cam_variant" \
      POST_TRAIN_EVAL_SEEDS="${POST_TRAIN_EVAL_SEEDS}" POST_TRAIN_HYPOTHESES="${POST_TRAIN_HYPOTHESES}" \
      ITERATION_EVAL_HYPOTHESES="${ITERATION_EVAL_HYPOTHESES}" DRY_RUN="$dry" \
      bash "${REPO_ROOT}/scripts/launch_cambridge_stage2_r2_hparam_matrix_gpu01.sh"
}

run_job() {
  local dataset="$1" scene="$2" variant="$3" gpu="$4"
  local job_root log_file
  job_root="$(job_root_for "$dataset" "$scene" "$variant")"
  log_file="$(job_log_for "$dataset" "$scene" "$variant" "$gpu")"
  append_status "$dataset" "$scene" "$variant" "$gpu" "dispatch" "start" 0 "$job_root" "$log_file"
  case "${dataset}" in
    wayspots) run_wayspots_job "$scene" "$variant" "$gpu" "$job_root" "$log_file" ;;
    indoor6) run_indoor6_job "$scene" "$variant" "$gpu" "$job_root" "$log_file" ;;
    cambridge) run_cambridge_job "$scene" "$variant" "$gpu" "$job_root" "$log_file" ;;
    *) log "ERROR unknown dataset=${dataset}"; return 2 ;;
  esac
}

run_worker() {
  if [[ -z "${QUEUE_FILE}" || ! -f "${QUEUE_FILE}" ]]; then
    echo "ERROR: worker requires QUEUE_FILE" >&2
    exit 2
  fi
  log "worker start gpu=${WORKER_GPU} queue=${QUEUE_FILE}"
  local failed=0
  while IFS=$'\t' read -r dataset scene variant gpu; do
    [[ -n "${dataset}" ]] || continue
    if ! run_job "$dataset" "$scene" "$variant" "$gpu"; then
      failed=1
      [[ "${CONTINUE_ON_ERROR}" == "true" ]] || exit 1
    fi
  done < "${QUEUE_FILE}"
  log "worker done gpu=${WORKER_GPU} failed=${failed}"
  return "${failed}"
}

aggregate_results() {
  if bool_true "${DRY_RUN}" || [[ "${ACTION}" == "preflight" ]]; then
    return 0
  fi
  log "aggregate metric-wise best under ${RUN_ROOT}"
  bash "${REPO_ROOT}/scripts/aggregate_lmc_metricwise_best.sh" "${RUN_ROOT}" 2>&1 | tee -a "${LOG_DIR}/aggregate_${STAMP}.log" || true
}

launch_tmux() {
  if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "ERROR: tmux session already exists: ${SESSION}" >&2
    exit 2
  fi
  local script_path="ace_dinov2_lmc/scripts/$(basename "$0")"
  tmux new-session -d -s "${SESSION}" -n "gpu0" \
    "cd '${ROOT_DIR}' && ACTION=worker WORKER_GPU=0 QUEUE_FILE='${QUEUE_DIR}/gpu0.tsv' RUN_ROOT='${RUN_ROOT}' STAMP='${STAMP}' SESSION='${SESSION}' bash '${script_path}' 2>&1 | tee '${LOG_DIR}/worker_gpu0.tmux.log'"
  tmux new-window -t "${SESSION}" -n "gpu1" \
    "cd '${ROOT_DIR}' && ACTION=worker WORKER_GPU=1 QUEUE_FILE='${QUEUE_DIR}/gpu1.tsv' RUN_ROOT='${RUN_ROOT}' STAMP='${STAMP}' SESSION='${SESSION}' bash '${script_path}' 2>&1 | tee '${LOG_DIR}/worker_gpu1.tmux.log'"
  log "tmux launched session=${SESSION} run_root=${RUN_ROOT}"
}

case "${ACTION}" in
  preflight)
    ensure_headers
    write_manifest
    write_queues
    log "preflight start run_root=${RUN_ROOT}"
    WORKER_GPU=0 QUEUE_FILE="${QUEUE_DIR}/gpu0.tsv" run_worker || true
    WORKER_GPU=1 QUEUE_FILE="${QUEUE_DIR}/gpu1.tsv" run_worker || true
    log "preflight done run_root=${RUN_ROOT}"
    ;;
  launch)
    ensure_headers
    write_manifest
    write_queues
    if [[ -x "${REPO_ROOT}/scripts/agent_safe_snapshot.sh" ]]; then
      bash "${REPO_ROOT}/scripts/agent_safe_snapshot.sh" run_provenance "final_protocol_iter_probe_${STAMP}" "${RUN_ROOT}" \
        >> "${LOG_DIR}/snapshot_${STAMP}.log" 2>&1 || log "WARN run_provenance snapshot failed; continuing"
    fi
    launch_tmux
    ;;
  worker)
    ensure_headers
    run_worker
    aggregate_results
    ;;
  *)
    echo "ERROR: unsupported ACTION=${ACTION}; use preflight | launch | worker" >&2
    exit 2
    ;;
esac
