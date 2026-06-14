#!/usr/bin/env bash
set -euo pipefail

# Run official MAREPO-S Wayspots evaluation on physical GPUs 0/1.
# Source checkout: /home/xwh/project/marepo
# Weights: /home/xwh/project/marepo/logs -> /data/xwh/marepo/logs
# Run from /home/xwh/project/ace_depth:
#   bash ace_dinov2_lmc/scripts/run_wayspots_marepo_s_gpu01.sh
# Useful:
#   DRY_RUN=true SCENES_STR="wayspots_bears wayspots_cubes" bash ...

ROOT_DIR="${ROOT_DIR:-/home/xwh/project/ace_depth}"
MAREPO_ROOT="${MAREPO_ROOT:-/home/xwh/project/marepo}"
WAYSPOTS_ROOT="${WAYSPOTS_ROOT:-/data/xwh/Wayspots}"
RUN_ROOT="${RUN_ROOT:-/data/xwh/ace_dinov2_lmc/04_evaluation/wayspots_marepo_s_gpu01/$(date +%Y%m%d_%H%M%S)}"
SCENES_STR="${SCENES_STR:-wayspots_bears wayspots_cubes wayspots_inscription wayspots_lawn wayspots_map wayspots_squarebench wayspots_statue wayspots_tendrils wayspots_therock wayspots_wintersign}"
GPUS_STR="${GPUS_STR:-0 1}"
CONDA_ENV="${CONDA_ENV:-mapanything}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATATYPE="${DATATYPE:-test}"
IMAGE_RESOLUTION="${IMAGE_RESOLUTION:-480}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-64}"
LOAD_SCHEME2_SC_MAP="${LOAD_SCHEME2_SC_MAP:-True}"
TRANSFORMER_JSON="${TRANSFORMER_JSON:-${MAREPO_ROOT}/transformer/config/nerf_focal_12T1R_256_homo.json}"
DRY_RUN="${DRY_RUN:-false}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-true}"
SKIP_EXISTING="${SKIP_EXISTING:-true}"

read -r -a SCENES <<< "${SCENES_STR}"
read -r -a GPUS <<< "${GPUS_STR}"
for gpu in "${GPUS[@]}"; do
  if [[ "${gpu}" != "0" && "${gpu}" != "1" ]]; then
    echo "ERROR: this script is constrained to GPUs 0/1, got ${gpu}" >&2
    exit 2
  fi
done

mkdir -p "${RUN_ROOT}/logs"
STATUS_FILE="${RUN_ROOT}/status.tsv"
SUMMARY_FILE="${RUN_ROOT}/summary.tsv"
printf 'timestamp\tscene\tstatus\texit_code\tgpu\tlog\n' > "${STATUS_FILE}"
printf 'scene\tmedian_deg\tmedian_cm\tacc500_10\tacc50_5\tacc25_2\tacc10_5\tacc5_5\tacc2_2\tacc1_1\tavg_ms\tlog\n' > "${SUMMARY_FILE}"

require_file() {
  local path="$1" label="$2"
  if [[ ! -s "${path}" ]]; then
    echo "ERROR: missing ${label}: ${path}" >&2
    return 2
  fi
}

parse_log_to_summary() {
  local scene="$1" log_file="$2"
  awk -v scene="${scene}" -v log_file="${log_file}" '
    /5m\/10deg:/ {gsub(/%/, "", $NF); acc500=$NF}
    /0\.5m\/5deg:/ {gsub(/%/, "", $NF); acc50=$NF}
    /0\.25m\/2deg:/ {gsub(/%/, "", $NF); acc25=$NF}
    /10cm\/5deg:/ {gsub(/%/, "", $NF); acc10=$NF}
    /5cm\/5deg:/ {gsub(/%/, "", $NF); acc5=$NF}
    /2cm\/2deg:/ {gsub(/%/, "", $NF); acc2=$NF}
    /1cm\/1deg:/ {gsub(/%/, "", $NF); acc1=$NF}
    /Median Error:/ {
      med_deg=$3; gsub(/,/, "", med_deg);
      med_cm=$5;
    }
    /Avg\. processing time:/ {avg_ms=$(NF-1)}
    END {
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", scene, med_deg, med_cm, acc500, acc50, acc25, acc10, acc5, acc2, acc1, avg_ms, log_file
    }
  ' "${log_file}" >> "${SUMMARY_FILE}"
}

run_one() {
  local scene="$1" gpu="$2"
  local scene_root="${WAYSPOTS_ROOT}/${scene}"
  local log_file="${RUN_ROOT}/logs/${scene}_gpu${gpu}.log"
  local model_path="${MAREPO_ROOT}/logs/paper_model/marepo_s_${scene}/marepo_s_${scene}.pt"
  local ace_head_path="${MAREPO_ROOT}/logs/wayspots_pretrain/${DATATYPE}/${scene}/${scene}.pt"
  local session="marepo_s_${DATATYPE}_gpu${gpu}"

  if [[ ! -d "${scene_root}/test/rgb" || ! -d "${scene_root}/test/poses" || ! -d "${scene_root}/test/calibration" ]]; then
    echo "ERROR: invalid Wayspots scene layout: ${scene_root}" | tee -a "${log_file}"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$(date +%Y-%m-%dT%H:%M:%S)" "${scene}" failed_missing_scene_layout 2 "cuda:${gpu}" "${log_file}" >> "${STATUS_FILE}"
    return 2
  fi
  require_file "${model_path}" "MAREPO-S model" || return 2
  require_file "${ace_head_path}" "Wayspots ACE head" || return 2
  require_file "${TRANSFORMER_JSON}" "transformer config" || return 2

  if [[ "${SKIP_EXISTING}" == "true" && -s "${log_file}" ]] && rg -q "Test complete\." "${log_file}"; then
    printf '[skip] %s existing complete log: %s\n' "${scene}" "${log_file}"
    parse_log_to_summary "${scene}" "${log_file}"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$(date +%Y-%m-%dT%H:%M:%S)" "${scene}" skipped_existing 0 "cuda:${gpu}" "${log_file}" >> "${STATUS_FILE}"
    return 0
  fi

  printf '[run] scene=%s gpu=%s log=%s\n' "${scene}" "${gpu}" "${log_file}"
  local cmd=(
    "${PYTHON_BIN}" "${MAREPO_ROOT}/test_marepo.py"
    "${scene_root}"
    "${model_path}"
    --encoder_path "${MAREPO_ROOT}/ace_encoder_pretrained.pt"
    --head_network_path "${ace_head_path}"
    --transformer_json "${TRANSFORMER_JSON}"
    --load_scheme2_sc_map "${LOAD_SCHEME2_SC_MAP}"
    --datatype "${DATATYPE}"
    --image_resolution "${IMAGE_RESOLUTION}"
    --test_batch_size "${TEST_BATCH_SIZE}"
    --session "${session}"
  )

  {
    printf '[%s] scene=%s gpu=%s\n' "$(date)" "${scene}" "${gpu}"
    printf 'cd %q\n' "${MAREPO_ROOT}"
    printf 'CUDA_VISIBLE_DEVICES=%q conda run --no-capture-output -n %q ' "${gpu}" "${CONDA_ENV}"
    printf '%q ' "${cmd[@]}"
    printf '\n'
  } | tee "${log_file}"

  if [[ "${DRY_RUN}" == "true" ]]; then
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$(date +%Y-%m-%dT%H:%M:%S)" "${scene}" dry_run 0 "cuda:${gpu}" "${log_file}" >> "${STATUS_FILE}"
    return 0
  fi

  (cd "${MAREPO_ROOT}" && CUDA_VISIBLE_DEVICES="${gpu}" conda run --no-capture-output -n "${CONDA_ENV}" "${cmd[@]}") 2>&1 | tee -a "${log_file}"
  local exit_code=${PIPESTATUS[0]}
  if [[ ${exit_code} -eq 0 ]]; then
    parse_log_to_summary "${scene}" "${log_file}"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$(date +%Y-%m-%dT%H:%M:%S)" "${scene}" ok 0 "cuda:${gpu}" "${log_file}" >> "${STATUS_FILE}"
  else
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$(date +%Y-%m-%dT%H:%M:%S)" "${scene}" failed "${exit_code}" "cuda:${gpu}" "${log_file}" >> "${STATUS_FILE}"
    [[ "${CONTINUE_ON_ERROR}" == "true" ]] || exit "${exit_code}"
  fi
}

printf 'MAREPO root : %s\n' "${MAREPO_ROOT}"
printf 'Wayspots    : %s\n' "${WAYSPOTS_ROOT}"
printf 'Run root    : %s\n' "${RUN_ROOT}"
printf 'Scenes      : %s\n' "${SCENES_STR}"
printf 'GPUs        : %s\n' "${GPUS_STR}"
printf 'Dry run     : %s\n' "${DRY_RUN}"

pids=()
for i in "${!SCENES[@]}"; do
  gpu="${GPUS[$(( i % ${#GPUS[@]} ))]}"
  run_one "${SCENES[$i]}" "${gpu}" &
  pids+=("$!")
  if (( ${#pids[@]} >= ${#GPUS[@]} )); then
    for pid in "${pids[@]}"; do wait "${pid}" || true; done
    pids=()
  fi
done
for pid in "${pids[@]}"; do wait "${pid}" || true; done
printf 'Done. Status: %s\nSummary: %s\n' "${STATUS_FILE}" "${SUMMARY_FILE}"
