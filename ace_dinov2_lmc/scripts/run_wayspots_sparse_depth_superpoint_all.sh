#!/usr/bin/env bash
set -uo pipefail

# Build Wayspots sparse depth for all ACE scenes using the SuperPoint strict recipe
# that we currently rely on for ACE-FCN memory extraction.
#
# Outputs per split:
#   <scene>/<split>/sparse_depth_superpoint_strict_nms_r4
#   <workspace_root>/<scene>/<split>/triangulated_model
#   <workspace_root>/<scene>/<split>/triangulated_points3D.ply
#
# Optional convenience links:
#   <scene>/<split>/sparse_depth -> sparse_depth_superpoint_strict_nms_r4
#   <scene>/<split>/sparse_depth_sampling_sp -> sparse_depth_superpoint_strict_nms_r4

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

CONDA_ENV="${CONDA_ENV:-mapanything}"
PYTHON_BIN="${PYTHON_BIN:-python}"
WAYSPOTS_ROOT="${WAYSPOTS_ROOT:-/data/xwh/Wayspots}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/data/xwh/Wayspots_sparse_depth_workspaces}"
RUN_ROOT="${RUN_ROOT:-/data/xwh/ace_dinov2_lmc/04_evaluation/wayspots_sparse_depth/$(date +%Y%m%d_%H%M%S)}"
SCENES="${SCENES:-}"
SPLITS="${SPLITS:-train test}"
DRY_RUN="${DRY_RUN:-false}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-true}"
SKIP_EXISTING="${SKIP_EXISTING:-true}"

GPU_ID="${GPU_ID:-0}"
USE_GPU="${USE_GPU:-true}"
GPU_INDEX="${GPU_INDEX:-${GPU_ID}}"
NUM_THREADS="${NUM_THREADS:-4}"

FEATURE_BACKEND="${FEATURE_BACKEND:-superpoint}"
MATCHER="${MATCHER:-pairs}"
MATCH_WINDOW="${MATCH_WINDOW:-20}"
EXHAUSTIVE_BLOCK_SIZE="${EXHAUSTIVE_BLOCK_SIZE:-50}"
MIN_TRIANGULATION_ANGLE="${MIN_TRIANGULATION_ANGLE:-1.0}"
MAX_DEPTH_M="${MAX_DEPTH_M:-1000.0}"
DEPTH_NMS_RADIUS="${DEPTH_NMS_RADIUS:-4}"
DEPTH_NMS_MAX_POINTS="${DEPTH_NMS_MAX_POINTS:-0}"
DEPTH_NMS_RANK="${DEPTH_NMS_RANK:-uniform}"
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-sparse_depth_superpoint_strict_nms_r4}"

SUPERPOINT_WEIGHTS="${SUPERPOINT_WEIGHTS:-${ROOT_DIR}/ace_dinov2_lmc/superpoint_v1.pth}"
SUPERPOINT_MAX_KEYPOINTS="${SUPERPOINT_MAX_KEYPOINTS:-4096}"
SUPERPOINT_CONF_THRESH="${SUPERPOINT_CONF_THRESH:-0.015}"
SUPERPOINT_NMS_DIST="${SUPERPOINT_NMS_DIST:-4}"
SUPERPOINT_NN_THRESH="${SUPERPOINT_NN_THRESH:-0.7}"
SUPERPOINT_RESIZE_MAX_LONG_EDGE="${SUPERPOINT_RESIZE_MAX_LONG_EDGE:-1600}"

LINK_SPARSE_DEPTH="${LINK_SPARSE_DEPTH:-true}"
LINK_SPARSE_DEPTH_SAMPLING_SP="${LINK_SPARSE_DEPTH_SAMPLING_SP:-true}"
PREPARE_WAI_LINKS="${PREPARE_WAI_LINKS:-true}"

STATUS_FILE="${RUN_ROOT}/status.tsv"
mkdir -p "${RUN_ROOT}" "${WORKSPACE_ROOT}"
if [[ ! -f "${STATUS_FILE}" ]]; then
  printf "timestamp\tscene\tsplit\tstage\tstatus\texit_code\tdevice\tlog\n" > "${STATUS_FILE}"
fi

discover_scenes() {
  local scene_dir scene_name found=()
  for scene_dir in "${WAYSPOTS_ROOT}"/*; do
    [[ -d "${scene_dir}" ]] || continue
    scene_name="$(basename "${scene_dir}")"
    if [[ -d "${scene_dir}/train/rgb" && -d "${scene_dir}/test/rgb" ]]; then
      found+=("${scene_name}")
    fi
  done
  printf "%s\n" "${found[*]}"
}

if [[ -z "${SCENES}" ]]; then
  SCENES="$(discover_scenes)"
fi

log_status() {
  local scene="$1" split="$2" stage="$3" status="$4" exit_code="$5" device="$6" log_file="$7"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$(date +%Y-%m-%dT%H:%M:%S)" "$scene" "$split" "$stage" "$status" "$exit_code" "$device" "$log_file" >> "${STATUS_FILE}"
}

print_cmd() {
  printf "%q " "$@"
  printf "\n"
}

run_logged() {
  local scene="$1" split="$2" stage="$3" device="$4" log_file="$5"
  shift 5
  mkdir -p "$(dirname "${log_file}")"
  {
    printf "[%s] %s/%s/%s device=%s\n" "$(date)" "$scene" "$split" "$stage" "$device"
    print_cmd "$@"
  } | tee -a "${log_file}"

  if [[ "${DRY_RUN}" == "true" ]]; then
    log_status "$scene" "$split" "$stage" "dry_run" 0 "$device" "$log_file"
    return 0
  fi

  "$@" 2>&1 | tee -a "${log_file}"
  local exit_code=${PIPESTATUS[0]}
  if [[ ${exit_code} -eq 0 ]]; then
    log_status "$scene" "$split" "$stage" "ok" 0 "$device" "$log_file"
  else
    log_status "$scene" "$split" "$stage" "failed" "${exit_code}" "$device" "$log_file"
    if [[ "${CONTINUE_ON_ERROR}" != "true" ]]; then
      exit "${exit_code}"
    fi
  fi
  return "${exit_code}"
}

ensure_link() {
  local link_path="$1"
  local target_path="$2"
  if [[ -e "${link_path}" || -L "${link_path}" ]]; then
    return 0
  fi
  ln -s "${target_path}" "${link_path}"
}

prepare_links() {
  local scene_root="$1"
  local split="$2"
  local split_root="${scene_root}/${split}"
  local target="${split_root}/${OUTPUT_SUBDIR}"
  [[ -d "${target}" ]] || return 0
  if [[ "${LINK_SPARSE_DEPTH}" == "true" ]]; then
    ensure_link "${split_root}/sparse_depth" "${target}"
  fi
  if [[ "${LINK_SPARSE_DEPTH_SAMPLING_SP}" == "true" ]]; then
    ensure_link "${split_root}/sparse_depth_sampling_sp" "${target}"
  fi
}

for scene in ${SCENES}; do
  scene_root="${WAYSPOTS_ROOT}/${scene}"
  if [[ ! -d "${scene_root}/train/rgb" || ! -d "${scene_root}/test/rgb" ]]; then
    printf "Skipping incomplete scene: %s\n" "${scene_root}" | tee -a "${RUN_ROOT}/preflight_errors.log"
    log_status "${scene}" "-" "preflight" "failed_scene_layout" 2 "cuda:${GPU_ID}" "${RUN_ROOT}/preflight_errors.log"
    [[ "${CONTINUE_ON_ERROR}" == "true" ]] && continue || exit 2
  fi

  for split in ${SPLITS}; do
    split_root="${scene_root}/${split}"
    output_dir="${split_root}/${OUTPUT_SUBDIR}"
    workspace="${WORKSPACE_ROOT}/${scene}/${split}/$(basename "${OUTPUT_SUBDIR}")"
    log_dir="${RUN_ROOT}/${scene}/${split}"
    if [[ "${SKIP_EXISTING}" == "true" && -d "${output_dir}" ]]; then
      count=$(find "${output_dir}" -maxdepth 1 -name '*.npz' | wc -l)
      if [[ "${count}" -gt 0 ]]; then
        log_status "${scene}" "${split}" "sparse_depth" "skipped_existing" 0 "cuda:${GPU_ID}" "${log_dir}/sparse_depth.log"
        prepare_links "${scene_root}" "${split}"
        continue
      fi
    fi

    cmd=(
      conda run --no-capture-output -n "${CONDA_ENV}" "${PYTHON_BIN}"
      "${ROOT_DIR}/ace_dinov2_lmc/tools/wayspots_known_pose_sparse_depth.py"
      "${scene_root}"
      --split "${split}"
      --workspace "${workspace}"
      --output-subdir "${OUTPUT_SUBDIR}"
      --feature-backend "${FEATURE_BACKEND}"
      --matcher "${MATCHER}"
      --match-window "${MATCH_WINDOW}"
      --exhaustive-block-size "${EXHAUSTIVE_BLOCK_SIZE}"
      --num-threads "${NUM_THREADS}"
      --min-triangulation-angle "${MIN_TRIANGULATION_ANGLE}"
      --max-depth-m "${MAX_DEPTH_M}"
      --depth-nms-radius "${DEPTH_NMS_RADIUS}"
      --depth-nms-max-points "${DEPTH_NMS_MAX_POINTS}"
      --depth-nms-rank "${DEPTH_NMS_RANK}"
      --superpoint-weights "${SUPERPOINT_WEIGHTS}"
      --superpoint-max-keypoints "${SUPERPOINT_MAX_KEYPOINTS}"
      --superpoint-conf-thresh "${SUPERPOINT_CONF_THRESH}"
      --superpoint-nms-dist "${SUPERPOINT_NMS_DIST}"
      --superpoint-nn-thresh "${SUPERPOINT_NN_THRESH}"
      --superpoint-resize-max-long-edge "${SUPERPOINT_RESIZE_MAX_LONG_EDGE}"
      --overwrite
    )
    if [[ "${USE_GPU}" == "true" ]]; then
      cmd+=(--use-gpu --gpu-index "${GPU_INDEX}")
    fi

    run_logged "${scene}" "${split}" "sparse_depth" "cuda:${GPU_ID}" "${log_dir}/sparse_depth.log" "${cmd[@]}"
    [[ $? -eq 0 ]] || continue

    model_dir="${workspace}/triangulated_model"
    if [[ -d "${model_dir}" ]]; then
      run_logged "${scene}" "${split}" "export_ply" "cuda:${GPU_ID}" "${log_dir}/export_ply.log" \
        conda run --no-capture-output -n "${CONDA_ENV}" "${PYTHON_BIN}" \
          "${ROOT_DIR}/ace_dinov2_lmc/tools/export_colmap_points3d_ply.py" \
          --model-dir "${model_dir}" \
          --out-ply "${workspace}/triangulated_points3D.ply"
    fi

    prepare_links "${scene_root}" "${split}"
  done

  if [[ "${PREPARE_WAI_LINKS}" == "true" ]]; then
    run_logged "${scene}" "-" "prepare_wai" "cpu" "${RUN_ROOT}/${scene}/prepare_wai.log" \
      env SCENES="${scene}" bash "${ROOT_DIR}/ace_dinov2_lmc/scripts/prepare_wayspots_lmc_wai.sh"
  fi
done

printf "Done. Logs/status: %s\n" "${RUN_ROOT}"
