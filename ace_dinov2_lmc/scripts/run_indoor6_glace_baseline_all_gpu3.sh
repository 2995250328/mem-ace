#!/usr/bin/env bash
set -uo pipefail

# Train and evaluate vanilla GLACE on all Indoor6 ACE-format scenes.
#
# This script intentionally uses the original GLACE repository entrypoints
# (train_ace.py/test_ace.py), not ACE-FCN-LMC or GLACE-LMC.
#
# Typical usage from /home/xwh/project/ace_depth:
#   bash ace_dinov2_lmc/scripts/run_indoor6_glace_baseline_all_gpu3.sh
#
# Useful overrides:
#   SCENES="scene3 scene4a" DRY_RUN=true bash ...
#   RUN_ROOT=/data/xwh/runs/indoor6_glace_gpu3 SKIP_EXISTING=false bash ...

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

CONDA_ENV="${CONDA_ENV:-mapanything}"
INDOOR6_ACE_ROOT="${INDOOR6_ACE_ROOT:-/home/xwh/data/indoor6_ace}"
RUN_ROOT="${RUN_ROOT:-/data/xwh/ace_dinov2_lmc/04_evaluation/indoor6_glace_baseline_all_gpu3/$(date +%Y%m%d_%H%M%S)}"
SCENES="${SCENES:-scene1 scene2a scene3 scene4a scene5 scene6}"
GPU_ID="${GPU_ID:-3}"
HYPOTHESES="${HYPOTHESES:-256}"
DRY_RUN="${DRY_RUN:-false}"
SKIP_EXISTING="${SKIP_EXISTING:-true}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-true}"

GLACE_ROOT="${GLACE_ROOT:-/home/xwh/project/glace}"
GLACE_R2FORMER_CKPT="${GLACE_R2FORMER_CKPT:-${GLACE_ROOT}/CVPR23_DeitS_Rerank.pth}"
GLACE_EXTRACT_FEATURES="${GLACE_EXTRACT_FEATURES:-true}"
GLACE_EXTRACT_BATCH_SIZE="${GLACE_EXTRACT_BATCH_SIZE:-128}"
GLACE_EXTRACT_WORKERS="${GLACE_EXTRACT_WORKERS:-4}"
GLACE_RENDER_FLIPPED_PORTRAIT="${GLACE_RENDER_FLIPPED_PORTRAIT:-true}"

GLACE_MAX_ITERATIONS="${GLACE_MAX_ITERATIONS:-30000}"
GLACE_TRAINING_BUFFER_SIZE="${GLACE_TRAINING_BUFFER_SIZE:-8000000}"
GLACE_SAMPLES_PER_IMAGE="${GLACE_SAMPLES_PER_IMAGE:-1024}"
GLACE_BATCH_SIZE="${GLACE_BATCH_SIZE:-32768}"
GLACE_IMAGE_RESOLUTION="${GLACE_IMAGE_RESOLUTION:-480}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CONDA_RUN=(conda run --no-capture-output -n "${CONDA_ENV}")
CUDA_ENV=(env "CUDA_VISIBLE_DEVICES=${GPU_ID}")

STATUS_FILE="${RUN_ROOT}/status.tsv"
CONFIG_TSV="${RUN_ROOT}/run_config.tsv"

mkdir -p "${RUN_ROOT}"
if [[ ! -f "${STATUS_FILE}" ]]; then
  printf "timestamp\tscene\tstage\tstatus\texit_code\tgpu_id\tlog\n" > "${STATUS_FILE}"
fi

log_status() {
  local scene="$1" stage="$2" status="$3" exit_code="$4" log_file="$5"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$(date +%Y-%m-%dT%H:%M:%S)" "$scene" "$stage" "$status" "$exit_code" \
    "$GPU_ID" "$log_file" >> "${STATUS_FILE}"
}

write_run_config() {
  cat > "${CONFIG_TSV}" <<EOF
key	value
CONDA_ENV	${CONDA_ENV}
INDOOR6_ACE_ROOT	${INDOOR6_ACE_ROOT}
RUN_ROOT	${RUN_ROOT}
SCENES	${SCENES}
GPU_ID	${GPU_ID}
CUDA_VISIBLE_DEVICES	${GPU_ID}
HYPOTHESES	${HYPOTHESES}
DRY_RUN	${DRY_RUN}
SKIP_EXISTING	${SKIP_EXISTING}
CONTINUE_ON_ERROR	${CONTINUE_ON_ERROR}
GLACE_ROOT	${GLACE_ROOT}
GLACE_R2FORMER_CKPT	${GLACE_R2FORMER_CKPT}
GLACE_EXTRACT_FEATURES	${GLACE_EXTRACT_FEATURES}
GLACE_EXTRACT_BATCH_SIZE	${GLACE_EXTRACT_BATCH_SIZE}
GLACE_EXTRACT_WORKERS	${GLACE_EXTRACT_WORKERS}
GLACE_RENDER_FLIPPED_PORTRAIT	${GLACE_RENDER_FLIPPED_PORTRAIT}
GLACE_MAX_ITERATIONS	${GLACE_MAX_ITERATIONS}
GLACE_TRAINING_BUFFER_SIZE	${GLACE_TRAINING_BUFFER_SIZE}
GLACE_SAMPLES_PER_IMAGE	${GLACE_SAMPLES_PER_IMAGE}
GLACE_BATCH_SIZE	${GLACE_BATCH_SIZE}
GLACE_IMAGE_RESOLUTION	${GLACE_IMAGE_RESOLUTION}
EOF
}

run_logged_cwd() {
  local scene="$1" stage="$2" log_file="$3" cwd="$4"
  shift 4

  mkdir -p "$(dirname "${log_file}")"
  printf "[%s] scene=%s stage=%s cwd=%s\n" "$(date +%Y-%m-%dT%H:%M:%S)" "$scene" "$stage" "$cwd" | tee "${log_file}"
  printf "CMD:" | tee -a "${log_file}"
  printf " %q" "$@" | tee -a "${log_file}"
  printf "\n" | tee -a "${log_file}"

  if [[ "${DRY_RUN}" == "true" ]]; then
    log_status "$scene" "$stage" "dry_run" 0 "$log_file"
    return 0
  fi

  (
    cd "${cwd}"
    "$@"
  ) >> "${log_file}" 2>&1
  local rc=$?
  if [[ "$rc" -eq 0 ]]; then
    log_status "$scene" "$stage" "ok" "$rc" "$log_file"
  else
    log_status "$scene" "$stage" "failed" "$rc" "$log_file"
  fi
  return "$rc"
}

maybe_skip_scene() {
  local scene="$1" pose_file="$2" skip_log="$3"
  if [[ "${SKIP_EXISTING}" == "true" && -s "${pose_file}" ]]; then
    printf "[skip] %s already has eval output: %s\n" "$scene" "$pose_file" | tee "${skip_log}"
    log_status "$scene" "all" "skipped_existing" 0 "$skip_log"
    return 0
  fi
  return 1
}

check_scene_root() {
  local scene="$1" scene_path="$2"
  if [[ ! -d "${scene_path}/train/rgb" || ! -d "${scene_path}/train/poses" || ! -d "${scene_path}/train/calibration" ]]; then
    echo "ERROR: incomplete train split for ${scene}: ${scene_path}/train" >&2
    return 2
  fi
  if [[ ! -d "${scene_path}/test/rgb" || ! -d "${scene_path}/test/poses" || ! -d "${scene_path}/test/calibration" ]]; then
    echo "ERROR: incomplete test split for ${scene}: ${scene_path}/test" >&2
    return 2
  fi
}

run_scene() {
  local scene="$1"
  local scene_path="${INDOOR6_ACE_ROOT}/${scene}"
  local method_dir="${RUN_ROOT}/${scene}/glace"
  local model="${method_dir}/model.pt"
  local pose_file="${method_dir}/poses_${scene}_post_train.txt"
  local skip_log="${method_dir}/skip.log"

  mkdir -p "${method_dir}"

  if maybe_skip_scene "$scene" "$pose_file" "$skip_log"; then
    return 0
  fi

  check_scene_root "$scene" "$scene_path" || return $?

  if [[ "${GLACE_EXTRACT_FEATURES}" == "true" && ( ! -s "${scene_path}/train/features.npy" || ! -s "${scene_path}/test/features.npy" ) ]]; then
    run_logged_cwd "$scene" "extract_features" "${method_dir}/extract_features.log" "${GLACE_ROOT}" \
      "${CUDA_ENV[@]}" "${CONDA_RUN[@]}" python "${GLACE_ROOT}/datasets/extract_features.py" "${scene_path}" \
        --checkpoint "${GLACE_R2FORMER_CKPT}" \
        --batch_size "${GLACE_EXTRACT_BATCH_SIZE}" \
        --num_workers "${GLACE_EXTRACT_WORKERS}"
    local rc=$?
    [[ "$rc" -eq 0 ]] || return "$rc"
  else
    printf "[%s] scene=%s stage=extract_features skipped_existing_features\n" "$(date +%Y-%m-%dT%H:%M:%S)" "$scene" \
      > "${method_dir}/extract_features.log"
    log_status "$scene" "extract_features" "skipped_existing_features" 0 "${method_dir}/extract_features.log"
  fi

  run_logged_cwd "$scene" "train" "${method_dir}/train.log" "${GLACE_ROOT}" \
    "${CUDA_ENV[@]}" "${CONDA_RUN[@]}" torchrun --standalone --nnodes 1 --nproc-per-node 1 "${GLACE_ROOT}/train_ace.py" "${scene_path}" "${model}" \
      --training_buffer_size "${GLACE_TRAINING_BUFFER_SIZE}" \
      --samples_per_image "${GLACE_SAMPLES_PER_IMAGE}" \
      --batch_size "${GLACE_BATCH_SIZE}" \
      --max_iterations "${GLACE_MAX_ITERATIONS}" \
      --image_resolution "${GLACE_IMAGE_RESOLUTION}" \
      --use_half True \
      --use_homogeneous True \
      --use_aug True \
      --render_flipped_portrait "${GLACE_RENDER_FLIPPED_PORTRAIT}"
  local train_rc=$?
  [[ "$train_rc" -eq 0 ]] || return "$train_rc"

  run_logged_cwd "$scene" "eval" "${method_dir}/eval.log" "${GLACE_ROOT}" \
    "${CUDA_ENV[@]}" "${CONDA_RUN[@]}" python "${GLACE_ROOT}/test_ace.py" "${scene_path}" "${model}" \
      --session post_train \
      --image_resolution "${GLACE_IMAGE_RESOLUTION}" \
      --hypotheses "${HYPOTHESES}" \
      --render_flipped_portrait "${GLACE_RENDER_FLIPPED_PORTRAIT}"
}

write_run_config
printf "Run root : %s\n" "${RUN_ROOT}"
printf "Dataset  : %s\n" "${INDOOR6_ACE_ROOT}"
printf "Scenes   : %s\n" "${SCENES}"
printf "GPU      : %s (CUDA_VISIBLE_DEVICES=%s)\n" "${GPU_ID}" "${GPU_ID}"
printf "GLACE    : %s\n" "${GLACE_ROOT}"
printf "Dry run  : %s\n" "${DRY_RUN}"

for scene in ${SCENES}; do
  printf "\n[%s] Starting scene=%s\n" "$(date +%Y-%m-%dT%H:%M:%S)" "$scene"
  run_scene "$scene"
  rc=$?
  if [[ "$rc" -ne 0 ]]; then
    printf "[failed] scene=%s rc=%s\n" "$scene" "$rc" >&2
    if [[ "${CONTINUE_ON_ERROR}" != "true" ]]; then
      exit "$rc"
    fi
  else
    printf "[ok] scene=%s\n" "$scene"
  fi
done

printf "\nDone. Status: %s\n" "${STATUS_FILE}"
