#!/usr/bin/env bash
set -uo pipefail

# Train and evaluate ACE-family baselines on ACE-format Wayspots scenes.
#
# Supported methods:
#   ace, dinoace, aceg25, glace
#
# Typical usage:
#   bash ace_dinov2_lmc/scripts/run_wayspots_baselines_all.sh
#
# Useful overrides:
#   WAYSPOTS_ROOT=/data/xwh/Wayspots SCENES="wayspots_bears" METHODS="ace dinoace" bash ...
#   GPU_ID=1 RUN_ROOT=/data/xwh/runs/wayspots_baselines bash ...
#   DRY_RUN=true SCENES="wayspots_bears" bash ...

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="${ROOT_DIR}/ace_dinov2_lmc/scripts"

CONDA_ENV="${CONDA_ENV:-mapanything}"
WAYSPOTS_ROOT="${WAYSPOTS_ROOT:-/data/xwh/Wayspots}"
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/ace_dinov2_lmc/04_evaluation/wayspots_baselines/$(date +%Y%m%d_%H%M%S)}"
SCENES="${SCENES:-}"
METHODS="${METHODS:-ace dinoace aceg25 glace}"
GPU_ID="${GPU_ID:-}"
if [[ -n "${GPU_ID}" ]]; then
  DEVICE="${DEVICE:-cuda:${GPU_ID}}"
  EVAL_DEVICE="${EVAL_DEVICE:-cuda:${GPU_ID}}"
else
  DEVICE="${DEVICE:-cuda:0}"
  EVAL_DEVICE="${EVAL_DEVICE:-${DEVICE}}"
fi
HYPOTHESES="${HYPOTHESES:-256}"
DRY_RUN="${DRY_RUN:-false}"
SKIP_EXISTING="${SKIP_EXISTING:-true}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-true}"

ACE_G_ROOT="${ACE_G_ROOT:-/home/xwh/project/ace-g}"
GLACE_ROOT="${GLACE_ROOT:-/home/xwh/project/glace}"
GLACE_R2FORMER_CKPT="${GLACE_R2FORMER_CKPT:-${GLACE_ROOT}/CVPR23_DeitS_Rerank.pth}"
GLACE_EXTRACT_FEATURES="${GLACE_EXTRACT_FEATURES:-true}"
GLACE_EXTRACT_BATCH_SIZE="${GLACE_EXTRACT_BATCH_SIZE:-128}"
GLACE_EXTRACT_WORKERS="${GLACE_EXTRACT_WORKERS:-4}"
GLACE_RENDER_FLIPPED_PORTRAIT="${GLACE_RENDER_FLIPPED_PORTRAIT:-true}"

ACE_EPOCHS="${ACE_EPOCHS:-24}"
ACE_TRAINING_BUFFER_SIZE="${ACE_TRAINING_BUFFER_SIZE:-8000000}"
ACE_SAMPLES_PER_IMAGE="${ACE_SAMPLES_PER_IMAGE:-1024}"
ACE_BATCH_SIZE="${ACE_BATCH_SIZE:-5120}"
ACE_IMAGE_RESOLUTION="${ACE_IMAGE_RESOLUTION:-480}"

DINOACE_EPOCHS="${DINOACE_EPOCHS:-32}"
DINOACE_TRAINING_BUFFER_SIZE="${DINOACE_TRAINING_BUFFER_SIZE:-8000000}"
DINOACE_SAMPLES_PER_IMAGE="${DINOACE_SAMPLES_PER_IMAGE:-512}"
DINOACE_BATCH_SIZE="${DINOACE_BATCH_SIZE:-5120}"
DINOACE_IMAGE_RESOLUTION="${DINOACE_IMAGE_RESOLUTION:-518}"

GLACE_MAX_ITERATIONS="${GLACE_MAX_ITERATIONS:-30000}"
GLACE_TRAINING_BUFFER_SIZE="${GLACE_TRAINING_BUFFER_SIZE:-8000000}"
GLACE_SAMPLES_PER_IMAGE="${GLACE_SAMPLES_PER_IMAGE:-1024}"
GLACE_BATCH_SIZE="${GLACE_BATCH_SIZE:-32768}"
GLACE_IMAGE_RESOLUTION="${GLACE_IMAGE_RESOLUTION:-480}"

ACEG25_CONFIG="${ACEG25_CONFIG:-${ACE_G_ROOT}/src/ace_g/configs/ace_g_25min.yaml}"
ACEG25_BUFFER_CREATION_BATCH_SIZE="${ACEG25_BUFFER_CREATION_BATCH_SIZE:-4}"
ACEG25_BUFFER_CREATION_WORKERS="${ACEG25_BUFFER_CREATION_WORKERS:-8}"
ACEG25_NUM_ITERATIONS="${ACEG25_NUM_ITERATIONS:-4000}"
ACEG25_MAX_DATASET_PASSES="${ACEG25_MAX_DATASET_PASSES:-40}"
ACEG25_MAX_BUFFER_SIZE="${ACEG25_MAX_BUFFER_SIZE:-6000000}"
ACEG25_BATCH_SIZE="${ACEG25_BATCH_SIZE:-32768}"
ACEG25_SAMPLES_PER_IMAGE="${ACEG25_SAMPLES_PER_IMAGE:-1024}"
ACEG25_NUM_MAP_EMBS="${ACEG25_NUM_MAP_EMBS:-}"
ACEG25_BUFFER_DEVICE="${ACEG25_BUFFER_DEVICE:-}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CONDA_RUN=(conda run --no-capture-output -n "${CONDA_ENV}")
CUDA_ENV=()
[[ -n "${GPU_ID}" ]] && CUDA_ENV=(env "CUDA_VISIBLE_DEVICES=${GPU_ID}")
ACEG25_EXTRA_ARGS=()
[[ -n "${ACEG25_NUM_ITERATIONS}" ]] && ACEG25_EXTRA_ARGS+=(--num_iterations "${ACEG25_NUM_ITERATIONS}")
[[ -n "${ACEG25_MAX_DATASET_PASSES}" ]] && ACEG25_EXTRA_ARGS+=(--max_dataset_passes "${ACEG25_MAX_DATASET_PASSES}")
[[ -n "${ACEG25_MAX_BUFFER_SIZE}" ]] && ACEG25_EXTRA_ARGS+=(--max_buffer_size "${ACEG25_MAX_BUFFER_SIZE}")
[[ -n "${ACEG25_BATCH_SIZE}" ]] && ACEG25_EXTRA_ARGS+=(--batch_size "${ACEG25_BATCH_SIZE}")
[[ -n "${ACEG25_SAMPLES_PER_IMAGE}" ]] && ACEG25_EXTRA_ARGS+=(--samples_per_image "${ACEG25_SAMPLES_PER_IMAGE}")
[[ -n "${ACEG25_NUM_MAP_EMBS}" ]] && ACEG25_EXTRA_ARGS+=(--num_map_embs "${ACEG25_NUM_MAP_EMBS}")
[[ -n "${ACEG25_BUFFER_DEVICE}" ]] && ACEG25_EXTRA_ARGS+=(--buffer_device "${ACEG25_BUFFER_DEVICE}")
STATUS_FILE="${RUN_ROOT}/status.tsv"
CONFIG_TSV="${RUN_ROOT}/run_config.tsv"
CONFIG_ENV="${RUN_ROOT}/run_config.env"

mkdir -p "${RUN_ROOT}"
if [[ ! -f "${STATUS_FILE}" ]]; then
  printf "timestamp\tscene\tmethod\tstage\tstatus\texit_code\ttrain_device\teval_device\tlog\n" > "${STATUS_FILE}"
fi

discover_scenes() {
  local scene_dir scene_name found=()
  for scene_dir in "${WAYSPOTS_ROOT}"/*; do
    [[ -d "${scene_dir}" ]] || continue
    scene_name="$(basename "${scene_dir}")"
    if [[ -d "${scene_dir}/train/rgb" && -d "${scene_dir}/train/poses" && -d "${scene_dir}/train/calibration" && -d "${scene_dir}/test/rgb" && -d "${scene_dir}/test/poses" && -d "${scene_dir}/test/calibration" ]]; then
      found+=("${scene_name}")
    fi
  done
  printf "%s\n" "${found[*]}"
}

if [[ -z "${SCENES}" ]]; then
  SCENES="$(discover_scenes)"
fi

log_status() {
  local scene="$1" method="$2" stage="$3" status="$4" exit_code="$5" log_file="$6"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$(date +%Y-%m-%dT%H:%M:%S)" "$scene" "$method" "$stage" "$status" "$exit_code" \
    "$DEVICE" "$EVAL_DEVICE" "$log_file" >> "${STATUS_FILE}"
}

write_run_config() {
  cat > "${CONFIG_TSV}" <<EOF
key	value
CONDA_ENV	${CONDA_ENV}
WAYSPOTS_ROOT	${WAYSPOTS_ROOT}
RUN_ROOT	${RUN_ROOT}
SCENES	${SCENES}
METHODS	${METHODS}
GPU_ID	${GPU_ID:-<unset>}
DEVICE	${DEVICE}
EVAL_DEVICE	${EVAL_DEVICE}
CUDA_VISIBLE_DEVICES_FOR_DEVICELESS_TOOLS	${GPU_ID:-<unset>}
HYPOTHESES	${HYPOTHESES}
DRY_RUN	${DRY_RUN}
SKIP_EXISTING	${SKIP_EXISTING}
CONTINUE_ON_ERROR	${CONTINUE_ON_ERROR}
ACE_EPOCHS	${ACE_EPOCHS}
ACE_TRAINING_BUFFER_SIZE	${ACE_TRAINING_BUFFER_SIZE}
ACE_SAMPLES_PER_IMAGE	${ACE_SAMPLES_PER_IMAGE}
ACE_BATCH_SIZE	${ACE_BATCH_SIZE}
ACE_IMAGE_RESOLUTION	${ACE_IMAGE_RESOLUTION}
DINOACE_EPOCHS	${DINOACE_EPOCHS}
DINOACE_TRAINING_BUFFER_SIZE	${DINOACE_TRAINING_BUFFER_SIZE}
DINOACE_SAMPLES_PER_IMAGE	${DINOACE_SAMPLES_PER_IMAGE}
DINOACE_BATCH_SIZE	${DINOACE_BATCH_SIZE}
DINOACE_IMAGE_RESOLUTION	${DINOACE_IMAGE_RESOLUTION}
GLACE_MAX_ITERATIONS	${GLACE_MAX_ITERATIONS}
GLACE_TRAINING_BUFFER_SIZE	${GLACE_TRAINING_BUFFER_SIZE}
GLACE_SAMPLES_PER_IMAGE	${GLACE_SAMPLES_PER_IMAGE}
GLACE_BATCH_SIZE	${GLACE_BATCH_SIZE}
GLACE_IMAGE_RESOLUTION	${GLACE_IMAGE_RESOLUTION}
GLACE_EXTRACT_FEATURES	${GLACE_EXTRACT_FEATURES}
GLACE_EXTRACT_BATCH_SIZE	${GLACE_EXTRACT_BATCH_SIZE}
GLACE_EXTRACT_WORKERS	${GLACE_EXTRACT_WORKERS}
GLACE_RENDER_FLIPPED_PORTRAIT	${GLACE_RENDER_FLIPPED_PORTRAIT}
ACEG25_CONFIG	${ACEG25_CONFIG}
ACEG25_NUM_ITERATIONS	${ACEG25_NUM_ITERATIONS:-<config-default>}
ACEG25_MAX_DATASET_PASSES	${ACEG25_MAX_DATASET_PASSES:-<config-default>}
ACEG25_MAX_BUFFER_SIZE	${ACEG25_MAX_BUFFER_SIZE:-<config-default>}
ACEG25_BATCH_SIZE	${ACEG25_BATCH_SIZE:-<config-default>}
ACEG25_SAMPLES_PER_IMAGE	${ACEG25_SAMPLES_PER_IMAGE:-<config-default>}
ACEG25_NUM_MAP_EMBS	${ACEG25_NUM_MAP_EMBS:-<config-default>}
ACEG25_BUFFER_DEVICE	${ACEG25_BUFFER_DEVICE:-<train-device>}
ACEG25_BUFFER_CREATION_BATCH_SIZE	${ACEG25_BUFFER_CREATION_BATCH_SIZE}
ACEG25_BUFFER_CREATION_WORKERS	${ACEG25_BUFFER_CREATION_WORKERS}
EOF
  cat > "${CONFIG_ENV}" <<EOF
CONDA_ENV=${CONDA_ENV}
WAYSPOTS_ROOT=${WAYSPOTS_ROOT}
RUN_ROOT=${RUN_ROOT}
SCENES=${SCENES}
METHODS=${METHODS}
GPU_ID=${GPU_ID}
DEVICE=${DEVICE}
EVAL_DEVICE=${EVAL_DEVICE}
CUDA_VISIBLE_DEVICES_FOR_DEVICELESS_TOOLS=${GPU_ID}
HYPOTHESES=${HYPOTHESES}
ACE_EPOCHS=${ACE_EPOCHS}
ACE_TRAINING_BUFFER_SIZE=${ACE_TRAINING_BUFFER_SIZE}
ACE_SAMPLES_PER_IMAGE=${ACE_SAMPLES_PER_IMAGE}
ACE_BATCH_SIZE=${ACE_BATCH_SIZE}
ACE_IMAGE_RESOLUTION=${ACE_IMAGE_RESOLUTION}
DINOACE_EPOCHS=${DINOACE_EPOCHS}
DINOACE_TRAINING_BUFFER_SIZE=${DINOACE_TRAINING_BUFFER_SIZE}
DINOACE_SAMPLES_PER_IMAGE=${DINOACE_SAMPLES_PER_IMAGE}
DINOACE_BATCH_SIZE=${DINOACE_BATCH_SIZE}
DINOACE_IMAGE_RESOLUTION=${DINOACE_IMAGE_RESOLUTION}
GLACE_MAX_ITERATIONS=${GLACE_MAX_ITERATIONS}
GLACE_TRAINING_BUFFER_SIZE=${GLACE_TRAINING_BUFFER_SIZE}
GLACE_SAMPLES_PER_IMAGE=${GLACE_SAMPLES_PER_IMAGE}
GLACE_BATCH_SIZE=${GLACE_BATCH_SIZE}
GLACE_IMAGE_RESOLUTION=${GLACE_IMAGE_RESOLUTION}
GLACE_EXTRACT_FEATURES=${GLACE_EXTRACT_FEATURES}
GLACE_EXTRACT_BATCH_SIZE=${GLACE_EXTRACT_BATCH_SIZE}
GLACE_EXTRACT_WORKERS=${GLACE_EXTRACT_WORKERS}
GLACE_RENDER_FLIPPED_PORTRAIT=${GLACE_RENDER_FLIPPED_PORTRAIT}
ACEG25_CONFIG=${ACEG25_CONFIG}
ACEG25_NUM_ITERATIONS=${ACEG25_NUM_ITERATIONS}
ACEG25_MAX_DATASET_PASSES=${ACEG25_MAX_DATASET_PASSES}
ACEG25_MAX_BUFFER_SIZE=${ACEG25_MAX_BUFFER_SIZE}
ACEG25_BATCH_SIZE=${ACEG25_BATCH_SIZE}
ACEG25_SAMPLES_PER_IMAGE=${ACEG25_SAMPLES_PER_IMAGE}
ACEG25_NUM_MAP_EMBS=${ACEG25_NUM_MAP_EMBS}
ACEG25_BUFFER_DEVICE=${ACEG25_BUFFER_DEVICE}
ACEG25_BUFFER_CREATION_BATCH_SIZE=${ACEG25_BUFFER_CREATION_BATCH_SIZE}
ACEG25_BUFFER_CREATION_WORKERS=${ACEG25_BUFFER_CREATION_WORKERS}
EOF
}

print_run_config() {
  printf "Config  : %s and %s\n" "${CONFIG_TSV}" "${CONFIG_ENV}"
  printf "ACE     : buffer=%s spi=%s batch=%s epochs=%s res=%s\n" \
    "${ACE_TRAINING_BUFFER_SIZE}" "${ACE_SAMPLES_PER_IMAGE}" "${ACE_BATCH_SIZE}" "${ACE_EPOCHS}" "${ACE_IMAGE_RESOLUTION}"
  printf "DINOACE : buffer=%s spi=%s batch=%s epochs=%s res=%s\n" \
    "${DINOACE_TRAINING_BUFFER_SIZE}" "${DINOACE_SAMPLES_PER_IMAGE}" "${DINOACE_BATCH_SIZE}" "${DINOACE_EPOCHS}" "${DINOACE_IMAGE_RESOLUTION}"
  printf "GLACE   : buffer=%s spi=%s batch=%s iterations=%s res=%s flipped_portrait=%s\n" \
    "${GLACE_TRAINING_BUFFER_SIZE}" "${GLACE_SAMPLES_PER_IMAGE}" "${GLACE_BATCH_SIZE}" "${GLACE_MAX_ITERATIONS}" "${GLACE_IMAGE_RESOLUTION}" "${GLACE_RENDER_FLIPPED_PORTRAIT}"
  printf "ACE-G25 : config=%s\n" "${ACEG25_CONFIG}"
  printf "ACE-G25 : overrides iterations=%s max_passes=%s buffer=%s spi=%s batch=%s map_embs=%s buffer_device=%s buffer_create_batch=%s workers=%s\n" \
    "${ACEG25_NUM_ITERATIONS:-<config>}" "${ACEG25_MAX_DATASET_PASSES:-<config>}" \
    "${ACEG25_MAX_BUFFER_SIZE:-<config>}" "${ACEG25_SAMPLES_PER_IMAGE:-<config>}" \
    "${ACEG25_BATCH_SIZE:-<config>}" "${ACEG25_NUM_MAP_EMBS:-<config>}" \
    "${ACEG25_BUFFER_DEVICE:-<train-device>}" \
    "${ACEG25_BUFFER_CREATION_BATCH_SIZE}" "${ACEG25_BUFFER_CREATION_WORKERS}"
}

print_cmd() {
  printf "%q " "$@"
  printf "\n"
}

run_logged() {
  local scene="$1" method="$2" stage="$3" log_file="$4"
  shift 4
  mkdir -p "$(dirname "${log_file}")"
  {
    printf "[%s] %s/%s/%s\n" "$(date)" "$scene" "$method" "$stage"
    print_cmd "$@"
  } | tee -a "${log_file}"

  if [[ "${DRY_RUN}" == "true" ]]; then
    log_status "$scene" "$method" "$stage" "dry_run" 0 "$log_file"
    return 0
  fi

  "$@" 2>&1 | tee -a "${log_file}"
  local exit_code=${PIPESTATUS[0]}
  if [[ ${exit_code} -eq 0 ]]; then
    log_status "$scene" "$method" "$stage" "ok" 0 "$log_file"
  else
    log_status "$scene" "$method" "$stage" "failed" "${exit_code}" "$log_file"
    if [[ "${CONTINUE_ON_ERROR}" != "true" ]]; then
      exit "${exit_code}"
    fi
  fi
  return "${exit_code}"
}

run_logged_cwd() {
  local scene="$1" method="$2" stage="$3" log_file="$4" cwd="$5"
  shift 5
  mkdir -p "$(dirname "${log_file}")"
  {
    printf "[%s] %s/%s/%s\n" "$(date)" "$scene" "$method" "$stage"
    printf "cd %q && " "${cwd}"
    print_cmd "$@"
  } | tee -a "${log_file}"

  if [[ "${DRY_RUN}" == "true" ]]; then
    log_status "$scene" "$method" "$stage" "dry_run" 0 "$log_file"
    return 0
  fi

  (cd "${cwd}" && "$@") 2>&1 | tee -a "${log_file}"
  local exit_code=${PIPESTATUS[0]}
  if [[ ${exit_code} -eq 0 ]]; then
    log_status "$scene" "$method" "$stage" "ok" 0 "$log_file"
  else
    log_status "$scene" "$method" "$stage" "failed" "${exit_code}" "$log_file"
    if [[ "${CONTINUE_ON_ERROR}" != "true" ]]; then
      exit "${exit_code}"
    fi
  fi
  return "${exit_code}"
}

resolve_dinov2_path() {
  if [[ -n "${DINOV2_PATH:-}" && -f "${DINOV2_PATH}" ]]; then
    printf "%s\n" "${DINOV2_PATH}"
    return 0
  fi
  local candidates=(
    "${ROOT_DIR}/checkpoints/dinov2_vitl14_pretrain.pth"
    "/home/xwh/data/checkpoints/dinov2_vitl14_pretrain.pth"
    "/mnt/storage/xwh/checkpoints/dinov2_vitl14_pretrain.pth"
    "/data/xwh/checkpoints/dinov2_vitl14_pretrain.pth"
  )
  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -f "${candidate}" ]]; then
      printf "%s\n" "${candidate}"
      return 0
    fi
  done
  printf "\n"
  return 1
}

has_method() {
  local needle="$1"
  local method
  for method in ${METHODS}; do
    [[ "${method}" == "${needle}" ]] && return 0
  done
  return 1
}

check_scene() {
  local scene_path="$1"
  [[ -d "${scene_path}/train/rgb" ]] || return 1
  [[ -d "${scene_path}/train/poses" ]] || return 1
  [[ -d "${scene_path}/train/calibration" ]] || return 1
  [[ -d "${scene_path}/test/rgb" ]] || return 1
  [[ -d "${scene_path}/test/poses" ]] || return 1
  [[ -d "${scene_path}/test/calibration" ]] || return 1
}

detect_rgb_glob() {
  local split_dir="$1"
  if compgen -G "${split_dir}/rgb/*.jpg" > /dev/null; then
    printf "%s\n" "${split_dir}/rgb/*.jpg"
    return 0
  fi
  if compgen -G "${split_dir}/rgb/*.png" > /dev/null; then
    printf "%s\n" "${split_dir}/rgb/*.png"
    return 0
  fi
  if compgen -G "${split_dir}/rgb/*.jpeg" > /dev/null; then
    printf "%s\n" "${split_dir}/rgb/*.jpeg"
    return 0
  fi
  return 1
}

detect_calibration_glob() {
  local split_dir="$1"
  if compgen -G "${split_dir}/calibration/*.calibration" > /dev/null; then
    printf "%s
" "${split_dir}/calibration/*.calibration"
    return 0
  fi
  if compgen -G "${split_dir}/calibration/*.txt" > /dev/null; then
    printf "%s
" "${split_dir}/calibration/*.txt"
    return 0
  fi
  return 1
}

locate_dinoace_model() {
  local exp_root="$1" scene="$2"
  local found
  found=$(find "${exp_root}" -name "${scene}_dinov2_ep${DINOACE_EPOCHS}_bs${DINOACE_BATCH_SIZE}_model.pt" | sort | head -n 1)
  [[ -n "${found}" ]] || return 1
  printf "%s
" "${found}"
}

maybe_skip() {
  local output_file="$1"
  [[ "${SKIP_EXISTING}" == "true" && -s "${output_file}" ]]
}

summarize() {
  "${CONDA_RUN[@]}" python "${SCRIPT_DIR}/summarize_mushroom_baselines.py" \
    --run-root "${RUN_ROOT}" \
    --dataset-root "${WAYSPOTS_ROOT}" \
    --scenes ${SCENES} \
    --methods ${METHODS}
}

train_ace() {
  local scene="$1" scene_path="$2" method_dir="$3"
  local model="${method_dir}/model.pt"
  local pose_file="${method_dir}/poses_${scene}_post_train.txt"
  mkdir -p "${method_dir}"

  if maybe_skip "${pose_file}"; then
    log_status "$scene" "ace" "all" "skipped_existing" 0 "${method_dir}/skip.log"
    return 0
  fi

  run_logged "$scene" "ace" "train" "${method_dir}/train.log" \
    "${CONDA_RUN[@]}" python "${ROOT_DIR}/train_ace.py" "${scene_path}" "${model}" \
      --device "${DEVICE}" \
      --training_buffer_size "${ACE_TRAINING_BUFFER_SIZE}" \
      --samples_per_image "${ACE_SAMPLES_PER_IMAGE}" \
      --batch_size "${ACE_BATCH_SIZE}" \
      --epochs "${ACE_EPOCHS}" \
      --image_resolution "${ACE_IMAGE_RESOLUTION}" \
      --use_half True \
      --use_homogeneous True \
      --use_aug True \
      --eval_after_train False

  [[ $? -eq 0 ]] || return 0

  run_logged "$scene" "ace" "eval" "${method_dir}/eval.log" \
    "${CONDA_RUN[@]}" python "${ROOT_DIR}/test_ace.py" "${scene_path}" "${model}" \
      --device "${EVAL_DEVICE}" \
      --session post_train \
      --image_resolution "${ACE_IMAGE_RESOLUTION}" \
      --hypotheses "${HYPOTHESES}"
}

train_dinoace() {
  local scene="$1" scene_path="$2" method_dir="$3"
  local dataset_name
  dataset_name="$(basename "${WAYSPOTS_ROOT}")"
  local dino_path
  dino_path="$(resolve_dinov2_path || true)"
  if [[ -z "${dino_path}" ]]; then
    log_status "$scene" "dinoace" "preflight" "failed_missing_dinov2" 2 "${method_dir}/preflight.log"
    return 0
  fi

  local exp_root="${method_dir}/exp"
  local model=""
  local pose_file="${method_dir}/poses_${scene}_post_train.txt"
  mkdir -p "${method_dir}"

  if maybe_skip "${pose_file}"; then
    log_status "$scene" "dinoace" "all" "skipped_existing" 0 "${method_dir}/skip.log"
    return 0
  fi

  run_logged "$scene" "dinoace" "train" "${method_dir}/train.log" \
    "${CONDA_RUN[@]}" python "${ROOT_DIR}/train_ace_dinov2.py" "${scene_path}" model \
      --experiment_root "${exp_root}" \
      --dinov2_path "${dino_path}" \
      --device "${DEVICE}" \
      --training_buffer_size "${DINOACE_TRAINING_BUFFER_SIZE}" \
      --samples_per_image "${DINOACE_SAMPLES_PER_IMAGE}" \
      --batch_size "${DINOACE_BATCH_SIZE}" \
      --epochs "${DINOACE_EPOCHS}" \
      --image_resolution "${DINOACE_IMAGE_RESOLUTION}" \
      --eval_after_train False

  [[ $? -eq 0 ]] || return 0

  model="$(locate_dinoace_model "${exp_root}" "${scene}" || true)"
  if [[ -z "${model}" ]]; then
    log_status "$scene" "dinoace" "eval" "failed_missing_model_after_train" 2 "${method_dir}/eval.log"
    return 0
  fi

  run_logged "$scene" "dinoace" "eval" "${method_dir}/eval.log" \
    "${CONDA_RUN[@]}" python "${ROOT_DIR}/test_ace_dinov2.py" "${scene_path}" "${model}" \
      --dinov2_path "${dino_path}" \
      --device "${EVAL_DEVICE}" \
      --session post_train \
      --image_resolution "${DINOACE_IMAGE_RESOLUTION}" \
      --eval_output_dir "${method_dir}" \
      --hypotheses "${HYPOTHESES}"
}

train_aceg25() {
  local scene="$1" scene_path="$2" method_dir="$3"
  local train_rgb_glob test_rgb_glob train_calib_glob test_calib_glob
  train_rgb_glob="$(detect_rgb_glob "${scene_path}/train")" || {
    log_status "$scene" "aceg25" "preflight" "failed_missing_train_rgb" 2 "${method_dir}/preflight.log"
    return 0
  }
  test_rgb_glob="$(detect_rgb_glob "${scene_path}/test")" || {
    log_status "$scene" "aceg25" "preflight" "failed_missing_test_rgb" 2 "${method_dir}/preflight.log"
    return 0
  }
  train_calib_glob="$(detect_calibration_glob "${scene_path}/train")" || {
    log_status "$scene" "aceg25" "preflight" "failed_missing_train_calibration" 2 "${method_dir}/preflight.log"
    return 0
  }
  test_calib_glob="$(detect_calibration_glob "${scene_path}/test")" || {
    log_status "$scene" "aceg25" "preflight" "failed_missing_test_calibration" 2 "${method_dir}/preflight.log"
    return 0
  }

  local session="${scene}_aceg25"
  local pose_file="${method_dir}/${session}_registered_poses.txt"
  mkdir -p "${method_dir}"

  if maybe_skip "${pose_file}"; then
    log_status "$scene" "aceg25" "all" "skipped_existing" 0 "${method_dir}/skip.log"
    return 0
  fi

  run_logged_cwd "$scene" "aceg25" "train" "${method_dir}/train.log" "${ACE_G_ROOT}" \
    "${CONDA_RUN[@]}" python -m ace_g.train_single_scene \
      --config "${ACEG25_CONFIG}" \
      "${ACEG25_EXTRA_ARGS[@]}" \
      --dataset.rgb_files "${train_rgb_glob}" \
      --dataset.pose_files "${scene_path}/train/poses/*.txt" \
      --dataset.calibration_files "${train_calib_glob}" \
      --output_dir "${method_dir}" \
      --session_id "${session}" \
      --buffer_creation_batch_size "${ACEG25_BUFFER_CREATION_BATCH_SIZE}" \
      --buffer_creation_workers "${ACEG25_BUFFER_CREATION_WORKERS}" \
      --device "${DEVICE}"

  [[ $? -eq 0 ]] || return 0

  run_logged_cwd "$scene" "aceg25" "register" "${method_dir}/register.log" "${ACE_G_ROOT}" \
    "${CONDA_RUN[@]}" python -m ace_g.register_images \
      --config "${method_dir}/${session}_map.yaml" \
      --dataset.rgb_files "${test_rgb_glob}" \
      --dataset.calibration_files "${test_calib_glob}" \
      --output_dir "${method_dir}" \
      --session_id "${session}" \
      --hypotheses "${HYPOTHESES}" \
      --device "${EVAL_DEVICE}"

  [[ $? -eq 0 ]] || return 0

  run_logged_cwd "$scene" "aceg25" "official_eval" "${method_dir}/official_eval.log" "${ACE_G_ROOT}" \
    "${CONDA_RUN[@]}" python -m ace_g.eval_poses \
      --config "${method_dir}/${session}_reg.yaml" \
      --gt_pose_files "${scene_path}/test/poses/*.txt" \
      --output_dir "${method_dir}" \
      --session_id "${session}"
}

train_glace() {
  local scene="$1" scene_path="$2" method_dir="$3"
  local model="${method_dir}/model.pt"
  local pose_file="${method_dir}/poses_${scene}_post_train.txt"
  mkdir -p "${method_dir}"

  if maybe_skip "${pose_file}"; then
    log_status "$scene" "glace" "all" "skipped_existing" 0 "${method_dir}/skip.log"
    return 0
  fi

  if [[ "${GLACE_EXTRACT_FEATURES}" == "true" && ( ! -s "${scene_path}/train/features.npy" || ! -s "${scene_path}/test/features.npy" ) ]]; then
    run_logged_cwd "$scene" "glace" "extract_features" "${method_dir}/extract_features.log" "${GLACE_ROOT}" \
      "${CUDA_ENV[@]}" "${CONDA_RUN[@]}" python "${GLACE_ROOT}/datasets/extract_features.py" "${scene_path}" \
        --checkpoint "${GLACE_R2FORMER_CKPT}" \
        --batch_size "${GLACE_EXTRACT_BATCH_SIZE}" \
        --num_workers "${GLACE_EXTRACT_WORKERS}"
    [[ $? -eq 0 ]] || return 0
  fi

  run_logged_cwd "$scene" "glace" "train" "${method_dir}/train.log" "${GLACE_ROOT}" \
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

  [[ $? -eq 0 ]] || return 0

  run_logged_cwd "$scene" "glace" "eval" "${method_dir}/eval.log" "${GLACE_ROOT}" \
    "${CUDA_ENV[@]}" "${CONDA_RUN[@]}" python "${GLACE_ROOT}/test_ace.py" "${scene_path}" "${model}" \
      --session post_train \
      --image_resolution "${GLACE_IMAGE_RESOLUTION}" \
      --hypotheses "${HYPOTHESES}" \
      --render_flipped_portrait "${GLACE_RENDER_FLIPPED_PORTRAIT}"
}

write_run_config
printf "Run root: %s\n" "${RUN_ROOT}"
printf "Dataset : %s\n" "${WAYSPOTS_ROOT}"
printf "Scenes  : %s\n" "${SCENES}"
printf "Methods : %s\n" "${METHODS}"
printf "GPU_ID  : %s\n" "${GPU_ID:-<unset>}"
printf "Device  : train=%s eval=%s hypotheses=%s\n" "${DEVICE}" "${EVAL_DEVICE}" "${HYPOTHESES}"
printf "Dry run : %s\n" "${DRY_RUN}"
print_run_config

for scene in ${SCENES}; do
  scene_path="${WAYSPOTS_ROOT}/${scene}"
  if ! check_scene "${scene_path}"; then
    printf "Scene layout is incomplete: %s\n" "${scene_path}" | tee -a "${RUN_ROOT}/preflight_errors.log"
    log_status "$scene" "all" "preflight" "failed_scene_layout" 2 "${RUN_ROOT}/preflight_errors.log"
    [[ "${CONTINUE_ON_ERROR}" == "true" ]] && continue || exit 2
  fi

  has_method ace && train_ace "$scene" "$scene_path" "${RUN_ROOT}/${scene}/ace"
  summarize || true

  has_method dinoace && train_dinoace "$scene" "$scene_path" "${RUN_ROOT}/${scene}/dinoace"
  summarize || true

  has_method aceg25 && train_aceg25 "$scene" "$scene_path" "${RUN_ROOT}/${scene}/aceg25"
  summarize || true

  has_method glace && train_glace "$scene" "$scene_path" "${RUN_ROOT}/${scene}/glace"
  summarize || true
done

summarize || true
printf "Done. Summary: %s/summary.tsv and %s/summary.md\n" "${RUN_ROOT}" "${RUN_ROOT}"
