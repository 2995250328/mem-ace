#!/usr/bin/env bash
set -euo pipefail

# Single-GPU sequential Cambridge runner for the official GLACE code path.
# This mirrors /home/xwh/project/glace/scripts/train_cambridge.sh as closely as
# possible, except that it uses one visible GPU and writes outputs under /data.

GLACE_ROOT="${GLACE_ROOT:-/home/xwh/project/glace}"
CAMBRIDGE_ROOT="${CAMBRIDGE_ROOT:-/data/xwh/Cambridge}"
RUN_ROOT="${RUN_ROOT:-/data/xwh/ace_dinov2_lmc/04_evaluation/glace_official_single_gpu1_seq_20260626}"
GPU="${GPU:-1}"

SCENES="${SCENES:-Cambridge_GreatCourt Cambridge_KingsCollege Cambridge_OldHospital Cambridge_ShopFacade Cambridge_StMarysChurch}"

TRAINING_BUFFER_SIZE="${TRAINING_BUFFER_SIZE:-16000000}"
SAMPLES_PER_IMAGE="${SAMPLES_PER_IMAGE:-1024}"
BATCH_SIZE="${BATCH_SIZE:-40960}"
MAX_ITERATIONS="${MAX_ITERATIONS:-30000}"
IMAGE_RESOLUTION="${IMAGE_RESOLUTION:-480}"
NUM_HEAD_BLOCKS="${NUM_HEAD_BLOCKS:-2}"
HYPOTHESES="${HYPOTHESES:-64}"

CONDA_ENV="${CONDA_ENV:-mapanything}"

OUT_DIR="${RUN_ROOT}/output/Cambridge"
LOG_DIR="${RUN_ROOT}/logs"
STATUS_FILE="${RUN_ROOT}/status.tsv"
CONFIG_FILE="${RUN_ROOT}/run_config.tsv"

mkdir -p "${OUT_DIR}" "${LOG_DIR}"

if [[ ! -s "${STATUS_FILE}" ]]; then
  printf "timestamp\tscene\tstage\tstatus\texit_code\tgpu\tlog\n" > "${STATUS_FILE}"
fi

cat > "${CONFIG_FILE}" <<EOF
GLACE_ROOT	${GLACE_ROOT}
CAMBRIDGE_ROOT	${CAMBRIDGE_ROOT}
RUN_ROOT	${RUN_ROOT}
GPU	${GPU}
SCENES	${SCENES}
TRAINING_BUFFER_SIZE	${TRAINING_BUFFER_SIZE}
SAMPLES_PER_IMAGE	${SAMPLES_PER_IMAGE}
BATCH_SIZE	${BATCH_SIZE}
MAX_ITERATIONS	${MAX_ITERATIONS}
IMAGE_RESOLUTION	${IMAGE_RESOLUTION}
NUM_HEAD_BLOCKS	${NUM_HEAD_BLOCKS}
HYPOTHESES	${HYPOTHESES}
RENDER_FLIPPED_PORTRAIT	False
CONDA_ENV	${CONDA_ENV}
EOF

log_status() {
  local scene="$1"
  local stage="$2"
  local status="$3"
  local exit_code="$4"
  local log_file="$5"
  printf "%s\t%s\t%s\t%s\t%s\tcuda:%s\t%s\n" \
    "$(date -Iseconds)" "${scene}" "${stage}" "${status}" "${exit_code}" "${GPU}" "${log_file}" >> "${STATUS_FILE}"
}

run_scene() {
  local scene="$1"
  local scene_path="${CAMBRIDGE_ROOT}/${scene}"
  local model="${OUT_DIR}/${scene}.pt"
  local train_log="${LOG_DIR}/train_${scene}.log"
  local eval_log="${LOG_DIR}/eval_${scene}.log"

  if [[ ! -d "${scene_path}/train" || ! -d "${scene_path}/test" ]]; then
    echo "ERROR: missing scene split dirs: ${scene_path}" | tee -a "${train_log}" >&2
    log_status "${scene}" "preflight" "failed_missing_scene" 2 "${train_log}"
    return 2
  fi
  if [[ ! -s "${scene_path}/train/features.npy" || ! -s "${scene_path}/test/features.npy" ]]; then
    echo "ERROR: missing GLACE features.npy for ${scene_path}" | tee -a "${train_log}" >&2
    log_status "${scene}" "preflight" "failed_missing_features" 2 "${train_log}"
    return 2
  fi

  log_status "${scene}" "train" "started" 0 "${train_log}"
  (
    cd "${GLACE_ROOT}"
    echo "scene=${scene}"
    echo "model=${model}"
    echo "gpu=${GPU}"
    echo "command=train official GLACE Cambridge single-GPU"
    CUDA_VISIBLE_DEVICES="${GPU}" conda run --no-capture-output -n "${CONDA_ENV}" \
      torchrun --standalone --nnodes 1 --nproc-per-node 1 "${GLACE_ROOT}/train_ace.py" \
        "${scene_path}" "${model}" \
        --num_head_blocks "${NUM_HEAD_BLOCKS}" \
        --training_buffer_size "${TRAINING_BUFFER_SIZE}" \
        --samples_per_image "${SAMPLES_PER_IMAGE}" \
        --batch_size "${BATCH_SIZE}" \
        --max_iterations "${MAX_ITERATIONS}" \
        --image_resolution "${IMAGE_RESOLUTION}" \
        --render_flipped_portrait False
  ) 2>&1 | tee "${train_log}"
  local train_ec=${PIPESTATUS[0]}
  if [[ ${train_ec} -ne 0 ]]; then
    log_status "${scene}" "train" "failed" "${train_ec}" "${train_log}"
    return "${train_ec}"
  fi
  log_status "${scene}" "train" "completed" 0 "${train_log}"

  log_status "${scene}" "eval" "started" 0 "${eval_log}"
  (
    cd "${GLACE_ROOT}"
    CUDA_VISIBLE_DEVICES="${GPU}" conda run --no-capture-output -n "${CONDA_ENV}" \
      python "${GLACE_ROOT}/test_ace.py" "${scene_path}" "${model}" \
        --image_resolution "${IMAGE_RESOLUTION}" \
        --hypotheses "${HYPOTHESES}" \
        --render_flipped_portrait False
  ) 2>&1 | tee "${eval_log}"
  local eval_ec=${PIPESTATUS[0]}
  if [[ ${eval_ec} -ne 0 ]]; then
    log_status "${scene}" "eval" "failed" "${eval_ec}" "${eval_log}"
    return "${eval_ec}"
  fi
  log_status "${scene}" "eval" "completed" 0 "${eval_log}"
}

for scene in ${SCENES}; do
  run_scene "${scene}"
done

for scene in ${SCENES}; do
  log_file="${OUT_DIR}/test_${scene}_.txt"
  if [[ -s "${log_file}" ]]; then
    echo "${scene}: $(tail -n 1 "${log_file}")"
  fi
done
