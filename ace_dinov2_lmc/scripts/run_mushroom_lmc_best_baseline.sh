#!/usr/bin/env bash
set -uo pipefail

# Extract MuSHRoom memories and train/evaluate our DINOv2+LMC method.
# The run keeps a copy of the current baseline summary and emits LMC results
# in the same summary format used by run_mushroom_baselines_all.sh.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="${ROOT_DIR}/ace_dinov2_lmc/scripts"
cd "${ROOT_DIR}"

CONDA_ENV="${CONDA_ENV:-mapanything}"
CONDA_RUN=(conda run --no-capture-output -n "${CONDA_ENV}")

MUSHROOM_ACE_ROOT="${MUSHROOM_ACE_ROOT:-/data/xwh/MuSHRoom_ace_colmap}"
MUSHROOM_WAI_ROOT="${MUSHROOM_WAI_ROOT:-/data/xwh/MuSHRoom_wai/kinect}"
BASELINE_RUN_ROOT="${BASELINE_RUN_ROOT:-${ROOT_DIR}/ace_dinov2_lmc/04_evaluation/mushroom_baselines/20260524_174735_4090_balanced}"
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/ace_dinov2_lmc/04_evaluation/mushroom_lmc/$(date +%Y%m%d_%H%M%S)_best_baseline}"
SCENES="${SCENES:-coffee_room classroom vr_room}"
GPU_ID="${GPU_ID:-1}"
DEVICE="${DEVICE:-cuda:${GPU_ID}}"
EVAL_DEVICE="${EVAL_DEVICE:-${DEVICE}}"
DRY_RUN="${DRY_RUN:-false}"
SKIP_EXISTING="${SKIP_EXISTING:-true}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-true}"

# Memory extraction: current best recorded BSE/ASB recipe from the indoor6/ACE-G comparisons.
N_VIEWS="${N_VIEWS:-40}"
VOXEL_SIZE="${VOXEL_SIZE:-0.05}"
UNIMODAL_THRESHOLD="${UNIMODAL_THRESHOLD:-0.02}"
POOL_MODE="${POOL_MODE:-bse}"
WAI_VIEW_MODE="${WAI_VIEW_MODE:-anchor_support}"
ASB_ADAPTIVE="${ASB_ADAPTIVE:-true}"
ASB_POST_REPAIR="${ASB_POST_REPAIR:-true}"
ASB_POSE_PRUNE="${ASB_POSE_PRUNE:-true}"
ASB_CANDIDATE_POOL_RATIO="${ASB_CANDIDATE_POOL_RATIO:-1.0}"

# LMC training: match the current DINOACE-compatible baseline surface, then add LMC.
TRAIN_PRESET="${TRAIN_PRESET:-memory_compare_ace_g_v2}"
LMC_MODE="${LMC_MODE:-global}"
LMC_IMAGE_RESOLUTION="${LMC_IMAGE_RESOLUTION:-518}"
LMC_TRAINING_BUFFER_SIZE="${LMC_TRAINING_BUFFER_SIZE:-4000000}"
LMC_BUFFER_SIZE_FINAL="${LMC_BUFFER_SIZE_FINAL:-8000000}"
LMC_SAMPLES_PER_IMAGE="${LMC_SAMPLES_PER_IMAGE:-512}"
LMC_BATCH_SIZE="${LMC_BATCH_SIZE:-10240}"
LMC_POST_TRAIN_HYPOTHESES="${LMC_POST_TRAIN_HYPOTHESES:-256}"
LMC_POST_TRAIN_SEED="${LMC_POST_TRAIN_SEED:-1305}"
LMC_EVAL_DETERMINISTIC="${LMC_EVAL_DETERMINISTIC:-true}"
LMC_BUFFER_ON_CPU="${LMC_BUFFER_ON_CPU:-true}"
LMC_BUFFER_ON_CPU_FINAL="${LMC_BUFFER_ON_CPU_FINAL:-true}"

STATUS_FILE="${RUN_ROOT}/status.tsv"
CONFIG_FILE="${RUN_ROOT}/run_config.tsv"
mkdir -p "${RUN_ROOT}"
if [[ ! -f "${STATUS_FILE}" ]]; then
  printf "timestamp\tscene\tmethod\tstage\tstatus\texit_code\ttrain_device\teval_device\tlog\n" > "${STATUS_FILE}"
fi

log_status() {
  local scene="$1" method="$2" stage="$3" status="$4" exit_code="$5" log_file="$6"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$(date +%Y-%m-%dT%H:%M:%S)" "$scene" "$method" "$stage" "$status" "$exit_code" \
    "$DEVICE" "$EVAL_DEVICE" "$log_file" >> "${STATUS_FILE}"
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
  return 1
}

write_config() {
  cat > "${CONFIG_FILE}" <<EOF
key	value
RUN_ROOT	${RUN_ROOT}
BASELINE_RUN_ROOT	${BASELINE_RUN_ROOT}
MUSHROOM_ACE_ROOT	${MUSHROOM_ACE_ROOT}
MUSHROOM_WAI_ROOT	${MUSHROOM_WAI_ROOT}
SCENES	${SCENES}
GPU_ID	${GPU_ID}
DEVICE	${DEVICE}
EVAL_DEVICE	${EVAL_DEVICE}
N_VIEWS	${N_VIEWS}
VOXEL_SIZE	${VOXEL_SIZE}
UNIMODAL_THRESHOLD	${UNIMODAL_THRESHOLD}
POOL_MODE	${POOL_MODE}
WAI_VIEW_MODE	${WAI_VIEW_MODE}
ASB_ADAPTIVE	${ASB_ADAPTIVE}
ASB_POST_REPAIR	${ASB_POST_REPAIR}
ASB_POSE_PRUNE	${ASB_POSE_PRUNE}
ASB_CANDIDATE_POOL_RATIO	${ASB_CANDIDATE_POOL_RATIO}
TRAIN_PRESET	${TRAIN_PRESET}
LMC_MODE	${LMC_MODE}
LMC_IMAGE_RESOLUTION	${LMC_IMAGE_RESOLUTION}
LMC_TRAINING_BUFFER_SIZE	${LMC_TRAINING_BUFFER_SIZE}
LMC_BUFFER_SIZE_FINAL	${LMC_BUFFER_SIZE_FINAL}
LMC_SAMPLES_PER_IMAGE	${LMC_SAMPLES_PER_IMAGE}
LMC_BATCH_SIZE	${LMC_BATCH_SIZE}
LMC_POST_TRAIN_HYPOTHESES	${LMC_POST_TRAIN_HYPOTHESES}
LMC_POST_TRAIN_SEED	${LMC_POST_TRAIN_SEED}
LMC_EVAL_DETERMINISTIC	${LMC_EVAL_DETERMINISTIC}
EOF
  if [[ -f "${BASELINE_RUN_ROOT}/summary.tsv" ]]; then
    cp "${BASELINE_RUN_ROOT}/summary.tsv" "${RUN_ROOT}/baseline_summary.tsv"
  fi
}

write_best_baseline() {
  if [[ ! -f "${RUN_ROOT}/baseline_summary.tsv" ]]; then
    return 0
  fi
  "${CONDA_RUN[@]}" python -c '
import csv, math, pathlib
root = pathlib.Path("'"${RUN_ROOT}"'")
rows = list(csv.DictReader((root / "baseline_summary.tsv").open(), delimiter="\t"))
best = {}
for row in rows:
    if row.get("status") != "ok":
        continue
    scene = row["scene"]
    score = (
        -float(row["5cm_5deg"]),
        float(row["median_cm"]),
        float(row["median_deg"]),
        row["method"],
    )
    if scene not in best or score < best[scene][0]:
        best[scene] = (score, row)
with (root / "best_baseline.tsv").open("w", newline="") as f:
    fields = ["scene", "method", "5cm_5deg", "median_cm", "median_deg", "output"]
    writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
    writer.writeheader()
    for scene in sorted(best):
        row = best[scene][1]
        writer.writerow({k: row.get(k, "") for k in fields})
'
}

latest_memory_for_scene() {
  local scene="$1"
  find "${RUN_ROOT}/memory/${scene}" -type f -name memory_bse.pt -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr | head -n 1 | cut -d' ' -f2-
}

extract_memory() {
  local scene="$1"
  local method_dir="${RUN_ROOT}/${scene}/lmc"
  local memory_log="${method_dir}/extract_memory.log"
  mkdir -p "${method_dir}"

  local existing
  existing="$(latest_memory_for_scene "${scene}" || true)"
  if [[ "${SKIP_EXISTING}" == "true" && -n "${existing}" && -s "${existing}" ]]; then
    printf "%s\n" "${existing}" > "${method_dir}/memory_path.txt"
    log_status "$scene" "lmc" "extract_memory" "skipped_existing" 0 "${memory_log}"
    return 0
  fi

  if [[ "${DRY_RUN}" == "true" ]]; then
    printf "%s\n" "${RUN_ROOT}/memory/${scene}/dryrun/memory_bse.pt" > "${method_dir}/memory_path.txt"
  fi

  run_logged "$scene" "lmc" "extract_memory" "${memory_log}" \
    env \
      "CUDA_VISIBLE_DEVICES=${GPU_ID}" \
      "GPU_ID=${GPU_ID}" \
      "ACE_DATA_ROOT=/home/xwh/data" \
      "DATASET_TYPE=mushroom" \
      "DATASET_LOADER=wai" \
      "MUSHROOM_WAI_ROOT=${MUSHROOM_WAI_ROOT}" \
      "SCENE_TRAIN=${scene}_train" \
      "SCENE_TEST=${scene}_test" \
      "OUTPUT_SCENE_NAME=${scene}" \
      "OUTPUT_ROOT=${RUN_ROOT}/memory" \
      "N_VIEWS=${N_VIEWS}" \
      "VOXEL_SIZE=${VOXEL_SIZE}" \
      "UNIMODAL_THRESHOLD=${UNIMODAL_THRESHOLD}" \
      "POOL_MODE=${POOL_MODE}" \
      "WAI_VIEW_MODE=${WAI_VIEW_MODE}" \
      "ASB_ADAPTIVE=${ASB_ADAPTIVE}" \
      "ASB_POST_REPAIR=${ASB_POST_REPAIR}" \
      "ASB_POSE_PRUNE=${ASB_POSE_PRUNE}" \
      "ASB_CANDIDATE_POOL_RATIO=${ASB_CANDIDATE_POOL_RATIO}" \
      "${CONDA_RUN[@]}" bash "${ROOT_DIR}/ace_dinov2_lmc/memory_extraction/extract_memory.sh"

  [[ $? -eq 0 ]] || return 1
  if [[ "${DRY_RUN}" == "true" ]]; then
    return 0
  fi
  local memory_path
  memory_path="$(latest_memory_for_scene "${scene}" || true)"
  if [[ -z "${memory_path}" || ! -s "${memory_path}" ]]; then
    log_status "$scene" "lmc" "extract_memory" "failed_missing_memory" 2 "${memory_log}"
    return 1
  fi
  printf "%s\n" "${memory_path}" > "${method_dir}/memory_path.txt"
}

train_lmc() {
  local scene="$1"
  local scene_path="${MUSHROOM_ACE_ROOT}/${scene}"
  local method_dir="${RUN_ROOT}/${scene}/lmc"
  local pose_file="${method_dir}/poses_${scene}_post_train.txt"
  local train_log="${method_dir}/train.log"
  mkdir -p "${method_dir}"

  if [[ "${SKIP_EXISTING}" == "true" && -s "${pose_file}" ]]; then
    log_status "$scene" "lmc" "train_eval" "skipped_existing" 0 "${train_log}"
    return 0
  fi

  local memory_path
  memory_path="$(cat "${method_dir}/memory_path.txt")"
  local dino_path
  dino_path="$(resolve_dinov2_path || true)"
  if [[ -z "${dino_path}" ]]; then
    log_status "$scene" "lmc" "preflight" "failed_missing_dinov2" 2 "${train_log}"
    return 1
  fi

  run_logged "$scene" "lmc" "train_eval" "${train_log}" \
    "${CONDA_RUN[@]}" python "${ROOT_DIR}/ace_dinov2_lmc/train_ace_dinov2_lmc.py" \
      "${scene_path}" \
      "${scene}_mushroom_lmc.pt" \
      --train_preset "${TRAIN_PRESET}" \
      --data_backend ace \
      --device "${DEVICE}" \
      --post_train_eval_device "${EVAL_DEVICE}" \
      --use_lmc True \
      --memory_path "${memory_path}" \
      --lmc_mode "${LMC_MODE}" \
      --lmc_fps_start_policy farthest_from_center \
      --experiment_root "${method_dir}/exp" \
      --experiment_subdir "best_baseline_matched" \
      --image_resolution "${LMC_IMAGE_RESOLUTION}" \
      --training_buffer_size "${LMC_TRAINING_BUFFER_SIZE}" \
      --buffer_size_final "${LMC_BUFFER_SIZE_FINAL}" \
      --samples_per_image "${LMC_SAMPLES_PER_IMAGE}" \
      --batch_size "${LMC_BATCH_SIZE}" \
      --buffer_on_cpu "${LMC_BUFFER_ON_CPU}" \
      --buffer_on_cpu_final "${LMC_BUFFER_ON_CPU_FINAL}" \
      --buffer_sample_valid_coords True \
      --buffer_valid_coord_sample_ratio 1.0 \
      --buffer_valid_coord_neighbor_radius 1 \
      --buffer_valid_coord_neighbor_mode cross \
      --c1_aux_ref_loss_weight 0.0 \
      --c1_aux_depth_root "${MUSHROOM_WAI_ROOT}/${scene}_train" \
      --c1_aux_depth_kind gt_depth \
      --dinov2_path "${dino_path}" \
      --eval_deterministic "${LMC_EVAL_DETERMINISTIC}" \
      --post_train_eval_seeds "${LMC_POST_TRAIN_SEED}" \
      --post_train_hypotheses "${LMC_POST_TRAIN_HYPOTHESES}"

  [[ $? -eq 0 ]] || return 1
  if [[ "${DRY_RUN}" == "true" ]]; then
    return 0
  fi

  local run_dir
  run_dir="$(find "${method_dir}/exp" -type f -name "poses_${scene}_post_train.txt" -printf '%T@ %h\n' 2>/dev/null \
    | sort -nr | head -n 1 | cut -d' ' -f2-)"
  if [[ -z "${run_dir}" ]]; then
    log_status "$scene" "lmc" "collect" "failed_missing_pose_log" 2 "${train_log}"
    return 1
  fi
  cp "${run_dir}/poses_${scene}_post_train.txt" "${pose_file}"
  cp "${run_dir}/post_train_eval.txt" "${method_dir}/post_train_eval.txt" 2>/dev/null || true
  printf "%s\n" "${run_dir}" > "${method_dir}/run_dir.txt"
}

summarize_lmc() {
  "${CONDA_RUN[@]}" python "${SCRIPT_DIR}/summarize_mushroom_baselines.py" \
    --run-root "${RUN_ROOT}" \
    --dataset-root "${MUSHROOM_ACE_ROOT}" \
    --scenes ${SCENES} \
    --methods lmc
}

write_config
write_best_baseline

printf "Run root: %s\n" "${RUN_ROOT}"
printf "Baseline: %s\n" "${BASELINE_RUN_ROOT}"
printf "Scenes  : %s\n" "${SCENES}"
printf "Device  : train=%s eval=%s gpu_id=%s\n" "${DEVICE}" "${EVAL_DEVICE}" "${GPU_ID}"
printf "Memory  : %sv %s/%s asb_adaptive=%s post_repair=%s pose_prune=%s\n" \
  "${N_VIEWS}" "${POOL_MODE}" "${VOXEL_SIZE}" "${ASB_ADAPTIVE}" "${ASB_POST_REPAIR}" "${ASB_POSE_PRUNE}"
printf "LMC     : preset=%s buffer=%s final=%s samples=%s batch=%s res=%s\n" \
  "${TRAIN_PRESET}" "${LMC_TRAINING_BUFFER_SIZE}" "${LMC_BUFFER_SIZE_FINAL}" \
  "${LMC_SAMPLES_PER_IMAGE}" "${LMC_BATCH_SIZE}" "${LMC_IMAGE_RESOLUTION}"

for scene in ${SCENES}; do
  extract_memory "$scene"
  if [[ $? -ne 0 ]]; then
    [[ "${CONTINUE_ON_ERROR}" == "true" ]] && continue || exit 1
  fi

  train_lmc "$scene"
  if [[ $? -ne 0 ]]; then
    summarize_lmc || true
    [[ "${CONTINUE_ON_ERROR}" == "true" ]] && continue || exit 1
  fi
  summarize_lmc || true
done

summarize_lmc || true
printf "Done. LMC summary: %s/summary.tsv and %s/summary.md\n" "${RUN_ROOT}" "${RUN_ROOT}"
