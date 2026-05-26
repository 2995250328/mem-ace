#!/usr/bin/env bash
set -euo pipefail

# Step19 conservative RIO10 migration runner.
# Contract:
#   - memory extraction is WAI/MapAnything, but training is ACE backend;
#   - true global ACE-G + per_iter S1, selected_key_concat_value, value_only_raw;
#   - sparse depth is used only for guided buffer sampling in run B;
#   - no auxiliary sparse-depth/reference supervision.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACE_DEPTH_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"

SCENE_ROOT="${SCENE_ROOT:-/data/xwh/RIO10_ace/scene01_seq01_01}"
SCENE_TAG="${SCENE_TAG:-scene01_seq01_01}"
IMAGE_RESOLUTION="${IMAGE_RESOLUTION:-518}"
BASE_IMAGE_RESOLUTION="${BASE_IMAGE_RESOLUTION:-518}"
BASE_SAMPLES_PER_IMAGE="${BASE_SAMPLES_PER_IMAGE:-384}"
BASE_TRAINING_BUFFER_SIZE="${BASE_TRAINING_BUFFER_SIZE:-2560000}"
BASE_BUFFER_SIZE_FINAL="${BASE_BUFFER_SIZE_FINAL:-7680000}"
SCALE_SAMPLING_BY_RES="${SCALE_SAMPLING_BY_RES:-true}"

scaled_value() {
  local base="$1"
  if [[ "${SCALE_SAMPLING_BY_RES,,}" =~ ^(true|1|yes|y|on)$ ]]; then
    echo $(( (base * IMAGE_RESOLUTION * IMAGE_RESOLUTION + (BASE_IMAGE_RESOLUTION * BASE_IMAGE_RESOLUTION / 2)) / (BASE_IMAGE_RESOLUTION * BASE_IMAGE_RESOLUTION) ))
  else
    echo "${base}"
  fi
}

SAMPLES_PER_IMAGE="${SAMPLES_PER_IMAGE:-$(scaled_value "${BASE_SAMPLES_PER_IMAGE}")}"
TRAINING_BUFFER_SIZE="${TRAINING_BUFFER_SIZE:-${BASE_TRAINING_BUFFER_SIZE}}"
BUFFER_SIZE_FINAL="${BUFFER_SIZE_FINAL:-${BASE_BUFFER_SIZE_FINAL}}"
BUFFER_ON_CPU="${BUFFER_ON_CPU:-auto}"
if [[ "${BUFFER_ON_CPU}" == "auto" ]]; then
  if [[ "${IMAGE_RESOLUTION}" != "${BASE_IMAGE_RESOLUTION}" ]]; then
    BUFFER_ON_CPU="true"
  else
    BUFFER_ON_CPU="false"
  fi
fi
BUFFER_ON_CPU_FINAL="${BUFFER_ON_CPU_FINAL:-true}"

RES_SUFFIX=""
if [[ "${IMAGE_RESOLUTION}" != "518" ]]; then
  RES_SUFFIX="_res${IMAGE_RESOLUTION}"
fi
MEMORY_PATH="${MEMORY_PATH:-${REPO_ROOT}/memory_extraction/04_evaluation/memory_extract/rio10/scene01_seq01_01_train${RES_SUFFIX}/memory_bse.pt}"
SPARSE_DEPTH_ROOT="${SPARSE_DEPTH_ROOT:-/data/xwh/RIO10_sparse_depth}"
DINOV2_PATH="${DINOV2_PATH:-/home/xwh/data/checkpoints/dinov2_vitl14_pretrain.pth}"
EXPERIMENT_SUBDIR="${EXPERIMENT_SUBDIR:-rio10_conservative_indoor6_migration${RES_SUFFIX}}"
RUN_TAG="${RUN_TAG:-${DATE_TAG:-$(date +%Y%m%d_%H%M%S)}}"
RUNS_STR="${RUNS_STR:-A B}"
GPUS_STR="${GPUS_STR:-0 1}"
DRY_RUN="${DRY_RUN:-false}"
GUIDED_RATIO="${GUIDED_RATIO:-1.0}"
GUIDED_RATIO_TAG="${GUIDED_RATIO_TAG:-r${GUIDED_RATIO//./}}"

PIDS=()

read -r -a RUNS <<< "${RUNS_STR}"
read -r -a GPUS <<< "${GPUS_STR}"
if [ "${#RUNS[@]}" -gt "${#GPUS[@]}" ]; then
  echo "ERROR: RUNS_STR has ${#RUNS[@]} runs but GPUS_STR has ${#GPUS[@]} GPUs." >&2
  exit 2
fi

common=(
  python ace_dinov2_lmc/train_ace_dinov2_lmc.py
  "${SCENE_ROOT}"
  __OUTPUT_PT__
  --run_name __RUN_NAME__
  --train_preset rio10_conservative_ace_g_v1
  --data_backend ace
  --device __DEVICE__
  --post_train_eval_device __DEVICE__
  --use_lmc True
  --memory_path "${MEMORY_PATH}"
  --dinov2_path "${DINOV2_PATH}"
  --experiment_subdir "${EXPERIMENT_SUBDIR}"
  --image_resolution "${IMAGE_RESOLUTION}"
  --samples_per_image "${SAMPLES_PER_IMAGE}"
  --training_buffer_size "${TRAINING_BUFFER_SIZE}"
  --buffer_size_final "${BUFFER_SIZE_FINAL}"
  --buffer_on_cpu "${BUFFER_ON_CPU}"
  --buffer_on_cpu_final "${BUFFER_ON_CPU_FINAL}"
  --lmc_iterations 28
  --c1_aux_ref_loss_weight 0.0
)

run_one() {
  local run="$1"
  local gpu="$2"
  local run_name output_pt device
  device="cuda:${gpu}"
  case "$run" in
    A|a|noguided|no_guided)
      run_name="rio10_${SCENE_TAG}_FGPI_step19_A_noguided${RES_SUFFIX}_${RUN_TAG}"
      output_pt="${run_name}.pt"
      extra=(--buffer_sample_valid_coords False)
      ;;
    B|b|guided|sparse_guided)
      run_name="rio10_${SCENE_TAG}_FGPI_step19_B_sparseguided_${GUIDED_RATIO_TAG}${RES_SUFFIX}_${RUN_TAG}"
      output_pt="${run_name}.pt"
      extra=(
        --buffer_sample_valid_coords True
        --buffer_valid_coord_sample_ratio "${GUIDED_RATIO}"
        --buffer_valid_coord_neighbor_radius 1
        --buffer_valid_coord_neighbor_mode cross
        --c1_aux_depth_root "${SPARSE_DEPTH_ROOT}"
        --c1_aux_depth_kind sparse_depth
      )
      ;;
    *)
      echo "ERROR: unknown run '${run}'. Use A/B." >&2
      exit 2
      ;;
  esac

  cmd=("${common[@]}")
  for i in "${!cmd[@]}"; do
    cmd[$i]="${cmd[$i]//__OUTPUT_PT__/${output_pt}}"
    cmd[$i]="${cmd[$i]//__RUN_NAME__/${run_name}}"
    cmd[$i]="${cmd[$i]//__DEVICE__/${device}}"
  done
  cmd+=("${extra[@]}")

  printf '\n[RIO10-Step19] run=%s gpu=%s name=%s\n' "$run" "$gpu" "$run_name"
  printf '%q ' "${cmd[@]}"
  printf '\n'
  if [ "${DRY_RUN}" != "true" ]; then
    "${cmd[@]}" &
    PIDS+=("$!")
  fi
}

printf '[RIO10-Train] image_resolution=%s scale_sampling=%s samples_per_image=%s train_buffer=%s final_buffer=%s buffer_on_cpu=%s guided_ratio=%s memory=%s\n' \
  "${IMAGE_RESOLUTION}" "${SCALE_SAMPLING_BY_RES}" "${SAMPLES_PER_IMAGE}" "${TRAINING_BUFFER_SIZE}" "${BUFFER_SIZE_FINAL}" "${BUFFER_ON_CPU}" "${GUIDED_RATIO}" "${MEMORY_PATH}"

cd "${ACE_DEPTH_ROOT}"
for idx in "${!RUNS[@]}"; do
  run_one "${RUNS[$idx]}" "${GPUS[$idx]}"
done

if [ "${DRY_RUN}" != "true" ]; then
  status=0
  for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  exit "$status"
fi
