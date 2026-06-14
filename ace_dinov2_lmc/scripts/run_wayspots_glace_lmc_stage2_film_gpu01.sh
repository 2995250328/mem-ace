#!/usr/bin/env bash
set -euo pipefail

# Run Wayspots ACE-FCN-LMC Stage2 with SQ-safe GLACE FiLM conditioning on GPUs 0/1.
# It reuses existing memory + Stage1 checkpoints from the 20260530 suite.
# Also supports memory controls via FEATURE_MODES_STR="glace zero random".
# Run from /home/xwh/project/ace_depth:
#   bash ace_dinov2_lmc/scripts/run_wayspots_glace_lmc_stage2_film_gpu01.sh
#   DRY_RUN=true KS_STR="64 256" FEATURE_MODES_STR="glace zero random" bash ...

ROOT_DIR="${ROOT_DIR:-/home/xwh/project/ace_depth}"
cd "${ROOT_DIR}"

CONDA_ENV="${CONDA_ENV:-mapanything}"
PYTHON_BIN="${PYTHON_BIN:-python}"
WAYSPOTS_ROOT="${WAYSPOTS_ROOT:-/data/xwh/Wayspots}"
SUITE_ROOT="${SUITE_ROOT:-/data/xwh/ace_dinov2_lmc/04_evaluation/wayspots_ace_fcn_lmc_suite/20260530_181745}"
SCENES_STR="${SCENES_STR:-wayspots_bears wayspots_cubes wayspots_inscription wayspots_lawn wayspots_map wayspots_squarebench wayspots_tendrils wayspots_therock}"
GPUS_STR="${GPUS_STR:-0 1}"
KS_STR="${KS_STR:-64}"
FEATURE_MODES_STR="${FEATURE_MODES_STR:-glace}"
RUN_TAG="${RUN_TAG:-stage2_film_unified}"
DRY_RUN="${DRY_RUN:-false}"
SKIP_EXISTING="${SKIP_EXISTING:-true}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-true}"

ACE_ENCODER_PATH="${ACE_ENCODER_PATH:-${ROOT_DIR}/ace_encoder_pretrained.pt}"
GLACE_FEAT_NAME="${GLACE_FEAT_NAME:-features.npy}"
LMC_ITERATIONS="${LMC_ITERATIONS:-12}"
IMAGE_RESOLUTION="${IMAGE_RESOLUTION:-512}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
TRAINING_BUFFER_SIZE="${TRAINING_BUFFER_SIZE:-2800000}"
BUFFER_SIZE_FINAL="${BUFFER_SIZE_FINAL:-7600000}"
SAMPLES_PER_IMAGE="${SAMPLES_PER_IMAGE:-512}"
POST_TRAIN_HYPOTHESES="${POST_TRAIN_HYPOTHESES:-256}"
POST_TRAIN_EVAL_SEEDS_STR="${POST_TRAIN_EVAL_SEEDS_STR:-1305 2026 4242}"
USE_CUBES_REDUCED_K1024="${USE_CUBES_REDUCED_K1024:-true}"
RANDOM_GLOBAL_SEED="${RANDOM_GLOBAL_SEED:-20260531}"

read -r -a SCENES <<< "${SCENES_STR}"
read -r -a GPUS <<< "${GPUS_STR}"
read -r -a KS <<< "${KS_STR}"
read -r -a FEATURE_MODES <<< "${FEATURE_MODES_STR}"
read -r -a POST_TRAIN_EVAL_SEEDS <<< "${POST_TRAIN_EVAL_SEEDS_STR}"

for gpu in "${GPUS[@]}"; do
  if [[ "${gpu}" != "0" && "${gpu}" != "1" ]]; then
    echo "ERROR: this script is constrained to GPUs 0/1, got ${gpu}" >&2
    exit 2
  fi
done

STATUS_FILE="${SUITE_ROOT}/${RUN_TAG}_gpu01_status.tsv"
mkdir -p "${SUITE_ROOT}"
if [[ ! -f "${STATUS_FILE}" ]]; then
  printf 'timestamp\tscene\tK\tfeature_mode\tstatus\texit_code\tgpu\tlog\n' > "${STATUS_FILE}"
fi

memory_dir_for() {
  local scene="$1" k="$2"
  if [[ "${scene}" == "wayspots_cubes" && "${k}" == "1024" && "${USE_CUBES_REDUCED_K1024}" == "true" ]]; then
    printf 'memory_cubes_k1024_vox010_max20k\n'
  else
    printf 'memory\n'
  fi
}

stage1_subdir_for() {
  local scene="$1" k="$2"
  if [[ "${k}" == "64" ]]; then
    printf 'stage1_local_ace_memory_it12\n'
  elif [[ "${scene}" == "wayspots_cubes" && "${k}" == "1024" && "${USE_CUBES_REDUCED_K1024}" == "true" ]]; then
    printf 'stage1_local_ace_memory_it12_K1024_cubes_vox010_max20k\n'
  else
    printf 'stage1_local_ace_memory_it12_K%s\n' "${k}"
  fi
}

experiment_subdir_for() {
  local scene="$1" k="$2" feature="$3"
  if [[ "${k}" == "64" ]]; then
    printf '%s_%s_it%s\n' "${RUN_TAG}" "${feature}" "${LMC_ITERATIONS}"
  else
    printf '%s_%s_it%s_K%s\n' "${RUN_TAG}" "${feature}" "${LMC_ITERATIONS}" "${k}"
  fi
}

find_stage1_ckpt() {
  local scene="$1" k="$2"
  local stage1_subdir
  stage1_subdir="$(stage1_subdir_for "${scene}" "${k}")"
  find "${SUITE_ROOT}/${stage1_subdir}" -type f -name "best_K${k}_it${LMC_ITERATIONS}_ace_fcn_local_stage1.pt" -path "*${scene}*" 2>/dev/null | sort | tail -n 1 || true
}

log_status() {
  local scene="$1" k="$2" feature="$3" status="$4" exit_code="$5" gpu="$6" log_file="$7"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date +%Y-%m-%dT%H:%M:%S)" "${scene}" "${k}" "${feature}" "${status}" "${exit_code}" "${gpu}" "${log_file}" >> "${STATUS_FILE}"
}

print_cmd() { printf '%q ' "$@"; printf '\n'; }

run_one() {
  local scene="$1" k="$2" feature="$3" gpu="$4"
  local scene_root="${WAYSPOTS_ROOT}/${scene}"
  local memory_dir stage1_ckpt experiment_subdir output_suffix log_file aux_depth_dir memory_path existing
  memory_dir="$(memory_dir_for "${scene}" "${k}")"
  memory_path="${SUITE_ROOT}/${memory_dir}/${scene}/memory_ace_fcn_sparse_sp_r4.pt"
  stage1_ckpt="$(find_stage1_ckpt "${scene}" "${k}")"
  experiment_subdir="$(experiment_subdir_for "${scene}" "${k}" "${feature}")"
  output_suffix="ace_fcn_glace_film_stage2_${feature}.pt"
  aux_depth_dir="${scene_root}/train/sparse_depth"
  log_file="${SUITE_ROOT}/${experiment_subdir}/train_${scene}.log"
  mkdir -p "$(dirname "${log_file}")"

  if [[ ! -s "${memory_path}" ]]; then
    echo "ERROR: missing memory: ${memory_path}" | tee -a "${log_file}"
    log_status "${scene}" "${k}" "${feature}" failed_missing_memory 2 "cuda:${gpu}" "${log_file}"
    return 2
  fi
  if [[ -z "${stage1_ckpt}" || ! -s "${stage1_ckpt}" ]]; then
    echo "ERROR: missing Stage1 checkpoint for scene=${scene} K=${k}" | tee -a "${log_file}"
    log_status "${scene}" "${k}" "${feature}" failed_missing_stage1 2 "cuda:${gpu}" "${log_file}"
    return 2
  fi
  existing="$(find "${SUITE_ROOT}/${experiment_subdir}" -type f -name "best_K${k}_it${LMC_ITERATIONS}_${output_suffix}" -path "*${scene}*" 2>/dev/null | sort | tail -n 1 || true)"
  if [[ -n "${existing}" && "${SKIP_EXISTING}" == "true" ]]; then
    log_status "${scene}" "${k}" "${feature}" skipped_existing 0 "cuda:${gpu}" "${log_file}"
    return 0
  fi

  local cmd=(
    conda run --no-capture-output -n "${CONDA_ENV}" "${PYTHON_BIN}" ace_dinov2_lmc/train_ace_dinov2_lmc.py
    "${scene_root}" "${output_suffix}"
    --model_backend ace_fcn_lmc
    --data_backend ace
    --use_lmc True
    --lmc_flow ace_g
    --memory_path "${memory_path}"
    --use_scale_token False
    --ace_encoder_path "${ACE_ENCODER_PATH}"
    --ace_lmc_global_head_mode glace_film
    --ace_lmc_local_checkpoint_path "${stage1_ckpt}"
    --ace_lmc_freeze_local_stack True
    --ace_lmc_global_feature_mode "${feature}"
    --ace_lmc_global_gate_init 1.0
    --ace_lmc_global_gate_learnable False
    --ace_lmc_global_gate_max 0.0
    --ace_lmc_global_gate_l1_weight 0.0
    --ace_lmc_random_global_seed "${RANDOM_GLOBAL_SEED}"
    --glace_feat_name "${GLACE_FEAT_NAME}"
    --device "cuda:${gpu}"
    --post_train_eval_device "cuda:${gpu}"
    --experiment_root "${SUITE_ROOT}"
    --experiment_subdir "${experiment_subdir}"
    --lmc_iterations "${LMC_ITERATIONS}"
    --num_latent_tokens "${k}"
    --ace_g_fusion_in_s2 True
    --ace_g_cross_iter_eval True
    --s1_use_buffer True
    --s1_loss_mode sample_per_image
    --s1_buffer_refill_mode full
    --image_resolution "${IMAGE_RESOLUTION}"
    --batch_size "${BATCH_SIZE}"
    --training_buffer_size "${TRAINING_BUFFER_SIZE}"
    --buffer_size_final "${BUFFER_SIZE_FINAL}"
    --buffer_on_cpu True
    --buffer_on_cpu_final True
    --samples_per_image "${SAMPLES_PER_IMAGE}"
    --buffer_sample_valid_coords True
    --buffer_valid_coord_sample_ratio 1.0
    --buffer_valid_coord_neighbor_radius 1
    --buffer_valid_coord_neighbor_mode cross
    --c1_aux_depth_root "${aux_depth_dir}"
    --c1_aux_depth_kind sparse_depth
    --post_train_eval_seeds "${POST_TRAIN_EVAL_SEEDS[@]}"
    --post_train_hypotheses "${POST_TRAIN_HYPOTHESES}"
  )

  {
    printf '[%s] scene=%s K=%s feature=%s gpu=%s\n' "$(date)" "${scene}" "${k}" "${feature}" "${gpu}"
    printf 'memory=%s\nstage1=%s\nexperiment_subdir=%s\n' "${memory_path}" "${stage1_ckpt}" "${experiment_subdir}"
    print_cmd "${cmd[@]}"
  } | tee -a "${log_file}"

  if [[ "${DRY_RUN}" == "true" ]]; then
    log_status "${scene}" "${k}" "${feature}" dry_run 0 "cuda:${gpu}" "${log_file}"
    return 0
  fi

  "${cmd[@]}" 2>&1 | tee -a "${log_file}"
  local exit_code=${PIPESTATUS[0]}
  if [[ ${exit_code} -eq 0 ]]; then
    log_status "${scene}" "${k}" "${feature}" ok 0 "cuda:${gpu}" "${log_file}"
  else
    log_status "${scene}" "${k}" "${feature}" failed "${exit_code}" "cuda:${gpu}" "${log_file}"
    [[ "${CONTINUE_ON_ERROR}" == "true" ]] || exit "${exit_code}"
  fi
  return "${exit_code}"
}

tasks=()
for k in "${KS[@]}"; do
  for feature in "${FEATURE_MODES[@]}"; do
    for scene in "${SCENES[@]}"; do
      tasks+=("${scene}:${k}:${feature}")
    done
  done
done

printf 'Suite root: %s\n' "${SUITE_ROOT}"
printf 'Scenes    : %s\n' "${SCENES_STR}"
printf 'K values  : %s\n' "${KS_STR}"
printf 'Features  : %s\n' "${FEATURE_MODES_STR}"
printf 'GPUs      : %s\n' "${GPUS_STR}"
printf 'Dry run   : %s\n' "${DRY_RUN}"

pids=()
labels=()
for i in "${!tasks[@]}"; do
  IFS=':' read -r scene k feature <<< "${tasks[$i]}"
  gpu="${GPUS[$(( i % ${#GPUS[@]} ))]}"
  run_one "${scene}" "${k}" "${feature}" "${gpu}" &
  pids+=("$!")
  labels+=("${scene}:K${k}:${feature}:gpu${gpu}")
  if (( ${#pids[@]} >= ${#GPUS[@]} )); then
    failed=0
    for j in "${!pids[@]}"; do wait "${pids[$j]}" || failed=1; done
    pids=(); labels=()
    [[ ${failed} -eq 0 || "${CONTINUE_ON_ERROR}" == "true" ]] || exit 1
  fi
done
for pid in "${pids[@]}"; do wait "${pid}" || true; done

printf 'Done. Status: %s\n' "${STATUS_FILE}"
