#!/usr/bin/env bash
set -uo pipefail

# Scheduled quick fusion launcher for ACE-DINOv2-LMC.
#
# Default policy:
#   - legacy-safe: one train job per physical GPU, inline eval, original worker counts
#   - 2x per-GPU train concurrency is opt-in via MAX_JOBS_PER_GPU=2 and EVAL_MODE=deferred
#   - CUDA_VISIBLE_DEVICES is unset before every Python command
#   - physical devices are passed as --device cuda:${gpu}
#
# Run this script from tmux for real experiments.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

CONDA_ENV="${CONDA_ENV:-mapanything}"
PYTHON_BIN="${PYTHON_BIN:-python}"
WAYSPOTS_ROOT="${WAYSPOTS_ROOT:-/data/xwh/Wayspots}"
MEMORY_ROOT="${MEMORY_ROOT:-/data/xwh/ace_dinov2_lmc/04_evaluation/wayspots_ace_fcn_lmc_suite/20260530_181745/memory}"
RUN_ROOT="${RUN_ROOT:-/data/xwh/ace_dinov2_lmc/04_evaluation/quick_fusion_v3_adapter_matrix_scheduled_$(date +%Y%m%d_%H%M%S)}"
ACE_ENCODER_PATH="${ACE_ENCODER_PATH:-${ROOT_DIR}/ace_encoder_pretrained.pt}"

IMAGE_RESOLUTION="${IMAGE_RESOLUTION:-512}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
SAMPLES_PER_IMAGE="${SAMPLES_PER_IMAGE:-512}"
NUM_LATENT_TOKENS="${NUM_LATENT_TOKENS:-64}"
EPOCHS="${EPOCHS:-24}"
LMC_ITERATIONS="${LMC_ITERATIONS:-1}"
TRAINING_BUFFER_SIZE="${TRAINING_BUFFER_SIZE:-10000000}"
BUFFER_SIZE_FINAL="${BUFFER_SIZE_FINAL:-10000000}"
POST_TRAIN_HYPOTHESES="${POST_TRAIN_HYPOTHESES:-256}"
ITERATION_EVAL_HYPOTHESES="${ITERATION_EVAL_HYPOTHESES:-256}"
POST_TRAIN_EVAL_SEEDS=(${POST_TRAIN_EVAL_SEEDS:-1305 2026 4242})

SCENES=(${SCENES:-wayspots_bears wayspots_squarebench wayspots_cubes})
VARIANTS=(${VARIANTS:-single pmrf_base cl_pmrf_v3 v3_adapter_control})
GPU_LIST=(${GPU_LIST:-0 1})
EVAL_GPU_LIST=(${EVAL_GPU_LIST:-${GPU_LIST[*]}})

MAX_JOBS_PER_GPU="${MAX_JOBS_PER_GPU:-1}"
TRAIN_WORKERS="${TRAIN_WORKERS:-12}"
EVAL_WORKERS="${EVAL_WORKERS:-6}"
MAX_EVAL_JOBS="${MAX_EVAL_JOBS:-1}"
STAGGER_SECONDS="${STAGGER_SECONDS:-90}"
PLACEMENT_CHECK_DELAY="${PLACEMENT_CHECK_DELAY:-25}"
CHECK_PLACEMENT="${CHECK_PLACEMENT:-true}"
PRECHECK_NVIDIA_SMI="${PRECHECK_NVIDIA_SMI:-true}"

# inline: keep eval inside train_ace_dinov2_lmc.py. This is the default legacy-compatible mode.
# deferred: train with eval disabled, then run post-train eval as a separate serial phase.
# none: train only, no post-train eval.
EVAL_MODE="${EVAL_MODE:-inline}"
ALLOW_INLINE_EVAL_WITH_CONCURRENCY="${ALLOW_INLINE_EVAL_WITH_CONCURRENCY:-false}"

SKIP_EXISTING="${SKIP_EXISTING:-true}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-true}"
DRY_RUN="${DRY_RUN:-false}"

mkdir -p "${RUN_ROOT}/logs"
STATUS_FILE="${RUN_ROOT}/status.tsv"
if [[ ! -f "${STATUS_FILE}" ]]; then
  printf "timestamp\tphase\tvariant\tscene\tstatus\texit_code\tgpu\tlog\n" > "${STATUS_FILE}"
fi

log_status() {
  local phase="$1" variant="$2" scene="$3" status="$4" exit_code="$5" gpu="$6" log_file="$7"
  printf "%s\t%s\t%s\t%s\t%s\t%s\tcuda:%s\t%s\n" \
    "$(date +%Y-%m-%dT%H:%M:%S)" "${phase}" "${variant}" "${scene}" \
    "${status}" "${exit_code}" "${gpu}" "${log_file}" >> "${STATUS_FILE}"
}

print_cmd() {
  printf "%q " "$@"
  printf "\n"
}

variant_flags() {
  local variant="$1"
  case "${variant}" in
    single)
      echo "single 1.0 False 0.0 True 1.0 0.01 0.0"
      ;;
    pmrf_base)
      echo "progressive_reread 0.10 True -2.0 False 1.0 0.01 0.0"
      ;;
    cl_pmrf_v3)
      echo "centered_reread_qknorm_layerscale 1.0 False 0.0 False 1.0 0.01 0.0"
      ;;
    v3_adapter_control)
      echo "v3_adapter_control 1.0 False 0.0 False 1.0 0.01 0.0"
      ;;
    *)
      echo "Unknown variant: ${variant}" >&2
      return 2
      ;;
  esac
}

checkpoint_pattern() {
  printf "best_K%s_it%s_ace_fcn_local_stage1.pt" "${NUM_LATENT_TOKENS}" "${LMC_ITERATIONS}"
}

find_checkpoint() {
  local variant="$1" scene="$2"
  local variant_root="${RUN_ROOT}/${variant}"
  local stage_subdir="stage1_local_ace_memory_it${LMC_ITERATIONS}"
  find "${variant_root}/${stage_subdir}" -type f -name "$(checkpoint_pattern)" -path "*${scene}*" 2>/dev/null | sort | tail -n 1 || true
}

maybe_check_placement() {
  local tag="$1"
  local initial_delay="${2:-0}"
  if [[ "${CHECK_PLACEMENT}" != "true" || "${DRY_RUN}" == "true" ]]; then
    return 0
  fi
  sleep $((initial_delay + PLACEMENT_CHECK_DELAY))
  {
    printf "\n[%s] placement check: %s\n" "$(date)" "${tag}"
    nvidia-smi || true
    ps -eo pid,ppid,pgid,etime,pcpu,pmem,rss,cmd | rg 'train_ace_dinov2_lmc|test_ace_dinov2_lmc|eval_best_post_train|conda run' || true
  } | tee -a "${RUN_ROOT}/logs/placement_checks.log"
}

run_train_job() {
  local variant="$1" scene="$2" gpu="$3" delay="$4"
  if [[ "${delay}" -gt 0 && "${DRY_RUN}" != "true" ]]; then
    sleep "${delay}"
  fi

  local mode alpha scalar_gate gate_init post_norm common_scale patch_init common_init
  read -r mode alpha scalar_gate gate_init post_norm common_scale patch_init common_init < <(variant_flags "${variant}")

  local scene_root="${WAYSPOTS_ROOT}/${scene}"
  local memory_path="${MEMORY_ROOT}/${scene}/memory_ace_fcn_sparse_sp_r4.pt"
  local variant_root="${RUN_ROOT}/${variant}"
  local stage_subdir="stage1_local_ace_memory_it${LMC_ITERATIONS}"
  local output_suffix="ace_fcn_local_stage1.pt"
  local log_dir="${variant_root}/logs"
  local log_file="${log_dir}/train_${scene}.log"
  mkdir -p "${log_dir}"

  {
    printf "[%s] phase=train variant=%s mode=%s scene=%s gpu=%s delay=%ss\n" "$(date)" "${variant}" "${mode}" "${scene}" "${gpu}" "${delay}"
    printf "memory=%s\n" "${memory_path}"
    printf "workers train=%s eval=%s eval_mode=%s max_jobs_per_gpu=%s\n" "${TRAIN_WORKERS}" "${EVAL_WORKERS}" "${EVAL_MODE}" "${MAX_JOBS_PER_GPU}"
    printf "protocol=stage1 it%s buf%s final%s epochs%s hypo_iter%s hypo_post%s seeds=%s\n" \
      "${LMC_ITERATIONS}" "${TRAINING_BUFFER_SIZE}" "${BUFFER_SIZE_FINAL}" "${EPOCHS}" \
      "${ITERATION_EVAL_HYPOTHESES}" "${POST_TRAIN_HYPOTHESES}" "${POST_TRAIN_EVAL_SEEDS[*]}"
  } | tee -a "${log_file}"

  if [[ ! -f "${memory_path}" ]]; then
    printf "Missing memory path: %s\n" "${memory_path}" | tee -a "${log_file}"
    log_status train "${variant}" "${scene}" missing_memory 2 "${gpu}" "${log_file}"
    return 2
  fi

  local existing_ckpt
  existing_ckpt="$(find_checkpoint "${variant}" "${scene}")"
  if [[ -n "${existing_ckpt}" && "${SKIP_EXISTING}" == "true" ]]; then
    printf "Skipping existing checkpoint: %s\n" "${existing_ckpt}" | tee -a "${log_file}"
    log_status train "${variant}" "${scene}" skipped_existing 0 "${gpu}" "${log_file}"
    return 0
  fi

  local cmd=(
    env -u CUDA_VISIBLE_DEVICES
    conda run --no-capture-output -n "${CONDA_ENV}" "${PYTHON_BIN}" ace_dinov2_lmc/train_ace_dinov2_lmc.py
    "${scene_root}" "${output_suffix}"
    --model_backend ace_fcn_lmc
    --data_backend ace
    --use_lmc True
    --lmc_flow ace_g
    --memory_path "${memory_path}"
    --use_scale_token False
    --ace_encoder_path "${ACE_ENCODER_PATH}"
    --ace_lmc_global_head_mode none
    --device "cuda:${gpu}"
    --post_train_eval_device "cuda:${gpu}"
    --experiment_root "${variant_root}"
    --experiment_subdir "${stage_subdir}"
    --lmc_iterations "${LMC_ITERATIONS}"
    --num_latent_tokens "${NUM_LATENT_TOKENS}"
    --lmc_fusion_refinement_mode "${mode}"
    --lmc_fusion_cascade_layers 4
    --lmc_fusion_assembly_gamma_init 0.0
    --lmc_fusion_reread_delta_alpha "${alpha}"
    --lmc_fusion_reread_scalar_gate "${scalar_gate}"
    --lmc_fusion_reread_gate_init "${gate_init}"
    --lmc_fusion_reread_post_norm "${post_norm}"
    --lmc_fusion_reread_trust_region_ratio 0.0
    --lmc_fusion_reread_temperature 1.0
    --lmc_fusion_reread_common_scale "${common_scale}"
    --lmc_fusion_reread_effective_ratio_cap 0.0
    --lmc_fusion_reread_qknorm_eps 1e-6
    --lmc_fusion_reread_qknorm_tau_init 0.0
    --lmc_fusion_reread_layerscale_patch_init "${patch_init}"
    --lmc_fusion_reread_layerscale_common_init "${common_init}"
    --lmc_fusion_reread_warmup_mode none
    --lmc_fusion_reread_warmup_iters 0
    --lmc_fusion_reread_warmup_start 0.0
    --lmc_fusion_reread_geo_lambda 1.0
    --lmc_fusion_reread_geo_sigma 1.0
    --lmc_fusion_reread_geo_sigma_mode fixed
    --lmc_fusion_reread_geo_sigma_beta 1.0
    --lmc_fusion_reread_geo_sigma_min 0.5
    --s1_loss_step_mode fixed_zero
    --s1_early_stop False
    --lmc_log_runtime_stats True
    --lmc_runtime_stats_interval 100
    --lmc_runtime_stats_max_pixels 4096
    --num_data_loader_workers "${TRAIN_WORKERS}"
    --eval_num_workers "${EVAL_WORKERS}"
    --eval_deterministic True
    --eval_dsacstar_seed 1305
    --eval_dsacstar_seed_per_frame True
    --iteration_eval_seed 1305
    --iteration_eval_hypotheses "${ITERATION_EVAL_HYPOTHESES}"
    --ace_g_fusion_in_s2 True
    --s1_use_buffer True
    --s1_loss_mode sample_per_image
    --s1_buffer_refill_mode full
    --s1_last_iter_use_final_buffer True
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
    --c1_aux_depth_root "${scene_root}/train/sparse_depth"
    --c1_aux_depth_kind sparse_depth
    --epochs "${EPOCHS}"
    --post_train_eval_seeds "${POST_TRAIN_EVAL_SEEDS[@]}"
    --post_train_hypotheses "${POST_TRAIN_HYPOTHESES}"
  )

  if [[ "${EVAL_MODE}" == "deferred" || "${EVAL_MODE}" == "none" ]]; then
    cmd+=(--eval_each_iteration False --eval_after_train False --ace_g_cross_iter_eval False)
  else
    cmd+=(--ace_g_cross_iter_eval True)
  fi

  print_cmd "${cmd[@]}" | tee -a "${log_file}"
  log_status train "${variant}" "${scene}" running 0 "${gpu}" "${log_file}"

  if [[ "${DRY_RUN}" == "true" ]]; then
    log_status train "${variant}" "${scene}" dry_run 0 "${gpu}" "${log_file}"
    return 0
  fi

  "${cmd[@]}" 2>&1 | tee -a "${log_file}"
  local exit_code=${PIPESTATUS[0]}
  if [[ ${exit_code} -eq 0 ]]; then
    log_status train "${variant}" "${scene}" ok 0 "${gpu}" "${log_file}"
  else
    log_status train "${variant}" "${scene}" failed "${exit_code}" "${gpu}" "${log_file}"
  fi
  return "${exit_code}"
}

run_eval_job() {
  local variant="$1" scene="$2" gpu="$3"
  local ckpt log_file
  ckpt="$(find_checkpoint "${variant}" "${scene}")"
  log_file="${RUN_ROOT}/${variant}/logs/eval_${scene}.log"
  mkdir -p "$(dirname "${log_file}")"

  if [[ -z "${ckpt}" && "${DRY_RUN}" == "true" ]]; then
    printf "[%s] dry-run: no checkpoint yet for eval: variant=%s scene=%s\n" "$(date)" "${variant}" "${scene}" | tee -a "${log_file}"
    log_status eval "${variant}" "${scene}" dry_run_no_checkpoint 0 "${gpu}" "${log_file}"
    return 0
  fi

  if [[ -z "${ckpt}" ]]; then
    printf "[%s] missing checkpoint for eval: variant=%s scene=%s\n" "$(date)" "${variant}" "${scene}" | tee -a "${log_file}"
    log_status eval "${variant}" "${scene}" missing_checkpoint 2 "${gpu}" "${log_file}"
    return 2
  fi

  local cmd=(
    env -u CUDA_VISIBLE_DEVICES
    conda run --no-capture-output -n "${CONDA_ENV}" "${PYTHON_BIN}" ace_dinov2_lmc/scripts/eval_best_post_train.py
    "${WAYSPOTS_ROOT}/${scene}" "${ckpt}"
    --device "cuda:${gpu}"
    --data_backend ace
    --image_resolution "${IMAGE_RESOLUTION}"
    --hypotheses "${POST_TRAIN_HYPOTHESES}"
    --seeds "${POST_TRAIN_EVAL_SEEDS[@]}"
    --output_dir "$(dirname "${ckpt}")"
    --ace_encoder_path "${ACE_ENCODER_PATH}"
    --eval_num_workers "${EVAL_WORKERS}"
    --eval_deterministic
  )

  {
    printf "[%s] phase=eval variant=%s scene=%s gpu=%s checkpoint=%s\n" "$(date)" "${variant}" "${scene}" "${gpu}" "${ckpt}"
    print_cmd "${cmd[@]}"
  } | tee -a "${log_file}"
  log_status eval "${variant}" "${scene}" running 0 "${gpu}" "${log_file}"

  if [[ "${DRY_RUN}" == "true" ]]; then
    log_status eval "${variant}" "${scene}" dry_run 0 "${gpu}" "${log_file}"
    return 0
  fi

  "${cmd[@]}" 2>&1 | tee -a "${log_file}"
  local exit_code=${PIPESTATUS[0]}
  if [[ ${exit_code} -eq 0 ]]; then
    log_status eval "${variant}" "${scene}" ok 0 "${gpu}" "${log_file}"
  else
    log_status eval "${variant}" "${scene}" failed "${exit_code}" "${gpu}" "${log_file}"
  fi
  return "${exit_code}"
}

wait_wave() {
  local -n pid_ref=$1
  local failures=0
  local pid rc
  for pid in "${pid_ref[@]}"; do
    wait "${pid}"
    rc=$?
    if [[ ${rc} -ne 0 ]]; then
      failures=$((failures + 1))
    fi
  done
  return "${failures}"
}

schedule_training() {
  local jobs_total=${#JOB_VARIANTS[@]}
  local gpu_count=${#GPU_LIST[@]}
  local wave_capacity=$((gpu_count * MAX_JOBS_PER_GPU))
  local start offset idx gpu delay slot_on_gpu
  local pids=()
  local check_pids=()
  local wave_id=0

  for ((start = 0; start < jobs_total; start += wave_capacity)); do
    wave_id=$((wave_id + 1))
    pids=()
    check_pids=()
    printf "\n[%s] train wave %d start=%d capacity=%d\n" "$(date)" "${wave_id}" "${start}" "${wave_capacity}" | tee -a "${RUN_ROOT}/logs/scheduler.log"
    for ((offset = 0; offset < wave_capacity && start + offset < jobs_total; offset++)); do
      idx=$((start + offset))
      gpu="${GPU_LIST[$((offset % gpu_count))]}"
      slot_on_gpu=$((offset / gpu_count))
      delay=$((slot_on_gpu * STAGGER_SECONDS))
      run_train_job "${JOB_VARIANTS[$idx]}" "${JOB_SCENES[$idx]}" "${gpu}" "${delay}" &
      pids+=("$!")
      maybe_check_placement "train wave ${wave_id} job ${idx} variant=${JOB_VARIANTS[$idx]} scene=${JOB_SCENES[$idx]} gpu=${gpu}" "${delay}" &
      check_pids+=("$!")
    done
    wait_wave pids
    local failures=$?
    wait_wave check_pids || true
    if [[ ${failures} -ne 0 && "${CONTINUE_ON_ERROR}" != "true" ]]; then
      printf "Stopping after train wave %d because %d job(s) failed.\n" "${wave_id}" "${failures}" | tee -a "${RUN_ROOT}/logs/scheduler.log"
      exit 1
    fi
  done
}

schedule_eval() {
  local jobs_total=${#JOB_VARIANTS[@]}
  local eval_gpu_count=${#EVAL_GPU_LIST[@]}
  local wave_capacity="${MAX_EVAL_JOBS}"
  local start offset idx gpu
  local pids=()
  local check_pids=()
  local wave_id=0

  if [[ "${EVAL_MODE}" != "deferred" ]]; then
    return 0
  fi

  for ((start = 0; start < jobs_total; start += wave_capacity)); do
    wave_id=$((wave_id + 1))
    pids=()
    check_pids=()
    printf "\n[%s] eval wave %d start=%d capacity=%d\n" "$(date)" "${wave_id}" "${start}" "${wave_capacity}" | tee -a "${RUN_ROOT}/logs/scheduler.log"
    for ((offset = 0; offset < wave_capacity && start + offset < jobs_total; offset++)); do
      idx=$((start + offset))
      gpu="${EVAL_GPU_LIST[$(((start + offset) % eval_gpu_count))]}"
      run_eval_job "${JOB_VARIANTS[$idx]}" "${JOB_SCENES[$idx]}" "${gpu}" &
      pids+=("$!")
      maybe_check_placement "eval wave ${wave_id} job ${idx} variant=${JOB_VARIANTS[$idx]} scene=${JOB_SCENES[$idx]} gpu=${gpu}" 0 &
      check_pids+=("$!")
    done
    wait_wave pids
    local failures=$?
    wait_wave check_pids || true
    if [[ ${failures} -ne 0 && "${CONTINUE_ON_ERROR}" != "true" ]]; then
      printf "Stopping after eval wave %d because %d job(s) failed.\n" "${wave_id}" "${failures}" | tee -a "${RUN_ROOT}/logs/scheduler.log"
      exit 1
    fi
  done
}

if [[ ${#GPU_LIST[@]} -eq 0 ]]; then
  echo "GPU_LIST is empty." >&2
  exit 2
fi
if [[ "${MAX_JOBS_PER_GPU}" -lt 1 ]]; then
  echo "MAX_JOBS_PER_GPU must be >= 1." >&2
  exit 2
fi
if [[ "${EVAL_MODE}" != "deferred" && "${EVAL_MODE}" != "inline" && "${EVAL_MODE}" != "none" ]]; then
  echo "EVAL_MODE must be deferred, inline, or none." >&2
  exit 2
fi
if [[ "${EVAL_MODE}" == "inline" && "${MAX_JOBS_PER_GPU}" -gt 1 && "${ALLOW_INLINE_EVAL_WITH_CONCURRENCY}" != "true" ]]; then
  echo "Refusing EVAL_MODE=inline with MAX_JOBS_PER_GPU>1. Use EVAL_MODE=deferred, or set ALLOW_INLINE_EVAL_WITH_CONCURRENCY=true." >&2
  exit 2
fi

JOB_VARIANTS=()
JOB_SCENES=()
for variant in "${VARIANTS[@]}"; do
  variant_flags "${variant}" >/dev/null || exit 2
  for scene in "${SCENES[@]}"; do
    JOB_VARIANTS+=("${variant}")
    JOB_SCENES+=("${scene}")
  done
done

{
  printf "Run root: %s\n" "${RUN_ROOT}"
  printf "Memory  : %s\n" "${MEMORY_ROOT}"
  printf "Scenes  : %s\n" "${SCENES[*]}"
  printf "Variants: %s\n" "${VARIANTS[*]}"
  printf "GPUs    : %s\n" "${GPU_LIST[*]}"
  printf "Eval GPUs: %s\n" "${EVAL_GPU_LIST[*]}"
  printf "Policy  : max_jobs_per_gpu=%s train_workers=%s eval_workers=%s eval_mode=%s max_eval_jobs=%s stagger=%ss\n" \
    "${MAX_JOBS_PER_GPU}" "${TRAIN_WORKERS}" "${EVAL_WORKERS}" "${EVAL_MODE}" "${MAX_EVAL_JOBS}" "${STAGGER_SECONDS}"
  printf "Important: this launcher unsets CUDA_VISIBLE_DEVICES and passes physical --device cuda:<gpu>.\n"
} | tee -a "${RUN_ROOT}/logs/scheduler.log"

if [[ "${PRECHECK_NVIDIA_SMI}" == "true" && "${DRY_RUN}" != "true" ]]; then
  {
    printf "\n[%s] preflight nvidia-smi\n" "$(date)"
    nvidia-smi || true
  } | tee -a "${RUN_ROOT}/logs/placement_checks.log"
fi

schedule_training
schedule_eval

if [[ "${DRY_RUN}" != "true" ]]; then
  conda run --no-capture-output -n "${CONDA_ENV}" "${PYTHON_BIN}" \
    ace_dinov2_lmc/.codex_skills/lmc-experiment-standard-workflow/scripts/best_metric_summary.py \
    "${RUN_ROOT}" | tee "${RUN_ROOT}/best_metric_summary.log" || true
fi

printf "Done. Root: %s\n" "${RUN_ROOT}" | tee -a "${RUN_ROOT}/logs/scheduler.log"
