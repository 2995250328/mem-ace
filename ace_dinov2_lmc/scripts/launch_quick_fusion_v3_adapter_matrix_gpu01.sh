#!/usr/bin/env bash
set -uo pipefail

# Quick fusion-structure screening for ACE-G stage1 local memory.
#
# Protocol:
#   - stage1 only
#   - 1 LMC iteration
#   - 10M training/final buffer
#   - deterministic eval, 256 DSAC hypotheses
#   - post-train seeds: 1305, 2026, 4242
#   - GPUs restricted to 0/1
#
# The matrix is intentionally small and diagnostic:
#   single              : strong first-read baseline
#   pmrf_base           : weak progressive reread baseline
#   cl_pmrf_v3          : centered + second-read QK norm + patch/common LayerScale
#   v3_adapter_control  : PMRF-v3 matched query-only adapter, no second memory read
#   v3_dual_refine      : query adapter + centered memory reread weak dual refinement

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

CONDA_ENV="${CONDA_ENV:-mapanything}"
PYTHON_BIN="${PYTHON_BIN:-python}"
WAYSPOTS_ROOT="${WAYSPOTS_ROOT:-/data/xwh/Wayspots}"
MEMORY_ROOT="${MEMORY_ROOT:-/data/xwh/ace_dinov2_lmc/04_evaluation/wayspots_ace_fcn_lmc_suite/20260530_181745/memory}"
RUN_ROOT="${RUN_ROOT:-/data/xwh/ace_dinov2_lmc/04_evaluation/quick_fusion_v3_adapter_matrix_20260623_gpu01}"
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

mkdir -p "${RUN_ROOT}/logs"
STATUS_FILE="${RUN_ROOT}/status.tsv"
if [[ ! -f "${STATUS_FILE}" ]]; then
  printf "timestamp\tvariant\tscene\tstatus\texit_code\tgpu\tlog\n" > "${STATUS_FILE}"
fi

log_status() {
  local variant="$1" scene="$2" status="$3" exit_code="$4" gpu="$5" log_file="$6"
  printf "%s\t%s\t%s\t%s\t%s\tcuda:%s\t%s\n" \
    "$(date +%Y-%m-%dT%H:%M:%S)" "${variant}" "${scene}" "${status}" "${exit_code}" "${gpu}" "${log_file}" >> "${STATUS_FILE}"
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
    v3_dual_refine)
      echo "v3_dual_refine 1.0 False 0.0 False 1.0 0.01 0.0"
      ;;
    *)
      echo "Unknown variant: ${variant}" >&2
      return 2
      ;;
  esac
}

run_job() {
  local variant="$1"
  local scene="$2"
  local gpu="$3"

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
    printf "[%s] variant=%s mode=%s scene=%s gpu=%s\n" "$(date)" "${variant}" "${mode}" "${scene}" "${gpu}"
    printf "memory=%s\n" "${memory_path}"
    printf "protocol=stage1 it%s buf%s final%s epochs%s hypo_iter%s hypo_post%s seeds=%s\n" \
      "${LMC_ITERATIONS}" "${TRAINING_BUFFER_SIZE}" "${BUFFER_SIZE_FINAL}" "${EPOCHS}" \
      "${ITERATION_EVAL_HYPOTHESES}" "${POST_TRAIN_HYPOTHESES}" "${POST_TRAIN_EVAL_SEEDS[*]}"
  } | tee -a "${log_file}"

  if [[ ! -f "${memory_path}" ]]; then
    printf "Missing memory path: %s\n" "${memory_path}" | tee -a "${log_file}"
    log_status "${variant}" "${scene}" "missing_memory" 2 "${gpu}" "${log_file}"
    return 2
  fi

  local checkpoint_pattern="best_K${NUM_LATENT_TOKENS}_it${LMC_ITERATIONS}_${output_suffix}"
  local existing_ckpt
  existing_ckpt=$(find "${variant_root}/${stage_subdir}" -type f -name "${checkpoint_pattern}" -path "*${scene}*" 2>/dev/null | sort | tail -n 1 || true)
  if [[ -n "${existing_ckpt}" && "${SKIP_EXISTING:-true}" == "true" ]]; then
    printf "Skipping existing checkpoint: %s\n" "${existing_ckpt}" | tee -a "${log_file}"
    log_status "${variant}" "${scene}" "skipped_existing" 0 "${gpu}" "${log_file}"
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
    --lmc_fusion_dual_memory_layerscale_patch_init "${DUAL_MEMORY_LAYERSCALE_PATCH_INIT:-0.005}"
    --lmc_fusion_dual_memory_layerscale_common_init "${DUAL_MEMORY_LAYERSCALE_COMMON_INIT:-0.0}"
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
    --num_data_loader_workers 12
    --eval_num_workers 6
    --eval_deterministic True
    --eval_dsacstar_seed 1305
    --eval_dsacstar_seed_per_frame True
    --iteration_eval_seed 1305
    --iteration_eval_hypotheses "${ITERATION_EVAL_HYPOTHESES}"
    --ace_g_fusion_in_s2 True
    --ace_g_cross_iter_eval True
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

  print_cmd "${cmd[@]}" | tee -a "${log_file}"
  "${cmd[@]}" 2>&1 | tee -a "${log_file}"
  local exit_code=${PIPESTATUS[0]}
  if [[ ${exit_code} -eq 0 ]]; then
    log_status "${variant}" "${scene}" "ok" 0 "${gpu}" "${log_file}"
  else
    log_status "${variant}" "${scene}" "failed" "${exit_code}" "${gpu}" "${log_file}"
    if [[ "${CONTINUE_ON_ERROR:-true}" != "true" ]]; then
      exit "${exit_code}"
    fi
  fi
  return "${exit_code}"
}

run_wave() {
  local variant="$1"
  local scene0="$2"
  local scene1="${3:-}"

  run_job "${variant}" "${scene0}" 0 &
  local pid0=$!
  local pid1=""
  if [[ -n "${scene1}" ]]; then
    run_job "${variant}" "${scene1}" 1 &
    pid1=$!
  fi

  wait "${pid0}"
  local rc0=$?
  local rc1=0
  if [[ -n "${pid1}" ]]; then
    wait "${pid1}"
    rc1=$?
  fi

  if [[ "${CONTINUE_ON_ERROR:-true}" != "true" ]]; then
    if [[ ${rc0} -ne 0 || ${rc1} -ne 0 ]]; then
      exit 1
    fi
  fi
}

printf "Run root: %s\n" "${RUN_ROOT}"
printf "Memory  : %s\n" "${MEMORY_ROOT}"
printf "Scenes  : %s\n" "${SCENES[*]}"
printf "Variants: %s\n" "${VARIANTS[*]}"
printf "GPUs    : 0/1 only\n"

for variant in "${VARIANTS[@]}"; do
  printf "\n[%s] Starting variant: %s\n" "$(date)" "${variant}"
  run_wave "${variant}" "${SCENES[0]}" "${SCENES[1]:-}"
  if [[ ${#SCENES[@]} -ge 3 ]]; then
    run_wave "${variant}" "${SCENES[2]}"
  fi
done

conda run --no-capture-output -n "${CONDA_ENV}" "${PYTHON_BIN}" \
  ace_dinov2_lmc/.codex_skills/lmc-experiment-standard-workflow/scripts/best_metric_summary.py \
  "${RUN_ROOT}" | tee "${RUN_ROOT}/best_metric_summary.log" || true

printf "Done. Root: %s\n" "${RUN_ROOT}"
