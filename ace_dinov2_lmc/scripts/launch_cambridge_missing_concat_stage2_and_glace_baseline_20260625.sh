#!/usr/bin/env bash
set -euo pipefail

# Corrected Cambridge completion run:
#   - Run only missing NEW concat-stage2 scenes:
#       Cambridge_KingsCollege, Cambridge_OldHospital
#   - Run original GLACE baseline for all Cambridge scenes.
#
# Existing NEW concat-stage2 results are intentionally not rerun:
#   - GreatCourt, ShopFacade:
#       /data/xwh/ace_dinov2_lmc/04_evaluation/cambridge_stage2_fused_backbone_glace_it10_buf10m_final12m_20260624_gpu23
#   - StMarysChurch:
#       /data/xwh/ace_dinov2_lmc/04_evaluation/cambridge_stmary_stage2_fused_from_global_it30_it10_buf10m_final12m_20260625_gpu0

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

CONDA_ENV="${CONDA_ENV:-mapanything}"
CAMBRIDGE_ROOT="${CAMBRIDGE_ROOT:-/data/xwh/Cambridge}"
SOURCE_RUN_ROOT="${SOURCE_RUN_ROOT:-/home/xwh/project/ace_depth/ace_dinov2_lmc/04_evaluation/cambridge_single_stage12_checked_20260624_gpu23}"
RUN_ROOT="${RUN_ROOT:-/data/xwh/ace_dinov2_lmc/04_evaluation/cambridge_missing_concat_stage2_and_glace_baseline_20260625_gpu0123}"
STATUS_FILE="${RUN_ROOT}/status.tsv"
DRY_RUN="${DRY_RUN:-false}"
SKIP_EXISTING="${SKIP_EXISTING:-true}"

ACE_ENCODER_PATH="${ACE_ENCODER_PATH:-${ROOT_DIR}/ace_encoder_pretrained.pt}"
GLACE_ROOT="${GLACE_ROOT:-/home/xwh/project/glace}"
GLACE_R2FORMER_CKPT="${GLACE_R2FORMER_CKPT:-${GLACE_ROOT}/CVPR23_DeitS_Rerank.pth}"
GLACE_FEAT_NAME="${GLACE_FEAT_NAME:-features.npy}"

POST_TRAIN_HYPOTHESES="${POST_TRAIN_HYPOTHESES:-256}"
POST_TRAIN_EVAL_SEEDS=(${POST_TRAIN_EVAL_SEEDS:-1305 2026 4242})

STAGE2_SUBDIR="${STAGE2_SUBDIR:-stage2_fused_backbone_glace_it10_buf10m_final12m}"
STAGE2_LMC_ITERATIONS="${STAGE2_LMC_ITERATIONS:-10}"
STAGE2_TRAINING_BUFFER_SIZE="${STAGE2_TRAINING_BUFFER_SIZE:-10000000}"
STAGE2_BUFFER_SIZE_FINAL="${STAGE2_BUFFER_SIZE_FINAL:-12000000}"
STAGE2_IMAGE_RESOLUTION="${STAGE2_IMAGE_RESOLUTION:-518}"
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-4096}"
STAGE2_SAMPLES_PER_IMAGE="${STAGE2_SAMPLES_PER_IMAGE:-512}"

GLACE_SUBDIR="${GLACE_SUBDIR:-glace_original_full}"
GLACE_TRAINING_BUFFER_SIZE="${GLACE_TRAINING_BUFFER_SIZE:-8000000}"
GLACE_SAMPLES_PER_IMAGE="${GLACE_SAMPLES_PER_IMAGE:-1024}"
GLACE_BATCH_SIZE="${GLACE_BATCH_SIZE:-32768}"
GLACE_MAX_ITERATIONS="${GLACE_MAX_ITERATIONS:-30000}"
GLACE_IMAGE_RESOLUTION="${GLACE_IMAGE_RESOLUTION:-480}"
GLACE_EXTRACT_BATCH_SIZE="${GLACE_EXTRACT_BATCH_SIZE:-128}"
GLACE_EXTRACT_WORKERS="${GLACE_EXTRACT_WORKERS:-4}"
GLACE_RENDER_FLIPPED_PORTRAIT="${GLACE_RENDER_FLIPPED_PORTRAIT:-true}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OMP_DYNAMIC="${OMP_DYNAMIC:-FALSE}"

mkdir -p "${RUN_ROOT}/logs"
if [[ ! -f "${STATUS_FILE}" ]]; then
  printf "timestamp\tscene\tmethod\tstage\tstatus\texit_code\tdevice\tlog\n" > "${STATUS_FILE}"
fi

print_cmd() {
  printf "%q " "$@"
  printf "\n"
}

log_status() {
  local scene="$1" method="$2" stage="$3" status="$4" exit_code="$5" device="$6" log_file="$7"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$(date +%Y-%m-%dT%H:%M:%S)" "$scene" "$method" "$stage" "$status" "$exit_code" "$device" "$log_file" >> "${STATUS_FILE}"
}

run_logged() {
  local scene="$1" method="$2" stage="$3" device="$4" log_file="$5"
  shift 5
  mkdir -p "$(dirname "${log_file}")"
  {
    printf "[%s] scene=%s method=%s stage=%s device=%s\n" "$(date)" "$scene" "$method" "$stage" "$device"
    print_cmd "$@"
  } | tee -a "${log_file}"

  if [[ "${DRY_RUN}" == "true" ]]; then
    log_status "$scene" "$method" "$stage" "dry_run" 0 "$device" "$log_file"
    return 0
  fi

  "$@" 2>&1 | tee -a "${log_file}"
  local exit_code=${PIPESTATUS[0]}
  if [[ ${exit_code} -eq 0 ]]; then
    log_status "$scene" "$method" "$stage" "ok" 0 "$device" "$log_file"
  else
    log_status "$scene" "$method" "$stage" "failed" "${exit_code}" "$device" "$log_file"
  fi
  return "${exit_code}"
}

find_local_stage1_ckpt() {
  local scene="$1"
  find "${SOURCE_RUN_ROOT}/stage1_local_ace_memory_it12" -type f \
    -name "best_K64_it12_ace_fcn_local_stage1.pt" -path "*${scene}*" 2>/dev/null | sort | tail -n 1 || true
}

preflight_scene() {
  local scene="$1"
  local scene_root="${CAMBRIDGE_ROOT}/${scene}"
  for split in train test; do
    [[ -d "${scene_root}/${split}/rgb" ]] || { echo "missing rgb: ${scene_root}/${split}/rgb" >&2; return 2; }
    [[ -d "${scene_root}/${split}/poses" ]] || { echo "missing poses: ${scene_root}/${split}/poses" >&2; return 2; }
    [[ -d "${scene_root}/${split}/calibration" ]] || { echo "missing calibration: ${scene_root}/${split}/calibration" >&2; return 2; }
    [[ -s "${scene_root}/${split}/${GLACE_FEAT_NAME}" ]] || { echo "missing GLACE feat: ${scene_root}/${split}/${GLACE_FEAT_NAME}" >&2; return 2; }
  done
  [[ -d "${scene_root}/train/sparse_depth" ]] || { echo "missing sparse depth: ${scene_root}/train/sparse_depth" >&2; return 2; }
}

run_missing_concat_stage2() {
  local scene="$1"
  local gpu="$2"
  local scene_root="${CAMBRIDGE_ROOT}/${scene}"
  local memory_path="${SOURCE_RUN_ROOT}/memory/${scene}/memory_ace_fcn_sparse_sp_r4.pt"
  local stage1_ckpt
  stage1_ckpt="$(find_local_stage1_ckpt "${scene}")"
  local log_file="${RUN_ROOT}/logs/stage2_concat_${scene}_gpu${gpu}.log"
  local output_suffix="ace_fcn_fused_backbone_glace_stage2.pt"

  [[ -s "${stage1_ckpt}" ]] || { echo "missing stage1 checkpoint for ${scene}" | tee -a "${log_file}" >&2; return 2; }
  [[ -s "${memory_path}" ]] || { echo "missing memory for ${scene}: ${memory_path}" | tee -a "${log_file}" >&2; return 2; }
  preflight_scene "${scene}" 2>&1 | tee -a "${log_file}"

  local existing
  existing=$(find "${RUN_ROOT}/${STAGE2_SUBDIR}" -type f -name "best_K64_it${STAGE2_LMC_ITERATIONS}_${output_suffix}" -path "*${scene}*" 2>/dev/null | sort | tail -n 1 || true)
  if [[ -n "${existing}" && "${SKIP_EXISTING}" == "true" ]]; then
    log_status "${scene}" "stage2_fused_backbone_glace" "train" "skipped_existing" 0 "cuda:${gpu}" "${log_file}"
    return 0
  fi

  run_logged "${scene}" "stage2_fused_backbone_glace" "train" "cuda:${gpu}" "${log_file}" \
    conda run --no-capture-output -n "${CONDA_ENV}" python ace_dinov2_lmc/train_ace_dinov2_lmc.py \
      "${scene_root}" "${output_suffix}" \
      --model_backend ace_fcn_lmc \
      --data_backend ace \
      --post_train_eval_scene "${scene_root}" \
      --use_lmc True \
      --lmc_flow ace_g \
      --memory_path "${memory_path}" \
      --use_scale_token False \
      --ace_encoder_path "${ACE_ENCODER_PATH}" \
      --ace_lmc_global_head_mode glace_concat \
      --ace_lmc_local_checkpoint_path "${stage1_ckpt}" \
      --ace_lmc_freeze_local_stack True \
      --ace_lmc_stage2_feature_source stage1_fused \
      --ace_lmc_global_feature_mode glace \
      --ace_lmc_global_gate_init 1.0 \
      --ace_lmc_global_gate_learnable False \
      --ace_lmc_global_normalize True \
      --ace_lmc_global_noise_std 0.1 \
      --glace_root "${GLACE_ROOT}" \
      --glace_feat_name "${GLACE_FEAT_NAME}" \
      --glace_head_channels 768 \
      --glace_mlp_ratio 1.0 \
      --num_head_blocks 2 \
      --best_metric median_error \
      --device "cuda:${gpu}" \
      --post_train_eval_device "cuda:${gpu}" \
      --experiment_root "${RUN_ROOT}" \
      --experiment_subdir "${STAGE2_SUBDIR}" \
      --lmc_iterations "${STAGE2_LMC_ITERATIONS}" \
      --num_latent_tokens 64 \
      --lmc_fusion_refinement_mode single \
      --lmc_fusion_cascade_layers 4 \
      --lmc_fusion_assembly_gamma_init 0.0 \
      --s1_loss_step_mode fixed_zero \
      --s1_early_stop False \
      --lmc_log_runtime_stats True \
      --lmc_runtime_stats_interval 100 \
      --lmc_runtime_stats_max_pixels 4096 \
      --num_data_loader_workers 12 \
      --eval_num_workers 6 \
      --eval_deterministic True \
      --eval_dsacstar_seed 1305 \
      --eval_dsacstar_seed_per_frame True \
      --iteration_eval_seed 1305 \
      --iteration_eval_hypotheses 256 \
      --ace_g_fusion_in_s2 False \
      --ace_g_cross_iter_eval False \
      --s1_use_buffer True \
      --s1_loss_mode sample_per_image \
      --s1_buffer_refill_mode full \
      --image_resolution "${STAGE2_IMAGE_RESOLUTION}" \
      --batch_size "${STAGE2_BATCH_SIZE}" \
      --training_buffer_size "${STAGE2_TRAINING_BUFFER_SIZE}" \
      --buffer_size_final "${STAGE2_BUFFER_SIZE_FINAL}" \
      --buffer_on_cpu True \
      --buffer_on_cpu_final True \
      --samples_per_image "${STAGE2_SAMPLES_PER_IMAGE}" \
      --buffer_sample_valid_coords True \
      --buffer_valid_coord_sample_ratio 1.0 \
      --buffer_valid_coord_neighbor_radius 1 \
      --buffer_valid_coord_neighbor_mode cross \
      --c1_aux_depth_root "${scene_root}/train/sparse_depth" \
      --c1_aux_depth_kind sparse_depth \
      --post_train_eval_seeds "${POST_TRAIN_EVAL_SEEDS[@]}" \
      --post_train_hypotheses "${POST_TRAIN_HYPOTHESES}"
}

run_glace_original() {
  local scene="$1"
  local gpu="$2"
  local scene_root="${CAMBRIDGE_ROOT}/${scene}"
  local method_dir="${RUN_ROOT}/${GLACE_SUBDIR}/${scene}"
  local model="${method_dir}/model.pt"
  local log_file="${RUN_ROOT}/logs/glace_original_${scene}_gpu${gpu}.log"
  mkdir -p "${method_dir}"
  preflight_scene "${scene}" 2>&1 | tee -a "${log_file}"

  if [[ -s "${method_dir}/poses_${scene}_post_train.txt" && "${SKIP_EXISTING}" == "true" ]]; then
    log_status "${scene}" "glace_original" "all" "skipped_existing" 0 "cuda:${gpu}" "${log_file}"
    return 0
  fi

  run_logged "${scene}" "glace_original" "train" "cuda:${gpu}" "${log_file}" \
    env CUDA_VISIBLE_DEVICES="${gpu}" conda run --no-capture-output -n "${CONDA_ENV}" \
      torchrun --standalone --nnodes 1 --nproc-per-node 1 "${GLACE_ROOT}/train_ace.py" "${scene_root}" "${model}" \
        --training_buffer_size "${GLACE_TRAINING_BUFFER_SIZE}" \
        --samples_per_image "${GLACE_SAMPLES_PER_IMAGE}" \
        --batch_size "${GLACE_BATCH_SIZE}" \
        --max_iterations "${GLACE_MAX_ITERATIONS}" \
        --image_resolution "${GLACE_IMAGE_RESOLUTION}" \
        --use_half True \
        --use_homogeneous True \
        --use_aug True \
        --render_flipped_portrait "${GLACE_RENDER_FLIPPED_PORTRAIT}"

  run_logged "${scene}" "glace_original" "eval" "cuda:${gpu}" "${log_file}" \
    env CUDA_VISIBLE_DEVICES="${gpu}" conda run --no-capture-output -n "${CONDA_ENV}" \
      python "${GLACE_ROOT}/test_ace.py" "${scene_root}" "${model}" \
        --session post_train \
        --image_resolution "${GLACE_IMAGE_RESOLUTION}" \
        --hypotheses "${POST_TRAIN_HYPOTHESES}" \
        --render_flipped_portrait "${GLACE_RENDER_FLIPPED_PORTRAIT}"
}

worker0() {
  run_missing_concat_stage2 Cambridge_KingsCollege 0 || true
  run_glace_original Cambridge_GreatCourt 0 || true
  run_glace_original Cambridge_KingsCollege 0 || true
}

worker1() {
  run_missing_concat_stage2 Cambridge_OldHospital 1 || true
  run_glace_original Cambridge_OldHospital 1 || true
}

worker2() {
  run_glace_original Cambridge_ShopFacade 2 || true
}

worker3() {
  run_glace_original Cambridge_StMarysChurch 3 || true
}

printf "RUN_ROOT=%s\n" "${RUN_ROOT}"
printf "Missing new concat stage2: Cambridge_KingsCollege Cambridge_OldHospital\n"
printf "GLACE original baseline: Cambridge_GreatCourt Cambridge_KingsCollege Cambridge_OldHospital Cambridge_ShopFacade Cambridge_StMarysChurch\n"
printf "DRY_RUN=%s SKIP_EXISTING=%s\n" "${DRY_RUN}" "${SKIP_EXISTING}"

worker0 & pid0=$!
worker1 & pid1=$!
worker2 & pid2=$!
worker3 & pid3=$!
wait "${pid0}" "${pid1}" "${pid2}" "${pid3}"
printf "Done. Status: %s\n" "${STATUS_FILE}"
