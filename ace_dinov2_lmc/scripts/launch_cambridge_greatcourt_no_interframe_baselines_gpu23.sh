#!/usr/bin/env bash
set -euo pipefail

# Cambridge GreatCourt no-inter-frame baselines for comparison with S2-G inter-frame runs.
# Variants:
#   nostgs: no STGS guided sampling, no inter-frame loss.
#   guided_only: STGS guided/anchor path enabled, inter-frame weight forced to 0.

ROOT_DIR="${ROOT_DIR:-/home/xwh/project/ace_depth}"
REPO_ROOT="${REPO_ROOT:-${ROOT_DIR}/ace_dinov2_lmc}"
cd "${ROOT_DIR}"

CONDA_ENV="${CONDA_ENV:-mapanything}"
ACE_ENCODER_PATH="${ACE_ENCODER_PATH:-${ROOT_DIR}/ace_encoder_pretrained.pt}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-/data/xwh/ace_dinov2_lmc/04_evaluation/cambridge_greatcourt_no_interframe/${STAMP}_gpu23}"
LOG_DIR="${RUN_ROOT}/launcher_logs"
mkdir -p "${LOG_DIR}"
SUPERVISOR_LOG="${LOG_DIR}/supervisor_${STAMP}.log"

CAMBRIDGE_ROOT="${CAMBRIDGE_ROOT:-/data/xwh/Cambridge}"
CAMBRIDGE_SCENE="${CAMBRIDGE_SCENE:-Cambridge_GreatCourt}"
SCENE_ROOT="${CAMBRIDGE_ROOT}/${CAMBRIDGE_SCENE}"
SOURCE_RUN_ROOT="${SOURCE_RUN_ROOT:-${REPO_ROOT}/04_evaluation/cambridge_single_stage12_checked_20260624_gpu23}"
MEMORY_PATH="${MEMORY_PATH:-${SOURCE_RUN_ROOT}/memory/${CAMBRIDGE_SCENE}/memory_ace_fcn_sparse_sp_r4.pt}"
SIDECAR_PATH="${SIDECAR_PATH:-/data/xwh/ace_dinov2_lmc/04_evaluation/interframe_s2g_wscan/20260703_151548_w025_w005_indoor_wayspots_cambridge_gpu23/preprocess/cambridge/Cambridge_GreatCourt/colmap_keyframe_channel_v1_sp_strict_acefcn_r512/keyframe_channel.npz}"

GPU_NOSTGS="${GPU_NOSTGS:-2}"
GPU_GUIDED_ONLY="${GPU_GUIDED_ONLY:-3}"

LMC_ITERATIONS="${LMC_ITERATIONS:-2}"
TRAINING_BUFFER_SIZE="${TRAINING_BUFFER_SIZE:-10000000}"
BUFFER_SIZE_FINAL="${BUFFER_SIZE_FINAL:-10000000}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
IMAGE_RESOLUTION="${IMAGE_RESOLUTION:-512}"
POST_TRAIN_EVAL_SEEDS=(${POST_TRAIN_EVAL_SEEDS:-1305 2026 4242})
POST_TRAIN_HYPOTHESES="${POST_TRAIN_HYPOTHESES:-256}"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "${SUPERVISOR_LOG}"
}

run_logged() {
  local label="$1"
  local log_file="$2"
  shift 2
  mkdir -p "$(dirname "${log_file}")"
  {
    printf '[%s] %s\n' "$(date)" "${label}"
    printf '%q ' "$@"
    printf '\n'
  } >> "${log_file}"
  log "start ${label}; log=${log_file}"
  "$@" >> "${log_file}" 2>&1
  log "done ${label}"
}

require_file() {
  local path="$1"
  local label="$2"
  [[ -f "${path}" ]] || { log "missing ${label}: ${path}"; exit 2; }
}

require_dir() {
  local path="$1"
  local label="$2"
  [[ -d "${path}" ]] || { log "missing ${label}: ${path}"; exit 2; }
}

write_manifest() {
  cat > "${RUN_ROOT}/matrix_plan_${STAMP}.txt" <<EOF
purpose: Cambridge GreatCourt no-inter-frame baselines
run_root: ${RUN_ROOT}
timestamp: ${STAMP}
scene: ${CAMBRIDGE_SCENE}
gpus:
  nostgs: ${GPU_NOSTGS}
  guided_only: ${GPU_GUIDED_ONLY}
variants:
  nostgs: use_sfm_track_guided_sampling=False, use_sfm_track_inter_frame_loss=False
  guided_only: use_sfm_track_guided_sampling=True, use_sfm_track_inter_frame_loss=False, sfm_track_inter_frame_weight=0.0
shared:
  lmc_iterations: ${LMC_ITERATIONS}
  buffer: ${TRAINING_BUFFER_SIZE}/${BUFFER_SIZE_FINAL}
  image_resolution: ${IMAGE_RESOLUTION}
  memory_path: ${MEMORY_PATH}
  sidecar_path: ${SIDECAR_PATH}
EOF
}

base_train_args() {
  local gpu="$1"
  local variant="$2"
  local tag="${CAMBRIDGE_SCENE}_${variant}_it${LMC_ITERATIONS}_buf10m_${STAMP}"
  local root="${RUN_ROOT}/${variant}"
  printf '%s\0' \
    conda run --no-capture-output -n "${CONDA_ENV}" python ace_dinov2_lmc/train_ace_dinov2_lmc.py \
    "${SCENE_ROOT}" "${tag}.pt" \
    --run_name "${tag}" \
    --model_backend ace_fcn_lmc \
    --data_backend ace \
    --post_train_eval_scene "${SCENE_ROOT}" \
    --use_lmc True \
    --lmc_flow ace_g \
    --memory_path "${MEMORY_PATH}" \
    --use_scale_token False \
    --ace_encoder_path "${ACE_ENCODER_PATH}" \
    --ace_lmc_global_head_mode none \
    --best_metric median_error \
    --device "cuda:${gpu}" \
    --post_train_eval_device "cuda:${gpu}" \
    --experiment_root "${root}" \
    --experiment_subdir "${variant}" \
    --lmc_iterations "${LMC_ITERATIONS}" \
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
    --eval_each_iteration True \
    --eval_after_train True \
    --keep_best_only True \
    --ace_g_fusion_in_s2 True \
    --ace_g_fusion_lr_ratio 0.01 \
    --ace_g_cross_iter_eval False \
    --ace_g_s2_schedule every_iter \
    --s1_use_buffer True \
    --s1_loss_mode sample_per_image \
    --s1_buffer_refill_mode full \
    --s1_last_iter_use_final_buffer True \
    --image_resolution "${IMAGE_RESOLUTION}" \
    --batch_size "${BATCH_SIZE}" \
    --training_buffer_size "${TRAINING_BUFFER_SIZE}" \
    --buffer_size_final "${BUFFER_SIZE_FINAL}" \
    --buffer_on_cpu True \
    --buffer_on_cpu_final True \
    --samples_per_image 512 \
    --buffer_sample_valid_coords True \
    --buffer_valid_coord_sample_ratio 1.0 \
    --buffer_valid_coord_neighbor_radius 1 \
    --buffer_valid_coord_neighbor_mode cross \
    --c1_aux_depth_root "${SCENE_ROOT}/train/sparse_depth" \
    --c1_aux_depth_kind sparse_depth \
    --post_train_eval_seeds "${POST_TRAIN_EVAL_SEEDS[@]}" \
    --post_train_hypotheses "${POST_TRAIN_HYPOTHESES}"
}

run_nostgs() {
  local -a args=()
  while IFS= read -r -d '' item; do args+=("${item}"); done < <(base_train_args "${GPU_NOSTGS}" "nostgs")
  args+=(
    --use_sfm_track_guided_sampling False
    --use_sfm_track_inter_frame_loss False
    --sfm_track_inter_frame_weight 0.0
  )
  run_logged "${CAMBRIDGE_SCENE} nostgs" "${LOG_DIR}/cambridge_${CAMBRIDGE_SCENE}_nostgs_gpu${GPU_NOSTGS}.log" "${args[@]}"
}

run_guided_only() {
  local -a args=()
  while IFS= read -r -d '' item; do args+=("${item}"); done < <(base_train_args "${GPU_GUIDED_ONLY}" "guided_only")
  args+=(
    --use_sfm_track_guided_sampling True
    --use_sfm_track_inter_frame_loss False
    --sfm_track_keyframe_channel_path "${SIDECAR_PATH}"
    --sfm_track_sidecar_mismatch_policy strict
    --sfm_track_guided_batch_size 512
    --sfm_track_guided_sampling_strategy balanced_replace
    --sfm_track_guided_fraction 0.10
    --sfm_track_guided_mode inter_frame
    --sfm_track_guided_main_loss_mode include
    --sfm_track_guided_source_target_mode patch_center
    --sfm_track_guided_aux_normalizer full_batch
    --sfm_track_anchor_self_weight 0.5
    --sfm_track_anchor_use_alignment_weight True
    --sfm_track_inter_frame_target_mode exact_target
    --sfm_track_inter_frame_apply_to stage2_g
    --sfm_track_inter_frame_weight 0.0
    --sfm_track_inter_frame_dropout 0.5
    --sfm_track_inter_frame_start_ratio 0.2
    --sfm_track_inter_frame_decay_last_ratio 0.3
    --sfm_track_inter_frame_max_px 100.0
    --sfm_track_max_anchor_patch_offset_px 2.0
    --sfm_track_max_target_patch_offset_px 0.0
    --sfm_track_min_alignment_weight 0.0
    --sfm_track_max_colmap_reproj_error_px 2.0
    --sfm_track_min_track_length 4
  )
  run_logged "${CAMBRIDGE_SCENE} guided_only" "${LOG_DIR}/cambridge_${CAMBRIDGE_SCENE}_guided_only_gpu${GPU_GUIDED_ONLY}.log" "${args[@]}"
}

aggregate_results() {
  log "aggregate metric-wise best under ${RUN_ROOT}"
  if bash "${REPO_ROOT}/scripts/aggregate_lmc_metricwise_best.sh" "${RUN_ROOT}" > "${RUN_ROOT}/metricwise_best.aggregate.log" 2>&1; then
    log "aggregate ok: ${RUN_ROOT}/metricwise_best.tsv"
  else
    log "aggregate failed; see ${RUN_ROOT}/metricwise_best.aggregate.log"
  fi
}

main() {
  require_dir "${SCENE_ROOT}/train/rgb" "${CAMBRIDGE_SCENE} train/rgb"
  require_dir "${SCENE_ROOT}/train/sparse_depth" "${CAMBRIDGE_SCENE} sparse_depth"
  require_file "${MEMORY_PATH}" "${CAMBRIDGE_SCENE} memory"
  require_file "${SIDECAR_PATH}" "${CAMBRIDGE_SCENE} sidecar"
  write_manifest
  log "run root: ${RUN_ROOT}"
  log "matrix plan: ${RUN_ROOT}/matrix_plan_${STAMP}.txt"

  run_nostgs &
  local pid_a=$!
  run_guided_only &
  local pid_b=$!
  wait "${pid_a}"
  wait "${pid_b}"
  aggregate_results
  log "all done"
}

main "$@"
