#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

CONDA_ENV="${CONDA_ENV:-mapanything}"
WAYSPOTS_ROOT="${WAYSPOTS_ROOT:-/data/xwh/Wayspots}"
MEMORY_ROOT="${MEMORY_ROOT:-/data/xwh/ace_dinov2_lmc/04_evaluation/wayspots_ace_fcn_lmc_suite/20260530_181745/memory}"
SIDECAR_ROOT="${SIDECAR_ROOT:-/data/xwh/dataset_staging/wayspots/extracted}"
ACE_ENCODER_PATH="${ACE_ENCODER_PATH:-${ROOT_DIR}/ace_encoder_pretrained.pt}"

LMC_ITERATIONS="${LMC_ITERATIONS:-2}"
EPOCHS="${EPOCHS:-24}"
TRAINING_BUFFER_SIZE="${TRAINING_BUFFER_SIZE:-5000000}"
BUFFER_SIZE_FINAL="${BUFFER_SIZE_FINAL:-7600000}"
IMAGE_RESOLUTION="${IMAGE_RESOLUTION:-512}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
SAMPLES_PER_IMAGE="${SAMPLES_PER_IMAGE:-512}"
GUIDED_BATCH="${GUIDED_BATCH:-512}"
POST_TRAIN_HYPOTHESES="${POST_TRAIN_HYPOTHESES:-256}"
ITERATION_EVAL_HYPOTHESES="${ITERATION_EVAL_HYPOTHESES:-256}"
POST_TRAIN_EVAL_SEEDS=(${POST_TRAIN_EVAL_SEEDS:-1305 2026 4242})

QUALITY_MIN_ALIGNMENT_WEIGHT="${QUALITY_MIN_ALIGNMENT_WEIGHT:-0.70}"
QUALITY_MAX_COLMAP_REPROJ_PX="${QUALITY_MAX_COLMAP_REPROJ_PX:-1.5}"
QUALITY_MIN_TRACK_LENGTH="${QUALITY_MIN_TRACK_LENGTH:-8}"
PATCH_MAX_ANCHOR_OFFSET_PX="${PATCH_MAX_ANCHOR_OFFSET_PX:-1.5}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-/data/xwh/ace_dinov2_lmc/04_evaluation/wayspots/stgs_guided_screen/${STAMP}_stgs_filter_matrix_gpu01_it${LMC_ITERATIONS}_buf5m}"
LOG_DIR="${RUN_ROOT}/launcher_logs"
SUPERVISOR_LOG="${LOG_DIR}/supervisor_${STAMP}.log"
mkdir -p "${LOG_DIR}"

VARIANTS=(
  current
  quality_gate
  patch_safe
  quality_patch
)

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${SUPERVISOR_LOG}"
}

write_plan() {
  cat >"${RUN_ROOT}/matrix_plan_${STAMP}.txt" <<EOF
purpose: STGS short-iteration filter matrix for next-method decision
run_root: ${RUN_ROOT}
timestamp: ${STAMP}
scenes:
  gpu0: wayspots_bears
  gpu1: wayspots_cubes
variants:
  current: current anchor-v2 STGS, no extra filtering
  quality_gate: current + min_alignment_weight=${QUALITY_MIN_ALIGNMENT_WEIGHT}, max_colmap_reproj_px=${QUALITY_MAX_COLMAP_REPROJ_PX}, min_track_length=${QUALITY_MIN_TRACK_LENGTH}
  patch_safe: current + max_anchor_patch_offset_px=${PATCH_MAX_ANCHOR_OFFSET_PX}
  quality_patch: quality_gate + patch_safe
base_stgs:
  guided_mode: anchor_only
  guided_strategy: balanced_replace
  guided_fraction: 0.10
  guided_main_loss_mode: include
  guided_source_target_mode: patch_center
  guided_aux_normalizer: full_batch
  anchor_self_weight: 0.5
  anchor_use_alignment_weight: True
  inter_frame_weight: 0.0
short_train:
  lmc_iterations: ${LMC_ITERATIONS}
  epochs: ${EPOCHS}
  training_buffer_size: ${TRAINING_BUFFER_SIZE}
  buffer_size_final: ${BUFFER_SIZE_FINAL}
  image_resolution: ${IMAGE_RESOLUTION}
  batch_size: ${BATCH_SIZE}
  samples_per_image: ${SAMPLES_PER_IMAGE}
  post_train_eval_seeds: ${POST_TRAIN_EVAL_SEEDS[*]}
  post_train_hypotheses: ${POST_TRAIN_HYPOTHESES}
gpu_policy: pass physical --device cuda:0/cuda:1 and clear outer CUDA_VISIBLE_DEVICES per train process
EOF
}

require_inputs() {
  local scene="$1"
  local scene_root="${WAYSPOTS_ROOT}/${scene}"
  local memory_path="${MEMORY_ROOT}/${scene}/memory_ace_fcn_sparse_sp_r4.pt"
  local keyframe_channel="${SIDECAR_ROOT}/${scene}/train/colmap_keyframe_channel_v1_sp_strict/keyframe_channel.npz"
  [[ -d "${scene_root}/train/rgb" ]] || { log "missing train/rgb for ${scene}: ${scene_root}/train/rgb"; exit 2; }
  [[ -d "${scene_root}/train/poses" ]] || { log "missing train/poses for ${scene}: ${scene_root}/train/poses"; exit 2; }
  [[ -d "${scene_root}/train/calibration" ]] || { log "missing train/calibration for ${scene}: ${scene_root}/train/calibration"; exit 2; }
  [[ -d "${scene_root}/train/sparse_depth" ]] || { log "missing sparse_depth for ${scene}: ${scene_root}/train/sparse_depth"; exit 2; }
  [[ -f "${memory_path}" ]] || { log "missing memory for ${scene}: ${memory_path}"; exit 2; }
  [[ -f "${keyframe_channel}" ]] || { log "missing keyframe channel for ${scene}: ${keyframe_channel}"; exit 2; }
}

append_common_args() {
  local -n out="$1"
  local scene="$2"
  local scene_root="${WAYSPOTS_ROOT}/${scene}"
  local memory_path="${MEMORY_ROOT}/${scene}/memory_ace_fcn_sparse_sp_r4.pt"
  out+=(
    --model_backend ace_fcn_lmc
    --data_backend ace
    --use_lmc True
    --lmc_flow ace_g
    --memory_path "${memory_path}"
    --use_scale_token False
    --ace_encoder_path "${ACE_ENCODER_PATH}"
    --ace_lmc_global_head_mode none
    --lmc_iterations "${LMC_ITERATIONS}"
    --num_latent_tokens 64
    --lmc_fusion_refinement_mode single
    --s1_loss_step_mode fixed_zero
    --s1_early_stop False
    --num_data_loader_workers 8
    --eval_num_workers 4
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
}

append_stgs_args() {
  local -n out="$1"
  local scene="$2"
  local variant="$3"
  local keyframe_channel="${SIDECAR_ROOT}/${scene}/train/colmap_keyframe_channel_v1_sp_strict/keyframe_channel.npz"
  out+=(
    --use_sfm_track_guided_sampling True
    --sfm_track_keyframe_channel_path "${keyframe_channel}"
    --sfm_track_guided_batch_size "${GUIDED_BATCH}"
    --sfm_track_guided_sampling_strategy balanced_replace
    --sfm_track_guided_fraction 0.10
    --sfm_track_guided_mode anchor_only
    --sfm_track_guided_main_loss_mode include
    --sfm_track_guided_source_target_mode patch_center
    --sfm_track_guided_aux_normalizer full_batch
    --sfm_track_anchor_self_weight 0.5
    --sfm_track_anchor_use_alignment_weight True
    --sfm_track_inter_frame_weight 0.0
    --sfm_track_inter_frame_dropout 0.5
    --sfm_track_inter_frame_start_ratio 0.2
    --sfm_track_inter_frame_decay_last_ratio 0.3
    --sfm_track_inter_frame_max_px 100.0
  )
  case "${variant}" in
    current)
      ;;
    quality_gate)
      out+=(
        --sfm_track_min_alignment_weight "${QUALITY_MIN_ALIGNMENT_WEIGHT}"
        --sfm_track_max_colmap_reproj_error_px "${QUALITY_MAX_COLMAP_REPROJ_PX}"
        --sfm_track_min_track_length "${QUALITY_MIN_TRACK_LENGTH}"
      )
      ;;
    patch_safe)
      out+=(--sfm_track_max_anchor_patch_offset_px "${PATCH_MAX_ANCHOR_OFFSET_PX}")
      ;;
    quality_patch)
      out+=(
        --sfm_track_min_alignment_weight "${QUALITY_MIN_ALIGNMENT_WEIGHT}"
        --sfm_track_max_colmap_reproj_error_px "${QUALITY_MAX_COLMAP_REPROJ_PX}"
        --sfm_track_min_track_length "${QUALITY_MIN_TRACK_LENGTH}"
        --sfm_track_max_anchor_patch_offset_px "${PATCH_MAX_ANCHOR_OFFSET_PX}"
      )
      ;;
    *)
      log "unknown variant: ${variant}"
      exit 2
      ;;
  esac
}

run_train() {
  local scene="$1"
  local variant="$2"
  local gpu="$3"
  local scene_root="${WAYSPOTS_ROOT}/${scene}"
  local tag="${scene}_${variant}_stgs_filter_it${LMC_ITERATIONS}_buf5m_${STAMP}"
  local exp_root="${RUN_ROOT}/${scene}/${variant}"
  local exp_subdir="stage1_local_stgs_filter_matrix_short"
  local log_file="${LOG_DIR}/${tag}_gpu${gpu}.log"

  local -a cmd=(
    conda run --no-capture-output -n "${CONDA_ENV}" python ace_dinov2_lmc/train_ace_dinov2_lmc.py
    "${scene_root}" "${tag}.pt"
    --run_name "${tag}"
    --device "cuda:${gpu}"
    --post_train_eval_device "cuda:${gpu}"
    --experiment_root "${exp_root}"
    --experiment_subdir "${exp_subdir}"
  )
  append_common_args cmd "${scene}"
  append_stgs_args cmd "${scene}" "${variant}"

  {
    echo "[$(date)] scene=${scene} variant=${variant} physical_gpu=${gpu} run_root=${RUN_ROOT}"
    printf '%q ' "${cmd[@]}"
    printf '\n'
  } >>"${log_file}"

  log "start ${scene} ${variant} on physical gpu${gpu}; log=${log_file}"
  env -u CUDA_VISIBLE_DEVICES "${cmd[@]}" >>"${log_file}" 2>&1
  log "done ${scene} ${variant} on physical gpu${gpu}"
}

run_scene_queue() {
  local scene="$1"
  local gpu="$2"
  require_inputs "${scene}"
  for variant in "${VARIANTS[@]}"; do
    run_train "${scene}" "${variant}" "${gpu}"
  done
}

main() {
  write_plan
  log "matrix plan: ${RUN_ROOT}/matrix_plan_${STAMP}.txt"
  log "launch queues: wayspots_bears->gpu0, wayspots_cubes->gpu1"
  run_scene_queue wayspots_bears 0 &
  local pid_bears=$!
  run_scene_queue wayspots_cubes 1 &
  local pid_cubes=$!

  local rc_bears=0
  local rc_cubes=0
  wait "${pid_bears}" || rc_bears=$?
  wait "${pid_cubes}" || rc_cubes=$?
  if [[ "${rc_bears}" -ne 0 || "${rc_cubes}" -ne 0 ]]; then
    log "matrix failed: bears_rc=${rc_bears}, cubes_rc=${rc_cubes}"
    exit 1
  fi
  log "matrix complete"
}

main "$@"
