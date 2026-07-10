#!/usr/bin/env bash
set -euo pipefail

# Short inter-frame STGS matrix using the validated S2-G injection path.
# This intentionally does not use the experimental S1-only branch.

ROOT_DIR="${ROOT_DIR:-/home/xwh/project/ace_depth}"
REPO_ROOT="${REPO_ROOT:-${ROOT_DIR}/ace_dinov2_lmc}"
cd "${ROOT_DIR}"

CONDA_ENV="${CONDA_ENV:-mapanything}"
PYTHON_BIN="${PYTHON_BIN:-/home/xwh/miniforge3/envs/mapanything/bin/python}"
ACE_ENCODER_PATH="${ACE_ENCODER_PATH:-${ROOT_DIR}/ace_encoder_pretrained.pt}"
SUPERPOINT_WEIGHTS="${SUPERPOINT_WEIGHTS:-${REPO_ROOT}/superpoint_v1.pth}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-/data/xwh/ace_dinov2_lmc/04_evaluation/interframe_s2g_wscan/${STAMP}_w025_w005_indoor_wayspots_cambridge_gpu23}"
LOG_DIR="${RUN_ROOT}/launcher_logs"
mkdir -p "${LOG_DIR}"
SUPERVISOR_LOG="${LOG_DIR}/supervisor_${STAMP}.log"

GPU_INDOOR="${GPU_INDOOR:-2}"
GPU_WAYSPOTS_A="${GPU_WAYSPOTS_A:-2}"
GPU_WAYSPOTS_B="${GPU_WAYSPOTS_B:-3}"
GPU_CAMBRIDGE="${GPU_CAMBRIDGE:-3}"
GPU_CAMBRIDGE_PREP="${GPU_CAMBRIDGE_PREP:-3}"

W_VALUES="${W_VALUES:-0.25 0.05}"
INDOOR_W_VALUES="${INDOOR_W_VALUES:-0.25}"

INDOOR_SCENE="${INDOOR_SCENE:-scene2a}"
INDOOR_SIDECAR_ROOT="${INDOOR_SIDECAR_ROOT:-/data/xwh/ace_dinov2_lmc/04_evaluation/indoor6/dino_stgs_matrix/20260629_203557_it2_buf10M_pair_gpu23/sidecars}"

WAYSPOTS_ROOT="${WAYSPOTS_ROOT:-/data/xwh/Wayspots}"
WAYSPOTS_MEMORY_ROOT="${WAYSPOTS_MEMORY_ROOT:-/data/xwh/ace_dinov2_lmc/04_evaluation/wayspots_ace_fcn_lmc_suite/20260530_181745/memory}"
WAYSPOTS_SIDECAR_ROOT="${WAYSPOTS_SIDECAR_ROOT:-/data/xwh/dataset_staging/wayspots/extracted}"
WAYSPOTS_SCENE_A="${WAYSPOTS_SCENE_A:-wayspots_bears}"
WAYSPOTS_SCENE_B="${WAYSPOTS_SCENE_B:-wayspots_cubes}"

CAMBRIDGE_ROOT="${CAMBRIDGE_ROOT:-/data/xwh/Cambridge}"
CAMBRIDGE_SCENE="${CAMBRIDGE_SCENE:-Cambridge_GreatCourt}"
CAMBRIDGE_SOURCE_RUN_ROOT="${CAMBRIDGE_SOURCE_RUN_ROOT:-${REPO_ROOT}/04_evaluation/cambridge_single_stage12_checked_20260624_gpu23}"
CAMBRIDGE_PREPROCESS_ROOT="${CAMBRIDGE_PREPROCESS_ROOT:-${RUN_ROOT}/preprocess/cambridge/${CAMBRIDGE_SCENE}}"
CAMBRIDGE_SPARSE_DEPTH_SUBDIR="${CAMBRIDGE_SPARSE_DEPTH_SUBDIR:-sparse_depth_superpoint_stgs_${STAMP}}"
CAMBRIDGE_SIDECAR_DIR="${CAMBRIDGE_SIDECAR_DIR:-${CAMBRIDGE_PREPROCESS_ROOT}/colmap_keyframe_channel_v1_sp_strict_acefcn_r512}"

ACEFCN_LMC_ITERATIONS="${ACEFCN_LMC_ITERATIONS:-2}"
ACEFCN_TRAINING_BUFFER_SIZE="${ACEFCN_TRAINING_BUFFER_SIZE:-10000000}"
ACEFCN_BUFFER_SIZE_FINAL="${ACEFCN_BUFFER_SIZE_FINAL:-10000000}"
ACEFCN_BATCH_SIZE="${ACEFCN_BATCH_SIZE:-4096}"
ACEFCN_IMAGE_RESOLUTION="${ACEFCN_IMAGE_RESOLUTION:-512}"
ACEFCN_EPOCHS="${ACEFCN_EPOCHS:-24}"

CAMBRIDGE_LMC_ITERATIONS="${CAMBRIDGE_LMC_ITERATIONS:-2}"
CAMBRIDGE_TRAINING_BUFFER_SIZE="${CAMBRIDGE_TRAINING_BUFFER_SIZE:-10000000}"
CAMBRIDGE_BUFFER_SIZE_FINAL="${CAMBRIDGE_BUFFER_SIZE_FINAL:-10000000}"
CAMBRIDGE_BATCH_SIZE="${CAMBRIDGE_BATCH_SIZE:-4096}"
CAMBRIDGE_IMAGE_RESOLUTION="${CAMBRIDGE_IMAGE_RESOLUTION:-512}"

POST_TRAIN_EVAL_SEEDS=(${POST_TRAIN_EVAL_SEEDS:-1305 2026 4242})
POST_TRAIN_HYPOTHESES="${POST_TRAIN_HYPOTHESES:-256}"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "${SUPERVISOR_LOG}"
}

safe_w_label() {
  local w="$1"
  printf '%s' "${w}" | sed 's/\./p/g'
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
purpose: short inter-frame STGS matrix on validated S2-G path
run_root: ${RUN_ROOT}
timestamp: ${STAMP}
gpus:
  indoor: ${GPU_INDOOR}
  wayspots: ${GPU_WAYSPOTS_A},${GPU_WAYSPOTS_B}
  cambridge: train=${GPU_CAMBRIDGE}, preprocess=${GPU_CAMBRIDGE_PREP}
weights:
  indoor_dino: ${INDOOR_W_VALUES}
  ace_fcn: ${W_VALUES}
injection:
  description: S2-G only
  code_apply_to: stage2_g
  ace_g_s2_schedule: every_iter
indoor:
  scene: ${INDOOR_SCENE}
  backbone: DINOv2 + MapAnything memory
  sidecar_root: ${INDOOR_SIDECAR_ROOT}
wayspots:
  scenes: ${WAYSPOTS_SCENE_A} ${WAYSPOTS_SCENE_B}
  backbone: ACE-FCN-LMC
  target_mode: exact_target
  buffer: ${ACEFCN_TRAINING_BUFFER_SIZE}/${ACEFCN_BUFFER_SIZE_FINAL}
cambridge:
  scene: ${CAMBRIDGE_SCENE}
  backbone: ACE-FCN-LMC
  target_mode: exact_target
  image_resolution: ${CAMBRIDGE_IMAGE_RESOLUTION}
  buffer: ${CAMBRIDGE_TRAINING_BUFFER_SIZE}/${CAMBRIDGE_BUFFER_SIZE_FINAL}
EOF
}

run_indoor_one_weight() {
  local w="$1"
  local label="indoor6_${INDOOR_SCENE}_dino_if_w$(safe_w_label "${w}")_s2g"
  local root="${RUN_ROOT}/indoor6_dino/${label}"
  run_logged "indoor ${INDOOR_SCENE} ${label}" "${LOG_DIR}/${label}.log" \
    env \
      PYTHON_BIN="${PYTHON_BIN}" \
      SCENES_STR="${INDOOR_SCENE}" \
      GPUS_STR="${GPU_INDOOR}" \
      RUN_ROOT="${root}" \
      SIDECAR_ROOT="${INDOOR_SIDECAR_ROOT}" \
      VARIANT_LABEL="${label}" \
      USE_STGS_GUIDED_SAMPLING=True \
      USE_STGS_INTER_FRAME_LOSS=True \
      SFM_TRACK_GUIDED_MODE=inter_frame \
      SFM_TRACK_INTER_FRAME_WEIGHT="${w}" \
      SFM_TRACK_INTER_FRAME_APPLY_TO=stage2_g \
      ACE_G_S2_SCHEDULE=every_iter \
      SFM_TRACK_INTER_FRAME_DROPOUT=0.5 \
      SFM_TRACK_INTER_FRAME_START_RATIO=0.2 \
      SFM_TRACK_INTER_FRAME_DECAY_LAST_RATIO=0.3 \
      SIDECAR_IMAGE_WIDTH=none \
      REBUILD_SIDECAR=False \
      bash "${REPO_ROOT}/scripts/launch_indoor6_dino_stgs_short_gpu01.sh"
}

wayspots_sidecar_path() {
  local scene="$1"
  printf '%s/%s/train/colmap_keyframe_channel_v1_sp_strict/keyframe_channel.npz' "${WAYSPOTS_SIDECAR_ROOT}" "${scene}"
}

run_wayspots_scene_weight() {
  local scene="$1"
  local gpu="$2"
  local w="$3"
  local wlabel
  wlabel="$(safe_w_label "${w}")"
  local scene_root="${WAYSPOTS_ROOT}/${scene}"
  local memory_path="${WAYSPOTS_MEMORY_ROOT}/${scene}/memory_ace_fcn_sparse_sp_r4.pt"
  local sidecar
  sidecar="$(wayspots_sidecar_path "${scene}")"
  local tag="${scene}_acefcn_if_w${wlabel}_s2g_it${ACEFCN_LMC_ITERATIONS}_buf10m_${STAMP}"
  local root="${RUN_ROOT}/wayspots_acefcn/w${wlabel}/${scene}"
  local log_file="${LOG_DIR}/wayspots_${scene}_w${wlabel}_gpu${gpu}.log"

  require_dir "${scene_root}/train/rgb" "${scene} train/rgb"
  require_file "${memory_path}" "${scene} ACE-FCN memory"
  require_file "${sidecar}" "${scene} STGS sidecar"

  run_logged "wayspots ${scene} S2-G w=${w}" "${log_file}" \
    conda run --no-capture-output -n "${CONDA_ENV}" python ace_dinov2_lmc/train_ace_dinov2_lmc.py \
      "${scene_root}" "${tag}.pt" \
      --run_name "${tag}" \
      --model_backend ace_fcn_lmc \
      --data_backend ace \
      --post_train_eval_scene "${scene_root}" \
      --use_lmc True \
      --lmc_flow ace_g \
      --memory_path "${memory_path}" \
      --use_scale_token False \
      --ace_encoder_path "${ACE_ENCODER_PATH}" \
      --ace_lmc_global_head_mode none \
      --best_metric pct25_5 \
      --device "cuda:${gpu}" \
      --post_train_eval_device "cuda:${gpu}" \
      --experiment_root "${root}" \
      --experiment_subdir "interframe_s2g_short" \
      --lmc_iterations "${ACEFCN_LMC_ITERATIONS}" \
      --num_latent_tokens 64 \
      --lmc_fusion_refinement_mode single \
      --s1_loss_step_mode fixed_zero \
      --s1_early_stop False \
      --num_data_loader_workers 8 \
      --eval_num_workers 4 \
      --eval_deterministic True \
      --eval_dsacstar_seed 1305 \
      --eval_dsacstar_seed_per_frame True \
      --iteration_eval_seed 1305 \
      --iteration_eval_hypotheses 256 \
      --eval_each_iteration True \
      --eval_after_train True \
      --keep_best_only True \
      --ace_g_fusion_in_s2 True \
      --ace_g_cross_iter_eval False \
      --ace_g_s2_schedule every_iter \
      --s1_use_buffer True \
      --s1_loss_mode sample_per_image \
      --s1_buffer_refill_mode full \
      --s1_last_iter_use_final_buffer True \
      --image_resolution "${ACEFCN_IMAGE_RESOLUTION}" \
      --batch_size "${ACEFCN_BATCH_SIZE}" \
      --training_buffer_size "${ACEFCN_TRAINING_BUFFER_SIZE}" \
      --buffer_size_final "${ACEFCN_BUFFER_SIZE_FINAL}" \
      --buffer_on_cpu True \
      --buffer_on_cpu_final True \
      --samples_per_image 512 \
      --buffer_sample_valid_coords True \
      --buffer_valid_coord_sample_ratio 1.0 \
      --buffer_valid_coord_neighbor_radius 1 \
      --buffer_valid_coord_neighbor_mode cross \
      --c1_aux_depth_root "${scene_root}/train/sparse_depth" \
      --c1_aux_depth_kind sparse_depth \
      --epochs "${ACEFCN_EPOCHS}" \
      --use_sfm_track_guided_sampling True \
      --use_sfm_track_inter_frame_loss True \
      --sfm_track_keyframe_channel_path "${sidecar}" \
      --sfm_track_sidecar_mismatch_policy strict \
      --sfm_track_guided_batch_size 512 \
      --sfm_track_guided_sampling_strategy balanced_replace \
      --sfm_track_guided_fraction 0.10 \
      --sfm_track_guided_mode inter_frame \
      --sfm_track_guided_main_loss_mode include \
      --sfm_track_guided_source_target_mode patch_center \
      --sfm_track_guided_aux_normalizer full_batch \
      --sfm_track_anchor_self_weight 0.5 \
      --sfm_track_anchor_use_alignment_weight True \
      --sfm_track_inter_frame_target_mode exact_target \
      --sfm_track_inter_frame_apply_to stage2_g \
      --sfm_track_inter_frame_weight "${w}" \
      --sfm_track_inter_frame_dropout 0.5 \
      --sfm_track_inter_frame_start_ratio 0.2 \
      --sfm_track_inter_frame_decay_last_ratio 0.3 \
      --sfm_track_inter_frame_max_px 100.0 \
      --sfm_track_max_anchor_patch_offset_px 2.0 \
      --sfm_track_max_target_patch_offset_px 0.0 \
      --sfm_track_min_alignment_weight 0.0 \
      --sfm_track_max_colmap_reproj_error_px 2.0 \
      --sfm_track_min_track_length 4 \
      --post_train_eval_seeds "${POST_TRAIN_EVAL_SEEDS[@]}" \
      --post_train_hypotheses "${POST_TRAIN_HYPOTHESES}"
}

run_wayspots_matrix() {
  local w
  for w in ${W_VALUES}; do
    run_wayspots_scene_weight "${WAYSPOTS_SCENE_A}" "${GPU_WAYSPOTS_A}" "${w}" &
    local pid_a=$!
    run_wayspots_scene_weight "${WAYSPOTS_SCENE_B}" "${GPU_WAYSPOTS_B}" "${w}" &
    local pid_b=$!
    wait "${pid_a}"
    wait "${pid_b}"
  done
}

build_cambridge_sidecar_if_needed() {
  local scene_root="${CAMBRIDGE_ROOT}/${CAMBRIDGE_SCENE}"
  local workspace="${CAMBRIDGE_PREPROCESS_ROOT}/superpoint_workspace"
  local model_dir="${workspace}/triangulated_model"
  local sidecar="${CAMBRIDGE_SIDECAR_DIR}/keyframe_channel.npz"
  local sp_depth_dir="${scene_root}/train/${CAMBRIDGE_SPARSE_DEPTH_SUBDIR}"

  require_dir "${scene_root}/train/rgb" "${CAMBRIDGE_SCENE} train/rgb"
  require_file "${SUPERPOINT_WEIGHTS}" "SuperPoint weights"
  mkdir -p "${CAMBRIDGE_PREPROCESS_ROOT}" "${CAMBRIDGE_SIDECAR_DIR}"

  if [[ ! -f "${model_dir}/points3D.bin" ]]; then
    run_logged "cambridge ${CAMBRIDGE_SCENE} build SuperPoint workspace" "${LOG_DIR}/cambridge_${CAMBRIDGE_SCENE}_build_workspace.log" \
      env CUDA_VISIBLE_DEVICES="${GPU_CAMBRIDGE_PREP}" PYTHONPATH="/data/xwh/SuperPointPretrainedNetwork:${PYTHONPATH:-}" \
      conda run --no-capture-output -n "${CONDA_ENV}" python ace_dinov2_lmc/tools/wayspots_known_pose_sparse_depth.py \
        "${scene_root}" \
        --split train \
        --workspace "${workspace}" \
        --output-subdir "${CAMBRIDGE_SPARSE_DEPTH_SUBDIR}" \
        --feature-backend superpoint \
        --superpoint-weights "${SUPERPOINT_WEIGHTS}" \
        --superpoint-conf-thresh 0.05 \
        --superpoint-max-keypoints 1024 \
        --superpoint-nms-dist 8 \
        --superpoint-nn-thresh 0.60 \
        --matcher pairs \
        --match-window 20 \
        --num-threads 4 \
        --min-triangulation-angle 3.0 \
        --max-depth-m 1000 \
        --use-gpu \
        --gpu-index 0 \
        --overwrite
  else
    log "reuse Cambridge workspace: ${model_dir}"
  fi

  if [[ ! -f "${sidecar}" ]]; then
    run_logged "cambridge ${CAMBRIDGE_SCENE} build ACE-FCN r512 sidecar" "${LOG_DIR}/cambridge_${CAMBRIDGE_SCENE}_build_sidecar.log" \
      conda run --no-capture-output -n "${CONDA_ENV}" python ace_dinov2_lmc/tools/build_colmap_keyframe_channel.py \
        "${scene_root}" \
        --split train \
        --model-dir "${model_dir}" \
        --output-dir "${CAMBRIDGE_SIDECAR_DIR}" \
        --target-backbone ace_fcn \
        --image-resolution "${CAMBRIDGE_IMAGE_RESOLUTION}" \
        --min-track-length 4 \
        --max-reproj-error 2.0 \
        --min-parallax-deg 2.0 \
        --max-parallax-deg 60.0 \
        --max-anchor-alignment-px 2.0 \
        --alignment-sigma-px 2.0 \
        --sparse-depth-dir "${sp_depth_dir}" \
        --image-match-mode auto
  else
    log "reuse Cambridge sidecar: ${sidecar}"
  fi

  require_file "${sidecar}" "Cambridge STGS sidecar"
}

run_cambridge_weight() {
  local w="$1"
  local wlabel
  wlabel="$(safe_w_label "${w}")"
  local scene_root="${CAMBRIDGE_ROOT}/${CAMBRIDGE_SCENE}"
  local memory_path="${CAMBRIDGE_SOURCE_RUN_ROOT}/memory/${CAMBRIDGE_SCENE}/memory_ace_fcn_sparse_sp_r4.pt"
  local sidecar="${CAMBRIDGE_SIDECAR_DIR}/keyframe_channel.npz"
  local tag="${CAMBRIDGE_SCENE}_acefcn_if_w${wlabel}_s2g_it${CAMBRIDGE_LMC_ITERATIONS}_buf10m_${STAMP}"
  local root="${RUN_ROOT}/cambridge_acefcn/w${wlabel}"
  local log_file="${LOG_DIR}/cambridge_${CAMBRIDGE_SCENE}_w${wlabel}_gpu${GPU_CAMBRIDGE}.log"

  require_dir "${scene_root}/train/rgb" "${CAMBRIDGE_SCENE} train/rgb"
  require_file "${memory_path}" "${CAMBRIDGE_SCENE} memory"
  require_file "${sidecar}" "${CAMBRIDGE_SCENE} STGS sidecar"

  run_logged "cambridge ${CAMBRIDGE_SCENE} S2-G w=${w}" "${log_file}" \
    conda run --no-capture-output -n "${CONDA_ENV}" python ace_dinov2_lmc/train_ace_dinov2_lmc.py \
      "${scene_root}" "${tag}.pt" \
      --run_name "${tag}" \
      --model_backend ace_fcn_lmc \
      --data_backend ace \
      --post_train_eval_scene "${scene_root}" \
      --use_lmc True \
      --lmc_flow ace_g \
      --memory_path "${memory_path}" \
      --use_scale_token False \
      --ace_encoder_path "${ACE_ENCODER_PATH}" \
      --ace_lmc_global_head_mode none \
      --best_metric median_error \
      --device "cuda:${GPU_CAMBRIDGE}" \
      --post_train_eval_device "cuda:${GPU_CAMBRIDGE}" \
      --experiment_root "${root}" \
      --experiment_subdir "interframe_s2g_short" \
      --lmc_iterations "${CAMBRIDGE_LMC_ITERATIONS}" \
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
      --image_resolution "${CAMBRIDGE_IMAGE_RESOLUTION}" \
      --batch_size "${CAMBRIDGE_BATCH_SIZE}" \
      --training_buffer_size "${CAMBRIDGE_TRAINING_BUFFER_SIZE}" \
      --buffer_size_final "${CAMBRIDGE_BUFFER_SIZE_FINAL}" \
      --buffer_on_cpu True \
      --buffer_on_cpu_final True \
      --samples_per_image 512 \
      --buffer_sample_valid_coords True \
      --buffer_valid_coord_sample_ratio 1.0 \
      --buffer_valid_coord_neighbor_radius 1 \
      --buffer_valid_coord_neighbor_mode cross \
      --c1_aux_depth_root "${scene_root}/train/sparse_depth" \
      --c1_aux_depth_kind sparse_depth \
      --use_sfm_track_guided_sampling True \
      --use_sfm_track_inter_frame_loss True \
      --sfm_track_keyframe_channel_path "${sidecar}" \
      --sfm_track_sidecar_mismatch_policy strict \
      --sfm_track_guided_batch_size 512 \
      --sfm_track_guided_sampling_strategy balanced_replace \
      --sfm_track_guided_fraction 0.10 \
      --sfm_track_guided_mode inter_frame \
      --sfm_track_guided_main_loss_mode include \
      --sfm_track_guided_source_target_mode patch_center \
      --sfm_track_guided_aux_normalizer full_batch \
      --sfm_track_anchor_self_weight 0.5 \
      --sfm_track_anchor_use_alignment_weight True \
      --sfm_track_inter_frame_target_mode exact_target \
      --sfm_track_inter_frame_apply_to stage2_g \
      --sfm_track_inter_frame_weight "${w}" \
      --sfm_track_inter_frame_dropout 0.5 \
      --sfm_track_inter_frame_start_ratio 0.2 \
      --sfm_track_inter_frame_decay_last_ratio 0.3 \
      --sfm_track_inter_frame_max_px 100.0 \
      --sfm_track_max_anchor_patch_offset_px 2.0 \
      --sfm_track_max_target_patch_offset_px 0.0 \
      --sfm_track_min_alignment_weight 0.0 \
      --sfm_track_max_colmap_reproj_error_px 2.0 \
      --sfm_track_min_track_length 4 \
      --post_train_eval_seeds "${POST_TRAIN_EVAL_SEEDS[@]}" \
      --post_train_hypotheses "${POST_TRAIN_HYPOTHESES}"
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
  write_manifest
  log "run root: ${RUN_ROOT}"
  log "matrix plan: ${RUN_ROOT}/matrix_plan_${STAMP}.txt"

  local indoor_pids=()
  local w
  for w in ${INDOOR_W_VALUES}; do
    run_indoor_one_weight "${w}" &
    indoor_pids+=("$!")
  done

  build_cambridge_sidecar_if_needed &
  local cambridge_prep_pid=$!

  local pid
  for pid in "${indoor_pids[@]}"; do
    wait "${pid}"
  done
  wait "${cambridge_prep_pid}"

  run_wayspots_matrix

  for w in ${W_VALUES}; do
    run_cambridge_weight "${w}"
  done

  aggregate_results
  log "all done"
}

main "$@"
