#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

CONDA_ENV="${CONDA_ENV:-mapanything}"
WAYSPOTS_ROOT="${WAYSPOTS_ROOT:-/data/xwh/Wayspots}"
MEMORY_ROOT="${MEMORY_ROOT:-/data/xwh/ace_dinov2_lmc/04_evaluation/wayspots_ace_fcn_lmc_suite/20260530_181745/memory}"
SIDECAR_ROOT="${SIDECAR_ROOT:-/data/xwh/dataset_staging/wayspots/extracted}"
ACE_ENCODER_PATH="${ACE_ENCODER_PATH:-${ROOT_DIR}/ace_encoder_pretrained.pt}"

LMC_ITERATIONS="${LMC_ITERATIONS:-6}"
TRAINING_BUFFER_SIZE="${TRAINING_BUFFER_SIZE:-10000000}"
BUFFER_SIZE_FINAL="${BUFFER_SIZE_FINAL:-10000000}"
NUM_LATENT_TOKENS="${NUM_LATENT_TOKENS:-64}"
IMAGE_RESOLUTION="${IMAGE_RESOLUTION:-512}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
EPOCHS="${EPOCHS:-24}"

GPU_BASELINE="${GPU_BASELINE:-2}"
GPU_ANCHOR="${GPU_ANCHOR:-3}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-/data/xwh/ace_dinov2_lmc/04_evaluation/wayspots/stgs_guided_bigbuf/$(date +%Y%m%d)_stage1_it${LMC_ITERATIONS}_buf10m_baseline_anchor_gpu23}"
LOG_DIR="${RUN_ROOT}/launcher_logs"
mkdir -p "${LOG_DIR}"

SUPERVISOR_LOG="${LOG_DIR}/supervisor_${STAMP}.log"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${SUPERVISOR_LOG}"
}

write_plan() {
  cat >"${RUN_ROOT}/matrix_plan_${STAMP}.txt" <<EOF
purpose: paired multi-iteration big-buffer Wayspots comparison
run_root: ${RUN_ROOT}
timestamp: ${STAMP}
lmc_iterations: ${LMC_ITERATIONS}
training_buffer_size: ${TRAINING_BUFFER_SIZE}
buffer_size_final: ${BUFFER_SIZE_FINAL}
num_latent_tokens: ${NUM_LATENT_TOKENS}
image_resolution: ${IMAGE_RESOLUTION}
batch_size: ${BATCH_SIZE}
epochs: ${EPOCHS}
post_train_eval_seeds: 1305,2026,4242
post_train_hypotheses: 256
gpu_policy: baseline on physical GPU ${GPU_BASELINE}, anchor-v2 on physical GPU ${GPU_ANCHOR}
queue:
  1. wayspots_bears baseline vs anchor_v2
  2. wayspots_cubes baseline vs anchor_v2
anchor_v2:
  use_sfm_track_guided_sampling: True
  sfm_track_guided_mode: anchor_only
  sfm_track_guided_fraction: 0.10
  sfm_track_anchor_self_weight: 0.5
  sfm_track_anchor_use_alignment_weight: True
  sfm_track_guided_source_target_mode: patch_center
  sfm_track_guided_aux_normalizer: full_batch
  sfm_track_inter_frame_weight: 0.0
EOF
}

common_args() {
  local scene="$1"
  local scene_root="${WAYSPOTS_ROOT}/${scene}"
  local memory_path="${MEMORY_ROOT}/${scene}/memory_ace_fcn_sparse_sp_r4.pt"

  printf '%s\0' \
    --model_backend ace_fcn_lmc \
    --data_backend ace \
    --use_lmc True \
    --lmc_flow ace_g \
    --memory_path "${memory_path}" \
    --use_scale_token False \
    --ace_encoder_path "${ACE_ENCODER_PATH}" \
    --ace_lmc_global_head_mode none \
    --lmc_iterations "${LMC_ITERATIONS}" \
    --num_latent_tokens "${NUM_LATENT_TOKENS}" \
    --lmc_fusion_refinement_mode single \
    --lmc_fusion_cascade_layers 4 \
    --lmc_fusion_assembly_gamma_init 0.0 \
    --lmc_fusion_reread_delta_alpha 1.0 \
    --lmc_fusion_reread_scalar_gate False \
    --lmc_fusion_reread_gate_init 0.0 \
    --lmc_fusion_reread_post_norm True \
    --lmc_fusion_reread_trust_region_ratio 0.0 \
    --lmc_fusion_reread_temperature 1.0 \
    --lmc_fusion_reread_common_scale 1.0 \
    --lmc_fusion_reread_effective_ratio_cap 0.0 \
    --lmc_fusion_reread_qknorm_eps 1e-6 \
    --lmc_fusion_reread_qknorm_tau_init 0.0 \
    --lmc_fusion_reread_layerscale_patch_init 0.01 \
    --lmc_fusion_reread_layerscale_common_init 0.0 \
    --lmc_fusion_reread_warmup_mode none \
    --lmc_fusion_reread_warmup_iters 0 \
    --lmc_fusion_reread_warmup_start 0.0 \
    --lmc_fusion_reread_geo_lambda 1.0 \
    --lmc_fusion_reread_geo_sigma 1.0 \
    --lmc_fusion_reread_geo_sigma_mode fixed \
    --lmc_fusion_reread_geo_sigma_beta 1.0 \
    --lmc_fusion_reread_geo_sigma_min 0.5 \
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
    --ace_g_fusion_in_s2 True \
    --ace_g_cross_iter_eval True \
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
    --c1_aux_depth_root "${scene_root}/train/sparse_depth" \
    --c1_aux_depth_kind sparse_depth \
    --epochs "${EPOCHS}" \
    --post_train_eval_seeds 1305 2026 4242 \
    --post_train_hypotheses 256
}

anchor_args() {
  local scene="$1"
  local keyframe_channel="${SIDECAR_ROOT}/${scene}/train/colmap_keyframe_channel_v1_sp_strict/keyframe_channel.npz"

  printf '%s\0' \
    --use_sfm_track_guided_sampling True \
    --sfm_track_keyframe_channel_path "${keyframe_channel}" \
    --sfm_track_guided_batch_size 512 \
    --sfm_track_guided_sampling_strategy balanced_replace \
    --sfm_track_guided_fraction 0.10 \
    --sfm_track_guided_mode anchor_only \
    --sfm_track_guided_main_loss_mode include \
    --sfm_track_guided_source_target_mode patch_center \
    --sfm_track_guided_aux_normalizer full_batch \
    --sfm_track_anchor_self_weight 0.5 \
    --sfm_track_anchor_use_alignment_weight True \
    --sfm_track_inter_frame_weight 0.0 \
    --sfm_track_inter_frame_dropout 0.5 \
    --sfm_track_inter_frame_start_ratio 0.2 \
    --sfm_track_inter_frame_decay_last_ratio 0.3 \
    --sfm_track_inter_frame_max_px 100.0
}

run_train() {
  local scene="$1"
  local mode="$2"
  local gpu="$3"
  local scene_root="${WAYSPOTS_ROOT}/${scene}"
  local tag="${scene}_${mode}_it${LMC_ITERATIONS}_buf10m_${STAMP}"
  local exp_subdir="stage1_local_ace_memory_it${LMC_ITERATIONS}_bigbuf"
  local exp_root="${RUN_ROOT}/${scene}/${mode}"
  local log_file="${LOG_DIR}/${tag}_gpu${gpu}.log"

  if [[ "${mode}" == "anchor_v2_f010_w05" ]]; then
    exp_subdir="stage1_local_stgs_anchor_bigbuf_it${LMC_ITERATIONS}"
  fi

  local -a cmd=(
    conda run --no-capture-output -n "${CONDA_ENV}" python ace_dinov2_lmc/train_ace_dinov2_lmc.py
    "${scene_root}" "${tag}.pt"
    --run_name "${tag}"
    --device "cuda:${gpu}"
    --post_train_eval_device "cuda:${gpu}"
    --experiment_root "${exp_root}"
    --experiment_subdir "${exp_subdir}"
  )

  local -a common=()
  local item
  while IFS= read -r -d '' item; do
    common+=("${item}")
  done < <(common_args "${scene}")
  cmd+=("${common[@]}")

  if [[ "${mode}" == "anchor_v2_f010_w05" ]]; then
    local -a anchor=()
    while IFS= read -r -d '' item; do
      anchor+=("${item}")
    done < <(anchor_args "${scene}")
    cmd+=("${anchor[@]}")
  fi

  {
    echo "[$(date)] scene=${scene} mode=${mode} physical_gpu=${gpu} run_root=${RUN_ROOT}"
    printf '%q ' "${cmd[@]}"
    printf '\n'
  } >>"${log_file}"

  log "start ${scene} ${mode} on gpu${gpu}; log=${log_file}"
  "${cmd[@]}" >>"${log_file}" 2>&1
  log "done ${scene} ${mode} on gpu${gpu}"
}

launch_pair() {
  local scene="$1"
  log "launch pair for ${scene}: baseline gpu${GPU_BASELINE}, anchor-v2 gpu${GPU_ANCHOR}"
  run_train "${scene}" baseline "${GPU_BASELINE}" &
  local pid_baseline=$!
  run_train "${scene}" anchor_v2_f010_w05 "${GPU_ANCHOR}" &
  local pid_anchor=$!

  local rc_baseline=0
  local rc_anchor=0
  wait "${pid_baseline}" || rc_baseline=$?
  wait "${pid_anchor}" || rc_anchor=$?

  if [[ "${rc_baseline}" -ne 0 || "${rc_anchor}" -ne 0 ]]; then
    log "pair failed for ${scene}: baseline_rc=${rc_baseline}, anchor_rc=${rc_anchor}"
    exit 1
  fi
  log "pair complete for ${scene}"
}

write_plan
log "matrix plan: ${RUN_ROOT}/matrix_plan_${STAMP}.txt"
log "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
log "queue starts: it${LMC_ITERATIONS}, buffer=${TRAINING_BUFFER_SIZE}, final_buffer=${BUFFER_SIZE_FINAL}"

launch_pair wayspots_bears
launch_pair wayspots_cubes

log "all matrix jobs complete"
