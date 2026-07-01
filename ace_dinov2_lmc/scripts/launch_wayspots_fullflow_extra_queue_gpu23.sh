#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

CONDA_ENV="${CONDA_ENV:-mapanything}"
WAYSPOTS_ROOT="${WAYSPOTS_ROOT:-/data/xwh/Wayspots}"
MEMORY_ROOT="${MEMORY_ROOT:-/data/xwh/ace_dinov2_lmc/04_evaluation/wayspots_ace_fcn_lmc_suite/20260530_181745/memory}"
ACE_ENCODER_PATH="${ACE_ENCODER_PATH:-${ROOT_DIR}/ace_encoder_pretrained.pt}"
SUPERPOINT_WEIGHTS="${SUPERPOINT_WEIGHTS:-${ROOT_DIR}/ace_dinov2_lmc/superpoint_v1.pth}"

WAIT_PID="${WAIT_PID:-513420}"
WAIT_INTERVAL_SEC="${WAIT_INTERVAL_SEC:-120}"

GPU_BASELINE="${GPU_BASELINE:-2}"
GPU_ANCHOR="${GPU_ANCHOR:-3}"
BUILD_GPU="${BUILD_GPU:-2}"

LMC_ITERATIONS="${LMC_ITERATIONS:-6}"
TRAINING_BUFFER_SIZE="${TRAINING_BUFFER_SIZE:-10000000}"
BUFFER_SIZE_FINAL="${BUFFER_SIZE_FINAL:-10000000}"
NUM_LATENT_TOKENS="${NUM_LATENT_TOKENS:-64}"
IMAGE_RESOLUTION="${IMAGE_RESOLUTION:-512}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
EPOCHS="${EPOCHS:-24}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-/data/xwh/ace_dinov2_lmc/04_evaluation/wayspots/stgs_guided_bigbuf/$(date +%Y%m%d)_stage1_it${LMC_ITERATIONS}_buf10m_fullflow_extra_gpu23}"
LOG_DIR="${RUN_ROOT}/launcher_logs"
PREPROCESS_ROOT="${RUN_ROOT}/preprocess"
SPARSE_DEPTH_SUBDIR="${SPARSE_DEPTH_SUBDIR:-sparse_depth_superpoint_strict_${STAMP}}"

mkdir -p "${LOG_DIR}" "${PREPROCESS_ROOT}"
SUPERVISOR_LOG="${LOG_DIR}/supervisor_${STAMP}.log"

if [[ "$#" -gt 0 ]]; then
  SCENES=("$@")
else
  SCENES=(
    wayspots_inscription
    wayspots_lawn
    wayspots_map
    wayspots_squarebench
    wayspots_tendrils
    wayspots_therock
  )
fi

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${SUPERVISOR_LOG}"
}

write_plan() {
  {
    echo "purpose: full-flow extra Wayspots queue after current GPU2/3 big-buffer run"
    echo "run_root: ${RUN_ROOT}"
    echo "timestamp: ${STAMP}"
    echo "wait_pid: ${WAIT_PID}"
    echo "scenes: ${SCENES[*]}"
    echo "physical_gpus: baseline=${GPU_BASELINE}, anchor=${GPU_ANCHOR}, preprocess=${BUILD_GPU}"
    echo "train_devices: baseline=cuda:${GPU_BASELINE}, anchor=cuda:${GPU_ANCHOR}"
    echo "lmc_iterations: ${LMC_ITERATIONS}"
    echo "training_buffer_size: ${TRAINING_BUFFER_SIZE}"
    echo "buffer_size_final: ${BUFFER_SIZE_FINAL}"
    echo "num_latent_tokens: ${NUM_LATENT_TOKENS}"
    echo "image_resolution: ${IMAGE_RESOLUTION}"
    echo "batch_size: ${BATCH_SIZE}"
    echo "epochs: ${EPOCHS}"
    echo "sparse_depth_subdir: ${SPARSE_DEPTH_SUBDIR}"
    echo "preprocess:"
    echo "  feature_backend: superpoint"
    echo "  matcher: pairs"
    echo "  match_window: 20"
    echo "  superpoint_conf_thresh: 0.05"
    echo "  superpoint_max_keypoints: 1024"
    echo "  superpoint_nms_dist: 8"
    echo "  superpoint_nn_thresh: 0.60"
    echo "  min_triangulation_angle: 3.0"
    echo "  keyframe_min_track_length: 4"
    echo "  keyframe_max_reproj_error: 2.0"
    echo "  keyframe_min_parallax_deg: 2.0"
    echo "  keyframe_max_anchor_alignment_px: 2.0"
    echo "anchor_v2:"
    echo "  use_sfm_track_guided_sampling: True"
    echo "  mode: anchor_only"
    echo "  fraction: 0.10"
    echo "  anchor_self_weight: 0.5"
    echo "  inter_frame_weight: 0.0"
    echo "post_train_eval_seeds: 1305,2026,4242"
    echo "post_train_hypotheses: 256"
  } >"${RUN_ROOT}/matrix_plan_${STAMP}.txt"
}

wait_for_pid() {
  if [[ -z "${WAIT_PID}" || "${WAIT_PID}" == "0" || "${WAIT_PID}" == "none" ]]; then
    log "no wait pid configured; starting immediately"
    return
  fi
  log "waiting for upstream supervisor pid=${WAIT_PID}"
  while kill -0 "${WAIT_PID}" 2>/dev/null; do
    log "upstream pid=${WAIT_PID} still alive; sleep ${WAIT_INTERVAL_SEC}s"
    sleep "${WAIT_INTERVAL_SEC}"
  done
  log "upstream pid=${WAIT_PID} finished; starting extra full-flow queue"
}

require_scene_inputs() {
  local scene="$1"
  local scene_root="${WAYSPOTS_ROOT}/${scene}"
  local memory_path="${MEMORY_ROOT}/${scene}/memory_ace_fcn_sparse_sp_r4.pt"
  [[ -d "${scene_root}/train/rgb" ]] || { log "missing train/rgb for ${scene}: ${scene_root}/train/rgb"; exit 2; }
  [[ -d "${scene_root}/train/poses" ]] || { log "missing train/poses for ${scene}: ${scene_root}/train/poses"; exit 2; }
  [[ -d "${scene_root}/train/calibration" ]] || { log "missing train/calibration for ${scene}: ${scene_root}/train/calibration"; exit 2; }
  [[ -d "${scene_root}/train/sparse_depth" ]] || { log "missing C1 sparse_depth for ${scene}: ${scene_root}/train/sparse_depth"; exit 2; }
  [[ -f "${memory_path}" ]] || { log "missing memory for ${scene}: ${memory_path}"; exit 2; }
  [[ -f "${SUPERPOINT_WEIGHTS}" ]] || { log "missing SuperPoint weights: ${SUPERPOINT_WEIGHTS}"; exit 2; }
}

run_logged() {
  local name="$1"
  local log_file="$2"
  shift 2
  log "start ${name}; log=${log_file}"
  {
    echo "[$(date)] ${name}"
    printf '%q ' "$@"
    printf '\n'
  } >>"${log_file}"
  "$@" >>"${log_file}" 2>&1
  log "done ${name}"
}

build_scene_assets() {
  local scene="$1"
  require_scene_inputs "${scene}"

  local scene_root="${WAYSPOTS_ROOT}/${scene}"
  local preprocess_dir="${PREPROCESS_ROOT}/${scene}"
  local workspace="${preprocess_dir}/superpoint_workspace"
  local model_dir="${workspace}/triangulated_model"
  local sparse_depth_dir="${scene_root}/train/${SPARSE_DEPTH_SUBDIR}"
  local sidecar_dir="${preprocess_dir}/colmap_keyframe_channel_v1_sp_strict"
  local sidecar="${sidecar_dir}/keyframe_channel.npz"
  local summary="${sidecar_dir}/summary.json"

  mkdir -p "${preprocess_dir}" "${sidecar_dir}"

  if [[ ! -f "${model_dir}/points3D.bin" ]]; then
    local build_log="${LOG_DIR}/${scene}_build_superpoint_workspace_${STAMP}.log"
    local -a build_cmd=(
      conda run --no-capture-output -n "${CONDA_ENV}" python ace_dinov2_lmc/tools/wayspots_known_pose_sparse_depth.py
      "${scene_root}"
      --split train
      --workspace "${workspace}"
      --output-subdir "${SPARSE_DEPTH_SUBDIR}"
      --feature-backend superpoint
      --superpoint-weights "${SUPERPOINT_WEIGHTS}"
      --superpoint-conf-thresh 0.05
      --superpoint-max-keypoints 1024
      --superpoint-nms-dist 8
      --superpoint-nn-thresh 0.60
      --matcher pairs
      --match-window 20
      --num-threads 4
      --min-triangulation-angle 3.0
      --max-depth-m 1000
      --use-gpu
      --gpu-index 0
      --overwrite
    )
    log "building SuperPoint known-pose workspace for ${scene} on physical gpu${BUILD_GPU}"
    (
      export CUDA_VISIBLE_DEVICES="${BUILD_GPU}"
      export PYTHONPATH="/data/xwh/SuperPointPretrainedNetwork:${PYTHONPATH:-}"
      run_logged "${scene} build_superpoint_workspace" "${build_log}" "${build_cmd[@]}"
    )
  else
    log "reuse existing workspace for ${scene}: ${model_dir}"
  fi

  if [[ ! -f "${sidecar}" ]]; then
    local channel_log="${LOG_DIR}/${scene}_build_keyframe_channel_${STAMP}.log"
    local -a channel_cmd=(
      conda run --no-capture-output -n "${CONDA_ENV}" python ace_dinov2_lmc/tools/build_colmap_keyframe_channel.py
      "${scene_root}"
      --split train
      --model-dir "${model_dir}"
      --output-dir "${sidecar_dir}"
      --image-resolution "${IMAGE_RESOLUTION}"
      --output-subsample 8
      --round-image-multiple 1
      --min-track-length 4
      --max-reproj-error 2.0
      --min-parallax-deg 2.0
      --max-parallax-deg 60.0
      --max-anchor-alignment-px 2.0
      --sparse-depth-dir "${sparse_depth_dir}"
    )
    run_logged "${scene} build_keyframe_channel" "${channel_log}" "${channel_cmd[@]}"
  else
    log "reuse existing sidecar for ${scene}: ${sidecar}"
  fi

  if [[ ! -f "${summary}" ]]; then
    log "missing sidecar summary for ${scene}: ${summary}"
    exit 3
  fi
  local rows
  rows="$(python3 -c 'import json,sys; print(int(json.load(open(sys.argv[1], encoding="utf-8")).get("rows", 0)))' "${summary}")"
  log "sidecar summary for ${scene}: rows=${rows}, sidecar=${sidecar}"
  if [[ "${rows}" -le 0 ]]; then
    log "sidecar rows <= 0 for ${scene}; aborting before training"
    exit 4
  fi
}

append_common_train_args() {
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
    --num_latent_tokens "${NUM_LATENT_TOKENS}"
    --lmc_fusion_refinement_mode single
    --lmc_fusion_cascade_layers 4
    --lmc_fusion_assembly_gamma_init 0.0
    --lmc_fusion_reread_delta_alpha 1.0
    --lmc_fusion_reread_scalar_gate False
    --lmc_fusion_reread_gate_init 0.0
    --lmc_fusion_reread_post_norm True
    --lmc_fusion_reread_trust_region_ratio 0.0
    --lmc_fusion_reread_temperature 1.0
    --lmc_fusion_reread_common_scale 1.0
    --lmc_fusion_reread_effective_ratio_cap 0.0
    --lmc_fusion_reread_qknorm_eps 1e-6
    --lmc_fusion_reread_qknorm_tau_init 0.0
    --lmc_fusion_reread_layerscale_patch_init 0.01
    --lmc_fusion_reread_layerscale_common_init 0.0
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
    --iteration_eval_hypotheses 256
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
    --samples_per_image 512
    --buffer_sample_valid_coords True
    --buffer_valid_coord_sample_ratio 1.0
    --buffer_valid_coord_neighbor_radius 1
    --buffer_valid_coord_neighbor_mode cross
    --c1_aux_depth_root "${scene_root}/train/sparse_depth"
    --c1_aux_depth_kind sparse_depth
    --epochs "${EPOCHS}"
    --post_train_eval_seeds 1305 2026 4242
    --post_train_hypotheses 256
  )
}

append_anchor_args() {
  local -n out="$1"
  local sidecar="$2"
  out+=(
    --use_sfm_track_guided_sampling True
    --sfm_track_keyframe_channel_path "${sidecar}"
    --sfm_track_guided_batch_size 512
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
}

run_train() {
  local scene="$1"
  local mode="$2"
  local device="$3"
  local physical_gpu="$4"
  local sidecar="$5"
  local scene_root="${WAYSPOTS_ROOT}/${scene}"
  local tag="${scene}_${mode}_it${LMC_ITERATIONS}_buf10m_fullflow_${STAMP}"
  local exp_subdir="stage1_local_ace_memory_it${LMC_ITERATIONS}_bigbuf"
  local exp_root="${RUN_ROOT}/${scene}/${mode}"
  local log_file="${LOG_DIR}/${tag}_physgpu${physical_gpu}.log"

  if [[ "${mode}" == "anchor_v2_f010_w05" ]]; then
    exp_subdir="stage1_local_stgs_anchor_bigbuf_it${LMC_ITERATIONS}"
  fi

  local -a cmd=(
    conda run --no-capture-output -n "${CONDA_ENV}" python ace_dinov2_lmc/train_ace_dinov2_lmc.py
    "${scene_root}" "${tag}.pt"
    --run_name "${tag}"
    --device "cuda:${device}"
    --post_train_eval_device "cuda:${device}"
    --experiment_root "${exp_root}"
    --experiment_subdir "${exp_subdir}"
  )
  append_common_train_args cmd "${scene}"
  if [[ "${mode}" == "anchor_v2_f010_w05" ]]; then
    append_anchor_args cmd "${sidecar}"
  fi

  {
    echo "[$(date)] scene=${scene} mode=${mode} physical_gpu=${physical_gpu} device=cuda:${device} run_root=${RUN_ROOT}"
    printf '%q ' "${cmd[@]}"
    printf '\n'
  } >>"${log_file}"

  log "start ${scene} ${mode} on physical gpu${physical_gpu} device cuda:${device}; log=${log_file}"
  env -u CUDA_VISIBLE_DEVICES "${cmd[@]}" >>"${log_file}" 2>&1
  log "done ${scene} ${mode}"
}

launch_training_pair() {
  local scene="$1"
  local sidecar="${PREPROCESS_ROOT}/${scene}/colmap_keyframe_channel_v1_sp_strict/keyframe_channel.npz"
  [[ -f "${sidecar}" ]] || { log "missing sidecar before training ${scene}: ${sidecar}"; exit 5; }

  log "launch pair for ${scene}: baseline physical gpu${GPU_BASELINE}, anchor physical gpu${GPU_ANCHOR}"
  run_train "${scene}" baseline "${GPU_BASELINE}" "${GPU_BASELINE}" "${sidecar}" &
  local pid_baseline=$!
  run_train "${scene}" anchor_v2_f010_w05 "${GPU_ANCHOR}" "${GPU_ANCHOR}" "${sidecar}" &
  local pid_anchor=$!

  local rc_baseline=0
  local rc_anchor=0
  wait "${pid_baseline}" || rc_baseline=$?
  wait "${pid_anchor}" || rc_anchor=$?

  if [[ "${rc_baseline}" -ne 0 || "${rc_anchor}" -ne 0 ]]; then
    log "pair failed for ${scene}: baseline_rc=${rc_baseline}, anchor_rc=${rc_anchor}"
    exit 6
  fi
  log "pair complete for ${scene}"
}

write_plan
log "full-flow queue plan: ${RUN_ROOT}/matrix_plan_${STAMP}.txt"
log "queued scenes: ${SCENES[*]}"
log "physical GPU policy: preprocess=${BUILD_GPU}, baseline=${GPU_BASELINE}, anchor=${GPU_ANCHOR}"

wait_for_pid

for scene in "${SCENES[@]}"; do
  log "begin full-flow scene ${scene}"
  build_scene_assets "${scene}"
  launch_training_pair "${scene}"
  log "finish full-flow scene ${scene}"
done

log "all full-flow extra queue jobs complete"
