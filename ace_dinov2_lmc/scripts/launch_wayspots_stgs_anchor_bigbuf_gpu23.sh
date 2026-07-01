#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

CONDA_ENV="${CONDA_ENV:-mapanything}"
WAYSPOTS_ROOT="${WAYSPOTS_ROOT:-/data/xwh/Wayspots}"
RUN_ROOT="${RUN_ROOT:-/data/xwh/ace_dinov2_lmc/04_evaluation/wayspots/stgs_guided_bigbuf/20260627_anchor_v2_it1_buf10m_gpu23}"
ACE_ENCODER_PATH="${ACE_ENCODER_PATH:-${ROOT_DIR}/ace_encoder_pretrained.pt}"
MEMORY_ROOT="${MEMORY_ROOT:-/data/xwh/ace_dinov2_lmc/04_evaluation/wayspots_ace_fcn_lmc_suite/20260530_181745/memory}"
SIDECAR_ROOT="${SIDECAR_ROOT:-/data/xwh/dataset_staging/wayspots/extracted}"

mkdir -p "${RUN_ROOT}/logs"

run_one() {
  local scene="$1"
  local gpu="$2"
  local scene_root="${WAYSPOTS_ROOT}/${scene}"
  local memory_path="${MEMORY_ROOT}/${scene}/memory_ace_fcn_sparse_sp_r4.pt"
  local keyframe_channel="${SIDECAR_ROOT}/${scene}/train/colmap_keyframe_channel_v1_sp_strict/keyframe_channel.npz"
  local tag="${scene}_anchor_v2_f010_w05_it1_buf10m_$(date +%Y%m%d_%H%M%S)"
  local log_file="${RUN_ROOT}/logs/${tag}_gpu${gpu}.log"

  local cmd=(
    conda run --no-capture-output -n "${CONDA_ENV}" python ace_dinov2_lmc/train_ace_dinov2_lmc.py
    "${scene_root}" "${tag}.pt"
    --run_name "${tag}"
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
    --experiment_root "${RUN_ROOT}/${scene}"
    --experiment_subdir stage1_local_stgs_bigbuf
    --lmc_iterations 1
    --num_latent_tokens 64
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
    --image_resolution 512
    --batch_size 4096
    --training_buffer_size 10000000
    --buffer_size_final 10000000
    --buffer_on_cpu True
    --buffer_on_cpu_final True
    --samples_per_image 512
    --buffer_sample_valid_coords True
    --buffer_valid_coord_sample_ratio 1.0
    --buffer_valid_coord_neighbor_radius 1
    --buffer_valid_coord_neighbor_mode cross
    --c1_aux_depth_root "${scene_root}/train/sparse_depth"
    --c1_aux_depth_kind sparse_depth
    --epochs 24
    --use_sfm_track_guided_sampling True
    --sfm_track_keyframe_channel_path "${keyframe_channel}"
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
    --post_train_eval_seeds 1305 2026 4242
    --post_train_hypotheses 256
  )

  {
    echo "[$(date)] scene=${scene} physical_gpu=${gpu} run_root=${RUN_ROOT}"
    printf '%q ' "${cmd[@]}"
    printf '\n'
  } | tee -a "${log_file}"
  "${cmd[@]}" 2>&1 | tee -a "${log_file}"
}

case "${1:-}" in
  bears) run_one wayspots_bears "${GPU:-2}" ;;
  cubes) run_one wayspots_cubes "${GPU:-3}" ;;
  *) echo "usage: GPU=2 $0 {bears|cubes}" >&2; exit 2 ;;
esac
