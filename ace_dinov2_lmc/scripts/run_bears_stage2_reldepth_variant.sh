#!/usr/bin/env bash
set -euo pipefail

variant="${1:?usage: $0 <variant> <gpu> <use_relative_depth>}"
gpu="${2:?usage: $0 <variant> <gpu> <use_relative_depth>}"
use_relative_depth="${3:?usage: $0 <variant> <gpu> <use_relative_depth>}"

ROOT=/home/xwh/project/ace_depth
PY=/home/xwh/miniforge3/envs/mapanything/bin/python
TRAIN=${ROOT}/ace_dinov2_lmc/train_ace_dinov2_lmc.py
SCENE=/data/xwh/Wayspots/wayspots_bears
MEM=/data/xwh/ace_dinov2_lmc/04_evaluation/wayspots_ace_fcn_lmc_suite/20260530_181745/memory/wayspots_bears/memory_ace_fcn_sparse_sp_r4.pt
STAGE1=/data/xwh/ace_dinov2_lmc/04_evaluation/stage1_bears_matrix_20260622/stage1_single_legacy/extracted/wayspots_bears/dino_ace_lmc_ace_g/20260622_190313_aceg_fS2cie_global_res512_buf2.8M_F7.6M_K64_it12_ep24_bs4096_spi_s1buf_sp512_onecycle_improved/best_K64_it12_ace_fcn_local_stage1_stage1_single_legacy.pt
TEACHER=/home/xwh/data/checkpoints/depth_anything_v2_vitb.pth
BASE=/data/xwh/ace_dinov2_lmc/04_evaluation/stage2_bears_reldepth_ab_20260622
log=${BASE}/logs/${variant}_cuda${gpu}.tmux.log

for path in "${TRAIN}" "${MEM}" "${STAGE1}" "${TEACHER}" "${SCENE}/train/features.npy"; do
  if [[ ! -e "${path}" ]]; then
    echo "Missing required input: ${path}" >&2
    exit 2
  fi
done

mkdir -p "${BASE}/logs"

relative_depth_args=(--use_relative_depth_loss False)
if [[ "${use_relative_depth}" == "true" ]]; then
  relative_depth_args=(
    --use_relative_depth_loss True
    --relative_depth_apply_to stage2_g
    --relative_depth_teacher depth_anything_v2_online
    --relative_depth_teacher_encoder vitb
    --relative_depth_teacher_checkpoint "${TEACHER}"
    --relative_depth_teacher_input_size 518
    --relative_depth_loss_weight 0.05
    --relative_depth_start_lmc_ratio 0.30
    --relative_depth_start_ratio 0.30
    --relative_depth_pair_weight 0.50
    --relative_depth_max_samples 1024
    --relative_depth_max_pairs 4096
    --relative_depth_min_points 16
    --relative_depth_teacher_cache_size 256
    --relative_depth_student_space neg_z_legacy
    --relative_depth_reprojection_gate hard
    --relative_depth_reprojection_threshold_px 4.0
    --relative_depth_min_ray_coverage 0.05
    --relative_depth_log_reprojection_stats True
    --relative_depth_image_step_interval 10
    --relative_depth_image_batch_size 1
  )
fi

{
  echo "[$(date +%Y-%m-%dT%H:%M:%S)] start variant=${variant} gpu=${gpu} relative_depth=${use_relative_depth}"
  env ACE_DATA_ROOT=/home/xwh/data PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${PY}" "${TRAIN}" "${SCENE}" "ace_fcn_glace_stage2_${variant}.pt" \
      --model_backend ace_fcn_lmc --data_backend ace --use_lmc True --lmc_flow ace_g \
      --memory_path "${MEM}" --use_scale_token False --ace_encoder_path "${ROOT}/ace_encoder_pretrained.pt" \
      --ace_lmc_global_head_mode glace_concat --ace_lmc_local_checkpoint_path "${STAGE1}" \
      --ace_lmc_freeze_local_stack True --glace_feat_name features.npy \
      --device "cuda:${gpu}" --post_train_eval_device "cuda:${gpu}" \
      --experiment_root "${BASE}" --experiment_subdir "${variant}" \
      --lmc_iterations 12 --num_latent_tokens 64 \
      --lmc_fusion_refinement_mode single --lmc_fusion_cascade_layers 4 \
      --lmc_fusion_assembly_gamma_init 0.0 --lmc_fusion_reread_delta_alpha 0.10 \
      --lmc_fusion_reread_scalar_gate False --lmc_fusion_reread_gate_init -2.0 \
      --lmc_fusion_reread_post_norm False --lmc_fusion_reread_trust_region_ratio 0.0 \
      --lmc_fusion_reread_temperature 1.0 --lmc_fusion_reread_common_scale 1.0 \
      --lmc_fusion_reread_effective_ratio_cap 0.0 --lmc_fusion_reread_warmup_mode none \
      --lmc_fusion_reread_warmup_iters 0 --lmc_fusion_reread_warmup_start 0.0 \
      --s1_loss_step_mode fixed_zero --s1_early_stop False \
      --lmc_log_runtime_stats True --lmc_runtime_stats_interval 100 --lmc_runtime_stats_max_pixels 4096 \
      --num_data_loader_workers 12 --eval_num_workers 6 --eval_deterministic True \
      --eval_dsacstar_seed 1305 --eval_dsacstar_seed_per_frame True \
      --iteration_eval_seed 1305 --iteration_eval_hypotheses 64 \
      --ace_g_fusion_in_s2 True --ace_g_cross_iter_eval True \
      --s1_use_buffer True --s1_loss_mode sample_per_image --s1_buffer_refill_mode full \
      --image_resolution 512 --batch_size 4096 --training_buffer_size 2800000 \
      --buffer_size_final 7600000 --buffer_on_cpu True --buffer_on_cpu_final True \
      --samples_per_image 512 --buffer_sample_valid_coords True \
      --buffer_valid_coord_sample_ratio 1.0 --buffer_valid_coord_neighbor_radius 1 \
      --buffer_valid_coord_neighbor_mode cross --c1_aux_depth_root "${SCENE}/train/sparse_depth" \
      --c1_aux_depth_kind sparse_depth --post_train_eval_seeds 1305 2026 4242 \
      --post_train_hypotheses 256 "${relative_depth_args[@]}"
  code=$?
  echo "[$(date +%Y-%m-%dT%H:%M:%S)] exit variant=${variant} code=${code}"
  exit "${code}"
} >"${log}" 2>&1
