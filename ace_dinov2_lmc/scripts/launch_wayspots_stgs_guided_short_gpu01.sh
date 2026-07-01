#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

CONDA_ENV="${CONDA_ENV:-mapanything}"
SCENE="${SCENE:-wayspots_bears}"
WAYSPOTS_ROOT="${WAYSPOTS_ROOT:-/data/xwh/Wayspots}"
SCENE_ROOT="${WAYSPOTS_ROOT}/${SCENE}"
RUN_ROOT="${RUN_ROOT:-/data/xwh/ace_dinov2_lmc/04_evaluation/wayspots/stgs_guided_screen/20260627_bears_balanced_anchor_gpu01}"
MEMORY_PATH="${MEMORY_PATH:-/data/xwh/ace_dinov2_lmc/04_evaluation/wayspots_ace_fcn_lmc_suite/20260530_181745/memory/wayspots_bears/memory_ace_fcn_sparse_sp_r4.pt}"
KEYFRAME_CHANNEL="${KEYFRAME_CHANNEL:-/data/xwh/dataset_staging/wayspots/extracted/wayspots_bears/train/colmap_keyframe_channel_v1_sp_strict/keyframe_channel.npz}"
ACE_ENCODER_PATH="${ACE_ENCODER_PATH:-${ROOT_DIR}/ace_encoder_pretrained.pt}"

LMC_ITERATIONS="${LMC_ITERATIONS:-2}"
EPOCHS="${EPOCHS:-24}"
TRAINING_BUFFER_SIZE="${TRAINING_BUFFER_SIZE:-5000000}"
BUFFER_SIZE_FINAL="${BUFFER_SIZE_FINAL:-7600000}"
IMAGE_RESOLUTION="${IMAGE_RESOLUTION:-512}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
SAMPLES_PER_IMAGE="${SAMPLES_PER_IMAGE:-512}"
GUIDED_BATCH="${GUIDED_BATCH:-512}"
GUIDED_STRATEGY="${GUIDED_STRATEGY:-balanced_replace}"
GUIDED_FRACTION="${GUIDED_FRACTION:-0.10}"
GUIDED_MAIN_LOSS_MODE="${GUIDED_MAIN_LOSS_MODE:-exclude}"
GUIDED_SOURCE_TARGET_MODE="${GUIDED_SOURCE_TARGET_MODE:-exact_anchor}"
GUIDED_AUX_NORMALIZER="${GUIDED_AUX_NORMALIZER:-terms}"
ANCHOR_SELF_WEIGHT="${ANCHOR_SELF_WEIGHT:-0.5}"
ANCHOR_USE_ALIGNMENT_WEIGHT="${ANCHOR_USE_ALIGNMENT_WEIGHT:-False}"
INTER_FRAME_WEIGHT="${INTER_FRAME_WEIGHT:-0.0}"
INTER_FRAME_DROPOUT="${INTER_FRAME_DROPOUT:-0.5}"
POST_TRAIN_HYPOTHESES="${POST_TRAIN_HYPOTHESES:-256}"
ITERATION_EVAL_HYPOTHESES="${ITERATION_EVAL_HYPOTHESES:-256}"
POST_TRAIN_EVAL_SEEDS=(${POST_TRAIN_EVAL_SEEDS:-1305 2026 4242})

mkdir -p "${RUN_ROOT}/logs"

run_one() {
  local mode="$1"
  local gpu="$2"
  local tag="${mode}_${GUIDED_STRATEGY}_f${GUIDED_FRACTION}_w${ANCHOR_SELF_WEIGHT}_iw${INTER_FRAME_WEIGHT}_drop${INTER_FRAME_DROPOUT}_main${GUIDED_MAIN_LOSS_MODE}_src${GUIDED_SOURCE_TARGET_MODE}_aux${GUIDED_AUX_NORMALIZER}_it${LMC_ITERATIONS}_buf${TRAINING_BUFFER_SIZE}_$(date +%Y%m%d_%H%M%S)"
  local log_file="${RUN_ROOT}/logs/${tag}_gpu${gpu}.log"
  local out="${SCENE}_${tag}.pt"

  local cmd=(
    conda run --no-capture-output -n "${CONDA_ENV}" python ace_dinov2_lmc/train_ace_dinov2_lmc.py
    "${SCENE_ROOT}" "${out}"
    --run_name "${SCENE}_${tag}"
    --model_backend ace_fcn_lmc
    --data_backend ace
    --use_lmc True
    --lmc_flow ace_g
    --memory_path "${MEMORY_PATH}"
    --use_scale_token False
    --ace_encoder_path "${ACE_ENCODER_PATH}"
    --ace_lmc_global_head_mode none
    --device "cuda:${gpu}"
    --post_train_eval_device "cuda:${gpu}"
    --experiment_root "${RUN_ROOT}/${mode}"
    --experiment_subdir "stage1_local_stgs_short"
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
    --c1_aux_depth_root "${SCENE_ROOT}/train/sparse_depth"
    --c1_aux_depth_kind sparse_depth
    --epochs "${EPOCHS}"
    --use_sfm_track_guided_sampling True
    --sfm_track_keyframe_channel_path "${KEYFRAME_CHANNEL}"
    --sfm_track_guided_batch_size "${GUIDED_BATCH}"
    --sfm_track_guided_sampling_strategy "${GUIDED_STRATEGY}"
    --sfm_track_guided_fraction "${GUIDED_FRACTION}"
    --sfm_track_guided_mode "${mode}"
    --sfm_track_guided_main_loss_mode "${GUIDED_MAIN_LOSS_MODE}"
    --sfm_track_guided_source_target_mode "${GUIDED_SOURCE_TARGET_MODE}"
    --sfm_track_guided_aux_normalizer "${GUIDED_AUX_NORMALIZER}"
    --sfm_track_anchor_self_weight "${ANCHOR_SELF_WEIGHT}"
    --sfm_track_anchor_use_alignment_weight "${ANCHOR_USE_ALIGNMENT_WEIGHT}"
    --sfm_track_inter_frame_weight "${INTER_FRAME_WEIGHT}"
    --sfm_track_inter_frame_dropout "${INTER_FRAME_DROPOUT}"
    --sfm_track_inter_frame_start_ratio 0.2
    --sfm_track_inter_frame_decay_last_ratio 0.3
    --sfm_track_inter_frame_max_px 100.0
    --post_train_eval_seeds "${POST_TRAIN_EVAL_SEEDS[@]}"
    --post_train_hypotheses "${POST_TRAIN_HYPOTHESES}"
  )

  {
    echo "[$(date)] mode=${mode} gpu=${gpu} strategy=${GUIDED_STRATEGY} fraction=${GUIDED_FRACTION} main=${GUIDED_MAIN_LOSS_MODE} src=${GUIDED_SOURCE_TARGET_MODE} aux_norm=${GUIDED_AUX_NORMALIZER} anchor_weight=${ANCHOR_SELF_WEIGHT} align_weight=${ANCHOR_USE_ALIGNMENT_WEIGHT} inter_weight=${INTER_FRAME_WEIGHT} run_root=${RUN_ROOT}"
    printf '%q ' "${cmd[@]}"
    printf '\n'
  } | tee -a "${log_file}"
  "${cmd[@]}" 2>&1 | tee -a "${log_file}"
}

case "${1:-}" in
  anchor_only) run_one anchor_only "${GPU:-0}" ;;
  anchor_bal_f010_w05) GUIDED_STRATEGY=balanced_replace GUIDED_FRACTION=0.10 ANCHOR_SELF_WEIGHT=0.5 INTER_FRAME_WEIGHT=0.0 run_one anchor_only "${GPU:-0}" ;;
  anchor_bal_f020_w025) GUIDED_STRATEGY=balanced_replace GUIDED_FRACTION=0.20 ANCHOR_SELF_WEIGHT=0.25 INTER_FRAME_WEIGHT=0.0 run_one anchor_only "${GPU:-1}" ;;
  anchor_v2_f010_w05) GUIDED_STRATEGY=balanced_replace GUIDED_FRACTION=0.10 GUIDED_MAIN_LOSS_MODE=include GUIDED_SOURCE_TARGET_MODE=patch_center GUIDED_AUX_NORMALIZER=full_batch ANCHOR_SELF_WEIGHT=0.5 ANCHOR_USE_ALIGNMENT_WEIGHT=True INTER_FRAME_WEIGHT=0.0 run_one anchor_only "${GPU:-0}" ;;
  anchor_v2_f015_w033) GUIDED_STRATEGY=balanced_replace GUIDED_FRACTION=0.15 GUIDED_MAIN_LOSS_MODE=include GUIDED_SOURCE_TARGET_MODE=patch_center GUIDED_AUX_NORMALIZER=full_batch ANCHOR_SELF_WEIGHT=0.33 ANCHOR_USE_ALIGNMENT_WEIGHT=True INTER_FRAME_WEIGHT=0.0 run_one anchor_only "${GPU:-1}" ;;
  inter_v2_f010_w003) GUIDED_STRATEGY=balanced_replace GUIDED_FRACTION=0.10 GUIDED_MAIN_LOSS_MODE=include GUIDED_SOURCE_TARGET_MODE=patch_center GUIDED_AUX_NORMALIZER=full_batch ANCHOR_SELF_WEIGHT=0.5 ANCHOR_USE_ALIGNMENT_WEIGHT=True INTER_FRAME_WEIGHT=0.03 INTER_FRAME_DROPOUT=0.75 run_one inter_frame "${GPU:-1}" ;;
  inter_frame) run_one inter_frame "${GPU:-1}" ;;
  same_image_shuffle) run_one same_image_shuffle "${GPU:-0}" ;;
  *) echo "usage: GPU=0 $0 {anchor_only|anchor_bal_f010_w05|anchor_bal_f020_w025|anchor_v2_f010_w05|anchor_v2_f015_w033|inter_v2_f010_w003|inter_frame|same_image_shuffle}" >&2; exit 2 ;;
esac
