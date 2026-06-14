#!/usr/bin/env bash
# Scene2a diagnostic run:
# force-global ACE-G with the same legacy S1 behavior as the current scene2a
# baseline, plus relative-depth distillation in S2-G.
#
# This isolates whether a monocular relative-depth regularizer improves the
# scene2a global ACE-G route without changing memory, K/iters/resolution/buffer,
# or the legacy S1 loss-step behavior.
#
# Run:
#   cd /home/xwh/project/ace_depth
#   conda activate mapanything
#   GPU_ID=0 bash ace_dinov2_lmc/scripts/run_indoor6_scene2a_dino_forceglobal_legacy_s1_reldepth_gpu0.sh

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/xwh/project/ace_depth}"
REPO_ROOT="${REPO_ROOT:-${ROOT_DIR}/ace_dinov2_lmc}"
DATA_ROOT="${ACE_DATA_ROOT:-/home/xwh/data}"
ACE_ROOT="${ACE_ROOT:-${DATA_ROOT}/indoor6_ace}"
WAI_ROOT="${WAI_ROOT:-${DATA_ROOT}/mapanything-dataset/wai_data/indoor6}"

SCENE="${SCENE:-scene2a}"
GPU_ID="${GPU_ID:-0}"
DEVICE="cuda:${GPU_ID}"
PYTHON_BIN="${PYTHON_BIN:-/home/xwh/miniforge3/envs/mapanything/bin/python}"

EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-/data/xwh/ace_dinov2_lmc/04_evaluation/train_compare}"
EXPERIMENT_SUBDIR="${EXPERIMENT_SUBDIR:-indoor6_full_baselines_4090_forceglobal_legacy_s1_reldepth_imagelevel_scene2a}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/04_evaluation/train_compare/run_logs/forceglobal_legacy_s1_reldepth_imagelevel_${SCENE}_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${RUN_ROOT}/logs"
RESULT_ROOT="${RUN_ROOT}/results"

MEMORY_PATH="${MEMORY_PATH:-${REPO_ROOT}/memory_extraction/04_evaluation/memory_extract/${SCENE}/40v_v0.05_bilinear_bse_ut0.02_noaug_asb_adaptive_pool1.0_rgate4_prepair8_sor_gm_l2/20260508_103423/memory_bse.pt}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-10240}"
TRAINING_BUFFER_SIZE="${TRAINING_BUFFER_SIZE:-2560000}"
BUFFER_SIZE_FINAL="${BUFFER_SIZE_FINAL:-7680000}"
BUFFER_ON_CPU="${BUFFER_ON_CPU:-true}"
BUFFER_ON_CPU_FINAL="${BUFFER_ON_CPU_FINAL:-true}"
POST_TRAIN_SEEDS_STR="${POST_TRAIN_SEEDS_STR:-1305 2026 4242 7777 9001}"
POST_TRAIN_HYPOTHESES="${POST_TRAIN_HYPOTHESES:-256}"
EVAL_EACH_ITERATION="${EVAL_EACH_ITERATION:-true}"
EVAL_AFTER_TRAIN="${EVAL_AFTER_TRAIN:-true}"
DRY_RUN="${DRY_RUN:-false}"
REL_DEPTH_TEACHER_CHECKPOINT="${REL_DEPTH_TEACHER_CHECKPOINT:-/home/xwh/data/checkpoints/depth_anything_v2_vitb.pth}"
REL_DEPTH_TEACHER_ENCODER="${REL_DEPTH_TEACHER_ENCODER:-vitb}"
REL_DEPTH_LOSS_WEIGHT="${REL_DEPTH_LOSS_WEIGHT:-0.03}"
REL_DEPTH_START_RATIO="${REL_DEPTH_START_RATIO:-0.30}"
REL_DEPTH_PAIR_WEIGHT="${REL_DEPTH_PAIR_WEIGHT:-0.50}"
REL_DEPTH_MAX_SAMPLES="${REL_DEPTH_MAX_SAMPLES:-1024}"
REL_DEPTH_MAX_PAIRS="${REL_DEPTH_MAX_PAIRS:-4096}"
REL_DEPTH_MIN_POINTS="${REL_DEPTH_MIN_POINTS:-16}"
REL_DEPTH_IMAGE_STEP_INTERVAL="${REL_DEPTH_IMAGE_STEP_INTERVAL:-10}"
REL_DEPTH_IMAGE_BATCH_SIZE="${REL_DEPTH_IMAGE_BATCH_SIZE:-1}"
LMC_ITERATIONS="${LMC_ITERATIONS:-28}"
LMC_WARMUP_STEPS="${LMC_WARMUP_STEPS:-2000}"
LMC_TRAIN_STEPS="${LMC_TRAIN_STEPS:-600}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-24}"

mkdir -p "$LOG_ROOT" "$RESULT_ROOT"
read -r -a POST_TRAIN_SEEDS <<< "$POST_TRAIN_SEEDS_STR"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

require_inputs() {
  if [ ! -d "${ACE_ROOT}/${SCENE}" ]; then
    echo "ERROR: missing ACE scene root: ${ACE_ROOT}/${SCENE}" >&2
    return 1
  fi
  if [ ! -d "${WAI_ROOT}/${SCENE}_train" ]; then
    echo "ERROR: missing WAI aux depth root: ${WAI_ROOT}/${SCENE}_train" >&2
    return 1
  fi
  if [ ! -f "$MEMORY_PATH" ]; then
    echo "ERROR: missing memory: ${MEMORY_PATH}" >&2
    return 1
  fi
  if [ ! -f "$REL_DEPTH_TEACHER_CHECKPOINT" ]; then
    echo "ERROR: missing relative-depth teacher checkpoint: ${REL_DEPTH_TEACHER_CHECKPOINT}" >&2
    return 1
  fi
}

main() {
  require_inputs

  local ts output_name log_file result_file
  ts="$(date +%Y%m%d_%H%M%S)"
  output_name="${SCENE}_forceglobal_legacy_s1_reldepth_imagelevel_${ts}.pt"
  log_file="${LOG_ROOT}/train_${SCENE}_gpu${GPU_ID}.log"
  result_file="${RESULT_ROOT}/train_${SCENE}.txt"

  log "Scene           : ${SCENE}"
  log "GPU             : ${GPU_ID}"
  log "Experiment root : ${EXPERIMENT_ROOT}"
  log "Experiment subdir: ${EXPERIMENT_SUBDIR}"
  log "Memory          : ${MEMORY_PATH}"
  log "Aux depth       : ${WAI_ROOT}/${SCENE}_train (gt_depth)"
  log "Config          : force-global, legacy S1 loss-step, image-level relative-depth S2-G w=${REL_DEPTH_LOSS_WEIGHT}, interval=${REL_DEPTH_IMAGE_STEP_INTERVAL}, K64 it=${LMC_ITERATIONS} ep=${TRAIN_EPOCHS} res518 buf=${TRAINING_BUFFER_SIZE}/${BUFFER_SIZE_FINAL}"
  log "RelDepth       : ${REL_DEPTH_TEACHER_ENCODER} ${REL_DEPTH_TEACHER_CHECKPOINT}"
  log "Log             : ${log_file}"
  log "Dry run         : ${DRY_RUN}"

  if [ "$DRY_RUN" = "true" ]; then
    return 0
  fi

  ACE_DATA_ROOT="$DATA_ROOT" \
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  "$PYTHON_BIN" "${REPO_ROOT}/train_ace_dinov2_lmc.py" \
    "${ACE_ROOT}/${SCENE}" \
    "$output_name" \
    --train_preset memory_compare_ace_g_v1 \
    --data_backend ace \
    --wai_repo_root "${DATA_ROOT}/map-anything" \
    --device "$DEVICE" \
    --post_train_eval_device "$DEVICE" \
    --use_lmc True \
    --lmc_flow ace_g \
    --lmc_profile legacy \
    --apply_baseline_contract True \
    --vanilla_iterations 1 \
    --memory_path "$MEMORY_PATH" \
    --lmc_memory_preflight True \
    --lmc_memory_preflight_strict True \
    --lmc_memory_preflight_center_tol 5.0 \
    --lmc_strict_scene_check True \
    --lmc_strict_center_check True \
    --lmc_scene_center_max_distance 4.0 \
    --bse_denorm_to_world True \
    --c1_ref_norm_alpha 1.0 \
    --c1_aux_ref_loss_weight 0.0 \
    --c1_aux_depth_root "${WAI_ROOT}/${SCENE}_train" \
    --c1_aux_depth_kind gt_depth \
    --c1_aux_ref_sample_ratio 0.5 \
    --use_relative_depth_loss True \
    --relative_depth_apply_to stage2_g \
    --relative_depth_teacher depth_anything_v2_online \
    --relative_depth_teacher_encoder "$REL_DEPTH_TEACHER_ENCODER" \
    --relative_depth_teacher_checkpoint "$REL_DEPTH_TEACHER_CHECKPOINT" \
    --relative_depth_teacher_input_size 518 \
    --relative_depth_loss_weight "$REL_DEPTH_LOSS_WEIGHT" \
    --relative_depth_start_ratio "$REL_DEPTH_START_RATIO" \
    --relative_depth_pair_weight "$REL_DEPTH_PAIR_WEIGHT" \
    --relative_depth_max_samples "$REL_DEPTH_MAX_SAMPLES" \
    --relative_depth_max_pairs "$REL_DEPTH_MAX_PAIRS" \
    --relative_depth_min_points "$REL_DEPTH_MIN_POINTS" \
    --relative_depth_teacher_cache_size 256 \
    --relative_depth_image_step_interval "$REL_DEPTH_IMAGE_STEP_INTERVAL" \
    --relative_depth_image_batch_size "$REL_DEPTH_IMAGE_BATCH_SIZE" \
    --lmc_head_mean_max_shift 4.0 \
    --lmc_strict_head_mean_alignment True \
    --ace_g_fusion_in_s2 True \
    --ace_g_fusion_lr_ratio 0.01 \
    --ace_g_cross_iter_eval True \
    --pe_normalize_input False \
    --lmc_mode global \
    --lmc_fps_start_policy farthest_from_center \
    --lmc_auto_mode_by_visibility False \
    --lmc_visibility_front_ratio_threshold 0.85 \
    --lmc_visibility_fallback_mode local \
    --lmc_visibility_sample_points 4096 \
    --lmc_key_slice_idx 2 \
    --num_latent_tokens 64 \
    --num_attn_layers 4 \
    --s1_batch_size 16 \
    --s1_use_buffer True \
    --s1_buffer_refill_mode full \
    --s1_buffer_keep_ratio 0.5 \
    --use_scale_token True \
    --lmc_iterations "$LMC_ITERATIONS" \
    --lmc_train_steps "$LMC_TRAIN_STEPS" \
    --lmc_warmup_steps "$LMC_WARMUP_STEPS" \
    --s1_early_stop True \
    --s1_early_stop_min_updates 400 \
    --s1_early_stop_patience 180 \
    --s1_early_stop_rel_improve 0.01 \
    --s1_early_stop_ema_beta 0.9 \
    --s1_loss_mode sample_per_image \
    --s1_full_map_max_points 0 \
    --s1_empty_cache_interval 30 \
    --training_buffer_size "$TRAINING_BUFFER_SIZE" \
    --buffer_size_final "$BUFFER_SIZE_FINAL" \
    --buffer_on_cpu "$BUFFER_ON_CPU" \
    --buffer_on_cpu_final "$BUFFER_ON_CPU_FINAL" \
    --buffer_sample_valid_coords True \
    --buffer_valid_coord_sample_ratio 1.0 \
    --buffer_valid_coord_neighbor_radius 1 \
    --buffer_valid_coord_neighbor_mode cross \
    --buffer_sampling_replacement True \
    --head_reset_strategy first_only \
    --head_lr_multiplier_s2 1.5 \
    --lmc_lr_scheduler_type onecycle_improved \
    --lmc_min_lr_ratio 0.01 \
    --s1_lr_scale_later 0.2 \
    --lmc_warmup_ratio 0.1 \
    --lmc_plateau_ratio 0.35 \
    --lmc_lr_pct_start 0.3 \
    --lmc_lr_div_factor 25.0 \
    --lmc_lr_final_div_factor 10000.0 \
    --geo_sigma 0.5 \
    --num_fine 128 \
    --num_coarse 16 \
    --s2_learning_rate_max 0.001 \
    --s2_repro_rewind_tau 300.0 \
    --s2_lr_warmup_steps 0 \
    --s2_step_eff_mode auto \
    --s2_grad_clip_max_norm 1.0 \
    --loss_invalid_max_delta 1000.0 \
    --s2_polish_epochs 0 \
    --s2_polish_head_lr 0.0001 \
    --s2_polish_fusion_lr_ratio 0.005 \
    --image_resolution 518 \
    --batch_size "$TRAIN_BATCH_SIZE" \
    --epochs "$TRAIN_EPOCHS" \
    --eval_each_iteration "$EVAL_EACH_ITERATION" \
    --keep_best_only True \
    --best_metric pct5 \
    --eval_after_train "$EVAL_AFTER_TRAIN" \
    --eval_session post_train \
    --eval_deterministic False \
    --eval_dsacstar_seed 1305 \
    --eval_dsacstar_seed_per_frame True \
    --eval_num_workers 6 \
    --post_train_eval_seeds "${POST_TRAIN_SEEDS[@]}" \
    --post_train_hypotheses "$POST_TRAIN_HYPOTHESES" \
    --s2_repro_rewind_first_ratio 0.2 \
    --s2_repro_rewind_later_ratio 0.08 \
    --s2_lr_boost_first 1.2 \
    --s2_lr_boost_later 1.0 \
    --experiment_root "$EXPERIMENT_ROOT" \
    --experiment_subdir "$EXPERIMENT_SUBDIR" \
    --run_name auto \
    --output_layout hierarchical \
    --overwrite_run_dir True \
    > "$log_file" 2>&1

  local train_root run_dir best_file
  train_root="${EXPERIMENT_ROOT}/${EXPERIMENT_SUBDIR}/indoor6_ace/${SCENE}/dino_ace_lmc_ace_g"
  run_dir="$(find "$train_root" -type f -name run_config.json -printf '%T@ %h\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2- || true)"
  best_file="$(find "$train_root" -type f -name 'best_*.pt' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2- || true)"

  {
    echo "status=ok"
    echo "scene=${SCENE}"
    echo "gpu=${GPU_ID}"
    echo "memory_path=${MEMORY_PATH}"
    echo "run_dir=${run_dir:-unknown}"
    echo "best_file=${best_file:-unknown}"
    echo "log=${log_file}"
    echo "s1_loss_step_mode=legacy_default_not_passed"
    echo "use_relative_depth_loss=true"
    echo "relative_depth_apply_to=stage2_g"
    echo "relative_depth_loss_weight=${REL_DEPTH_LOSS_WEIGHT}"
    echo "relative_depth_image_step_interval=${REL_DEPTH_IMAGE_STEP_INTERVAL}"
    echo "relative_depth_image_batch_size=${REL_DEPTH_IMAGE_BATCH_SIZE}"
    echo "relative_depth_teacher_checkpoint=${REL_DEPTH_TEACHER_CHECKPOINT}"
  } > "$result_file"

  log "[train done] ${SCENE}: ${run_dir:-unknown}"
  log "Result marker: ${result_file}"
}

main "$@"
