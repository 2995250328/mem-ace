#!/usr/bin/env bash
set -euo pipefail

# Indoor6-only DINOv2 + MapAnything-memory + STGS short validation.
# Default matrix is scene2a/scene5 because both have a comparable fast4_b5_f10
# DINOv2+MapAnything-memory baseline.

ROOT_DIR="${ROOT_DIR:-/home/xwh/project/ace_depth}"
REPO_ROOT="${REPO_ROOT:-${ROOT_DIR}/ace_dinov2_lmc}"
DATA_ROOT="${ACE_DATA_ROOT:-/home/xwh/data}"
ACE_ROOT="${ACE_ROOT:-${DATA_ROOT}/indoor6_ace}"
WAI_ROOT="${WAI_ROOT:-${DATA_ROOT}/mapanything-dataset/wai_data/indoor6}"
COLMAP_ROOT="${COLMAP_ROOT:-/data/xwh/indoor6/indoor6-colmap}"

SCENES_STR="${SCENES_STR:-scene2a scene5}"
GPUS_STR="${GPUS_STR:-0 1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
VARIANT_LABEL="${VARIANT_LABEL:-fast4_b5_f10_stgs}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-/data/xwh/ace_dinov2_lmc/04_evaluation/indoor6/dino_stgs_short/${STAMP}_dino_stgs_fast4_b5_f10_gpu01}"
LOG_ROOT="${LOG_ROOT:-${RUN_ROOT}/logs/${VARIANT_LABEL}}"
RESULT_ROOT="${RESULT_ROOT:-${RUN_ROOT}/results/${VARIANT_LABEL}}"
SIDECAR_ROOT="${SIDECAR_ROOT:-${RUN_ROOT}/sidecars}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-${RUN_ROOT}/train_manifest_${VARIANT_LABEL}.tsv}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-10240}"
TRAINING_BUFFER_SIZE="${TRAINING_BUFFER_SIZE:-5000000}"
BUFFER_SIZE_FINAL="${BUFFER_SIZE_FINAL:-10000000}"
BUFFER_BATCH_SIZE="${BUFFER_BATCH_SIZE:-1}"
BUFFER_ON_CPU="${BUFFER_ON_CPU:-True}"
BUFFER_ON_CPU_FINAL="${BUFFER_ON_CPU_FINAL:-True}"
SAMPLES_PER_IMAGE="${SAMPLES_PER_IMAGE:-384}"
POST_TRAIN_SEEDS_STR="${POST_TRAIN_SEEDS_STR:-1305 2026 4242 7777 9001}"
POST_TRAIN_HYPOTHESES="${POST_TRAIN_HYPOTHESES:-256}"
NUM_LATENT_TOKENS="${NUM_LATENT_TOKENS:-64}"
LMC_ITERATIONS="${LMC_ITERATIONS:-4}"
LMC_TRAIN_STEPS="${LMC_TRAIN_STEPS:-600}"
LMC_WARMUP_STEPS="${LMC_WARMUP_STEPS:-2000}"
S1_LOSS_STEP_MODE="${S1_LOSS_STEP_MODE:-per_iter}"
EVAL_EACH_ITERATION="${EVAL_EACH_ITERATION:-True}"
EVAL_AFTER_TRAIN="${EVAL_AFTER_TRAIN:-True}"
BEST_METRIC="${BEST_METRIC:-pct25_5}"
IMAGE_RESOLUTION="${IMAGE_RESOLUTION:-518}"

SIDECAR_IMAGE_WIDTH="${SIDECAR_IMAGE_WIDTH:-none}"
REBUILD_SIDECAR="${REBUILD_SIDECAR:-False}"
PREP_ONLY="${PREP_ONLY:-False}"
DRY_RUN="${DRY_RUN:-False}"

SFM_TRACK_GUIDED_FRACTION="${SFM_TRACK_GUIDED_FRACTION:-0.10}"
USE_STGS_GUIDED_SAMPLING="${USE_STGS_GUIDED_SAMPLING:-True}"
USE_STGS_INTER_FRAME_LOSS="${USE_STGS_INTER_FRAME_LOSS:-False}"
SFM_TRACK_GUIDED_BATCH_SIZE="${SFM_TRACK_GUIDED_BATCH_SIZE:-512}"
SFM_TRACK_GUIDED_MODE="${SFM_TRACK_GUIDED_MODE:-anchor_only}"
SFM_TRACK_INTER_FRAME_APPLY_TO="${SFM_TRACK_INTER_FRAME_APPLY_TO:-stage2}"
SFM_TRACK_INTER_FRAME_WEIGHT="${SFM_TRACK_INTER_FRAME_WEIGHT:-0.0}"
SFM_TRACK_INTER_FRAME_START_RATIO="${SFM_TRACK_INTER_FRAME_START_RATIO:-0.2}"
SFM_TRACK_INTER_FRAME_DECAY_LAST_RATIO="${SFM_TRACK_INTER_FRAME_DECAY_LAST_RATIO:-0.3}"
SFM_TRACK_INTER_FRAME_MAX_PX="${SFM_TRACK_INTER_FRAME_MAX_PX:-100.0}"
SFM_TRACK_INTER_FRAME_DROPOUT="${SFM_TRACK_INTER_FRAME_DROPOUT:-0.5}"
SFM_TRACK_ANCHOR_SELF_WEIGHT="${SFM_TRACK_ANCHOR_SELF_WEIGHT:-0.5}"
SFM_TRACK_MAX_ANCHOR_PATCH_OFFSET_PX="${SFM_TRACK_MAX_ANCHOR_PATCH_OFFSET_PX:-3.5}"
SFM_TRACK_MAX_TARGET_PATCH_OFFSET_PX="${SFM_TRACK_MAX_TARGET_PATCH_OFFSET_PX:-3.5}"
SFM_TRACK_MAX_COLMAP_REPROJ_ERROR_PX="${SFM_TRACK_MAX_COLMAP_REPROJ_ERROR_PX:-0.0}"
SFM_TRACK_MIN_TRACK_LENGTH="${SFM_TRACK_MIN_TRACK_LENGTH:-0}"

mkdir -p "${LOG_ROOT}" "${RESULT_ROOT}" "${SIDECAR_ROOT}"

read -r -a SCENES <<< "${SCENES_STR}"
read -r -a GPUS <<< "${GPUS_STR}"
read -r -a POST_TRAIN_SEEDS <<< "${POST_TRAIN_SEEDS_STR}"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

bool_true() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

memory_path_for_scene() {
  case "$1" in
    scene1)
      echo "${MEMORY_SCENE1:-${REPO_ROOT}/memory_extraction/04_evaluation/memory_extract/scene1/40v_v0.05_bilinear_bse_ut0.02_noaug_asb_adaptive_pool1.0_rgate4_prepair8_sor_gm_l2/20260508_104337/memory_bse.pt}"
      ;;
    scene2a)
      echo "${MEMORY_SCENE2A:-${REPO_ROOT}/memory_extraction/04_evaluation/memory_extract/scene2a/40v_v0.05_bilinear_bse_ut0.02_noaug_asb_adaptive_pool1.0_rgate4_prepair8_sor_gm_l2/20260508_103423/memory_bse.pt}"
      ;;
    scene3)
      echo "${MEMORY_SCENE3:-${REPO_ROOT}/memory_extraction/04_evaluation/memory_extract/scene3/40v_v0.05_bilinear_bse_ut0.02_noaug_asb_adaptive_pool1.0_rgate4_prepair8_sor_gm_l2/20260508_104704/memory_bse.pt}"
      ;;
    scene4a)
      echo "${MEMORY_SCENE4A:-${REPO_ROOT}/memory_extraction/04_evaluation/memory_extract/scene4a/40v_v0.05_bilinear_bse_ut0.02_noaug_asb_adaptive_pool1.0_rgate4_prepair8_sor_gm_l2/20260508_104704/memory_bse.pt}"
      ;;
    scene5)
      echo "${MEMORY_SCENE5:-${REPO_ROOT}/memory_extraction/04_evaluation/memory_extract/scene5/40v_v0.05_bilinear_bse_ut0.02_noaug_asb_adaptive_pool1.0_rgate4_prepair8_sor_gm_l2/20260508_105011/memory_bse.pt}"
      ;;
    scene6)
      echo "${MEMORY_SCENE6:-${REPO_ROOT}/memory_extraction/04_evaluation/memory_extract/scene6/40v_v0.05_bilinear_bse_ut0.02_noaug_asb_adaptive_pool1.0_rgate4_prepair8_sor_gm_l2/20260508_105011/memory_bse.pt}"
      ;;
    *)
      echo "ERROR: not an Indoor6 scene: $1" >&2
      return 1
      ;;
  esac
}

sidecar_width_label() {
  case "${SIDECAR_IMAGE_WIDTH,,}" in
    ""|none|null) echo "var" ;;
    auto)
      if [ "${BUFFER_BATCH_SIZE}" -le 1 ]; then
        echo "var"
      else
        echo $(( (IMAGE_RESOLUTION * 4 / 3 + 13) / 14 * 14 ))
      fi
      ;;
    *) echo "${SIDECAR_IMAGE_WIDTH}" ;;
  esac
}

sidecar_image_width_value() {
  case "${SIDECAR_IMAGE_WIDTH,,}" in
    ""|none|null) echo "none" ;;
    auto)
      if [ "${BUFFER_BATCH_SIZE}" -le 1 ]; then
        echo "none"
      else
        echo $(( (IMAGE_RESOLUTION * 4 / 3 + 13) / 14 * 14 ))
      fi
      ;;
    *) echo "${SIDECAR_IMAGE_WIDTH}" ;;
  esac
}

sidecar_path_for_scene() {
  local scene="$1"
  local width_label
  width_label="$(sidecar_width_label)"
  echo "${SIDECAR_ROOT}/${scene}/train/colmap_keyframe_channel_v1_sp_strict_dino_r${IMAGE_RESOLUTION}_w${width_label}/keyframe_channel.npz"
}

require_scene_inputs() {
  local scene="$1"
  local memory_path="$2"
  local model_dir="$3"
  if [ ! -d "${ACE_ROOT}/${scene}" ]; then
    echo "ERROR: missing ACE scene root: ${ACE_ROOT}/${scene}" >&2
    return 1
  fi
  if [ ! -d "${WAI_ROOT}/${scene}_train" ]; then
    echo "ERROR: missing WAI aux depth root: ${WAI_ROOT}/${scene}_train" >&2
    return 1
  fi
  if [ ! -f "${memory_path}" ]; then
    echo "ERROR: missing MapAnything memory for ${scene}: ${memory_path}" >&2
    return 1
  fi
  if [ ! -f "${model_dir}/points3D.bin" ]; then
    echo "ERROR: missing train-only COLMAP model for ${scene}: ${model_dir}" >&2
    return 1
  fi
}

build_sidecar_if_needed() {
  local scene="$1"
  local scene_root="${ACE_ROOT}/${scene}"
  local model_dir="${COLMAP_ROOT}/${scene}-tr/sparse/0"
  local keyframe_channel
  keyframe_channel="$(sidecar_path_for_scene "$scene")"
  local output_dir
  output_dir="$(dirname "${keyframe_channel}")"
  local width_value
  width_value="$(sidecar_image_width_value)"

  if [ -s "${keyframe_channel}" ] && ! bool_true "${REBUILD_SIDECAR}"; then
    log "[sidecar skip] ${scene}: ${keyframe_channel}"
    return 0
  fi
  log "[sidecar build] scene=${scene} model=${model_dir}"
  if bool_true "${DRY_RUN}"; then
    printf 'DRY_RUN sidecar scene=%s output=%s model=%s width=%s\n' "${scene}" "${output_dir}" "${model_dir}" "${width_value}"
    return 0
  fi
  IMAGE_RESOLUTION="${IMAGE_RESOLUTION}" \
  IMAGE_WIDTH="${width_value}" \
  MIN_TRACK_LENGTH="${MIN_TRACK_LENGTH:-4}" \
  MAX_REPROJ_ERROR="${MAX_REPROJ_ERROR:-2.0}" \
  MIN_PARALLAX_DEG="${MIN_PARALLAX_DEG:-2.0}" \
  MAX_PARALLAX_DEG="${MAX_PARALLAX_DEG:-60.0}" \
  MAX_ANCHOR_ALIGNMENT_PX="${SFM_TRACK_MAX_ANCHOR_PATCH_OFFSET_PX}" \
  MAX_TARGET_ALIGNMENT_PX="${SFM_TRACK_MAX_TARGET_PATCH_OFFSET_PX}" \
  SIDECAR_ROOT="${SIDECAR_ROOT}" \
  bash "${REPO_ROOT}/scripts/build_dino_stgs_keyframe_channel.sh" "${scene_root}" "${model_dir}" "${output_dir}"
}

latest_run_dir_for_scene() {
  local scene="$1"
  local train_root="${RUN_ROOT}/${VARIANT_LABEL}/indoor6_ace/${scene}/dino_ace_lmc_ace_g"
  find "${train_root}" -type f -name "run_config.json" -printf '%T@ %h\n' 2>/dev/null \
    | sort -nr \
    | head -n 1 \
    | cut -d' ' -f2- \
    || true
}

latest_best_file_for_scene() {
  local scene="$1"
  local train_root="${RUN_ROOT}/${VARIANT_LABEL}/indoor6_ace/${scene}/dino_ace_lmc_ace_g"
  find "${train_root}" -type f -name "best_*.pt" -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr \
    | head -n 1 \
    | cut -d' ' -f2- \
    || true
}

run_scene() {
  local gpu="$1"
  local scene="$2"
  local device="cuda:${gpu}"
  local scene_root="${ACE_ROOT}/${scene}"
  local memory_path
  memory_path="$(memory_path_for_scene "${scene}")"
  local model_dir="${COLMAP_ROOT}/${scene}-tr/sparse/0"
  local keyframe_channel
  keyframe_channel="$(sidecar_path_for_scene "${scene}")"
  local log_file="${LOG_ROOT}/train_${scene}_gpu${gpu}.log"
  local result_file="${RESULT_ROOT}/train_${scene}.txt"
  local output_name="${scene}_${VARIANT_LABEL}.pt"

  require_scene_inputs "${scene}" "${memory_path}" "${model_dir}"
  if bool_true "${USE_STGS_GUIDED_SAMPLING}" || bool_true "${USE_STGS_INTER_FRAME_LOSS}"; then
    build_sidecar_if_needed "${scene}"
  else
    log "[sidecar not needed] ${scene}: STGS disabled for ${VARIANT_LABEL}"
  fi
  if bool_true "${PREP_ONLY}"; then
    return 0
  fi
  if { bool_true "${USE_STGS_GUIDED_SAMPLING}" || bool_true "${USE_STGS_INTER_FRAME_LOSS}"; } && [ ! -s "${keyframe_channel}" ] && ! bool_true "${DRY_RUN}"; then
    echo "ERROR: missing built sidecar: ${keyframe_channel}" >&2
    return 1
  fi

  log "[train start] scene=${scene} gpu=${gpu} log=${log_file}"
  if bool_true "${DRY_RUN}"; then
    printf 'DRY_RUN train scene=%s gpu=%s memory=%s sidecar=%s log=%s\n' "${scene}" "${gpu}" "${memory_path}" "${keyframe_channel}" "${log_file}"
    return 0
  fi

  ACE_DATA_ROOT="${DATA_ROOT}" \
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  "${PYTHON_BIN}" "${REPO_ROOT}/train_ace_dinov2_lmc.py" \
    "${scene_root}" \
    "${output_name}" \
    --model_backend ace_dinov2 \
    --train_preset memory_compare_ace_g_v1 \
    --data_backend ace \
    --wai_repo_root "${DATA_ROOT}/map-anything" \
    --device "${device}" \
    --post_train_eval_device "${device}" \
    --post_train_eval_scene "${scene_root}" \
    --use_lmc True \
    --lmc_flow ace_g \
    --lmc_profile legacy \
    --apply_baseline_contract True \
    --vanilla_iterations 1 \
    --memory_path "${memory_path}" \
    --lmc_memory_preflight True \
    --lmc_memory_preflight_strict True \
    --lmc_memory_preflight_center_tol 5.0 \
    --lmc_strict_scene_check True \
    --lmc_strict_center_check True \
    --lmc_scene_center_max_distance 4.0 \
    --bse_denorm_to_world True \
    --c1_ref_norm_alpha 1.0 \
    --c1_aux_ref_loss_weight 0.0 \
    --c1_aux_depth_root "${WAI_ROOT}/${scene}_train" \
    --c1_aux_depth_kind gt_depth \
    --c1_aux_ref_sample_ratio 0.5 \
    --lmc_head_mean_max_shift 4.0 \
    --lmc_strict_head_mean_alignment True \
    --ace_g_fusion_in_s2 True \
    --ace_g_fusion_lr_ratio 0.01 \
    --ace_g_cross_iter_eval False \
    --pe_normalize_input False \
    --lmc_mode global \
    --lmc_fps_start_policy farthest_from_center \
    --lmc_auto_mode_by_visibility False \
    --lmc_visibility_front_ratio_threshold 0.85 \
    --lmc_visibility_fallback_mode local \
    --lmc_visibility_sample_points 4096 \
    --lmc_key_slice_idx 2 \
    --num_latent_tokens "${NUM_LATENT_TOKENS}" \
    --num_attn_layers 4 \
    --s1_batch_size 16 \
    --s1_use_buffer True \
    --s1_buffer_refill_mode full \
    --s1_buffer_keep_ratio 0.5 \
    --use_scale_token True \
    --lmc_iterations "${LMC_ITERATIONS}" \
    --lmc_train_steps "${LMC_TRAIN_STEPS}" \
    --lmc_warmup_steps "${LMC_WARMUP_STEPS}" \
    --s1_early_stop True \
    --s1_early_stop_min_updates 400 \
    --s1_early_stop_patience 180 \
    --s1_early_stop_rel_improve 0.01 \
    --s1_early_stop_ema_beta 0.9 \
    --s1_loss_mode sample_per_image \
    --s1_loss_step_mode "${S1_LOSS_STEP_MODE}" \
    --s1_full_map_max_points 0 \
    --s1_empty_cache_interval 30 \
    --training_buffer_size "${TRAINING_BUFFER_SIZE}" \
    --buffer_size_final "${BUFFER_SIZE_FINAL}" \
    --buffer_batch_size "${BUFFER_BATCH_SIZE}" \
    --buffer_on_cpu "${BUFFER_ON_CPU}" \
    --buffer_on_cpu_final "${BUFFER_ON_CPU_FINAL}" \
    --samples_per_image "${SAMPLES_PER_IMAGE}" \
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
    --image_resolution "${IMAGE_RESOLUTION}" \
    --batch_size "${TRAIN_BATCH_SIZE}" \
    --eval_each_iteration "${EVAL_EACH_ITERATION}" \
    --keep_best_only True \
    --best_metric "${BEST_METRIC}" \
    --eval_after_train "${EVAL_AFTER_TRAIN}" \
    --eval_session post_train \
    --eval_deterministic False \
    --eval_dsacstar_seed 1305 \
    --eval_dsacstar_seed_per_frame True \
    --eval_num_workers 6 \
    --post_train_eval_seeds "${POST_TRAIN_SEEDS[@]}" \
    --post_train_hypotheses "${POST_TRAIN_HYPOTHESES}" \
    --s2_repro_rewind_first_ratio 0.2 \
    --s2_repro_rewind_later_ratio 0.08 \
    --s2_lr_boost_first 1.2 \
    --s2_lr_boost_later 1.0 \
    --use_sfm_track_guided_sampling "${USE_STGS_GUIDED_SAMPLING}" \
    --use_sfm_track_inter_frame_loss "${USE_STGS_INTER_FRAME_LOSS}" \
    --sfm_track_keyframe_channel_path "${keyframe_channel}" \
    --sfm_track_sidecar_mismatch_policy strict \
    --sfm_track_guided_sampling_strategy balanced_replace \
    --sfm_track_guided_fraction "${SFM_TRACK_GUIDED_FRACTION}" \
    --sfm_track_guided_batch_size "${SFM_TRACK_GUIDED_BATCH_SIZE}" \
    --sfm_track_guided_mode "${SFM_TRACK_GUIDED_MODE}" \
    --sfm_track_guided_main_loss_mode include \
    --sfm_track_guided_source_target_mode patch_center \
    --sfm_track_inter_frame_target_mode patch_center \
    --sfm_track_inter_frame_apply_to "${SFM_TRACK_INTER_FRAME_APPLY_TO}" \
    --sfm_track_guided_aux_normalizer full_batch \
    --sfm_track_anchor_self_weight "${SFM_TRACK_ANCHOR_SELF_WEIGHT}" \
    --sfm_track_anchor_use_alignment_weight True \
    --sfm_track_inter_frame_weight "${SFM_TRACK_INTER_FRAME_WEIGHT}" \
    --sfm_track_inter_frame_start_ratio "${SFM_TRACK_INTER_FRAME_START_RATIO}" \
    --sfm_track_inter_frame_decay_last_ratio "${SFM_TRACK_INTER_FRAME_DECAY_LAST_RATIO}" \
    --sfm_track_inter_frame_max_px "${SFM_TRACK_INTER_FRAME_MAX_PX}" \
    --sfm_track_inter_frame_dropout "${SFM_TRACK_INTER_FRAME_DROPOUT}" \
    --sfm_track_max_anchor_patch_offset_px "${SFM_TRACK_MAX_ANCHOR_PATCH_OFFSET_PX}" \
    --sfm_track_max_target_patch_offset_px "${SFM_TRACK_MAX_TARGET_PATCH_OFFSET_PX}" \
    --sfm_track_min_alignment_weight 0.0 \
    --sfm_track_max_colmap_reproj_error_px "${SFM_TRACK_MAX_COLMAP_REPROJ_ERROR_PX}" \
    --sfm_track_min_track_length "${SFM_TRACK_MIN_TRACK_LENGTH}" \
    --experiment_root "${RUN_ROOT}" \
    --experiment_subdir "${VARIANT_LABEL}" \
    --run_name auto \
    --output_layout hierarchical \
    --overwrite_run_dir True \
    > "${log_file}" 2>&1

  local run_dir best_file
  run_dir="$(latest_run_dir_for_scene "${scene}")"
  best_file="$(latest_best_file_for_scene "${scene}")"
  {
    echo "status=ok"
    echo "scene=${scene}"
    echo "gpu=${gpu}"
    echo "memory_path=${memory_path}"
    echo "keyframe_channel=${keyframe_channel}"
    echo "run_dir=${run_dir:-unknown}"
    echo "best_file=${best_file:-unknown}"
    echo "log=${log_file}"
  } > "${result_file}"
  printf '%s\t%s\t%s\t%s\t%s\n' "${scene}" "${gpu}" "${keyframe_channel}" "${run_dir:-unknown}" "${best_file:-unknown}" >> "${TRAIN_MANIFEST}"
  log "[train done] scene=${scene}: ${run_dir:-unknown}"
}

wait_batch() {
  local -n pids_ref="$1"
  local -n labels_ref="$2"
  local failed=0
  for i in "${!pids_ref[@]}"; do
    if ! wait "${pids_ref[$i]}"; then
      log "[failed] ${labels_ref[$i]}"
      failed=1
    else
      log "[ok] ${labels_ref[$i]}"
    fi
  done
  return "${failed}"
}

main() {
  if [ "${#GPUS[@]}" -eq 0 ]; then
    echo "ERROR: GPUS_STR is empty" >&2
    exit 1
  fi
  if [ "${BUFFER_BATCH_SIZE}" -gt 1 ] && [[ "$(sidecar_image_width_value)" == "none" ]]; then
    echo "ERROR: BUFFER_BATCH_SIZE>1 requires fixed SIDECAR_IMAGE_WIDTH, e.g. SIDECAR_IMAGE_WIDTH=700." >&2
    exit 2
  fi

  log "Run root : ${RUN_ROOT}"
  log "Variant  : ${VARIANT_LABEL}"
  log "Scenes   : ${SCENES[*]}"
  log "GPUs     : ${GPUS[*]}"
  log "Config   : DINOv2+MapAnything-memory+STGS, K${NUM_LATENT_TOKENS}, it${LMC_ITERATIONS}, res${IMAGE_RESOLUTION}, buf=${TRAINING_BUFFER_SIZE}/${BUFFER_SIZE_FINAL}, bs=${TRAIN_BATCH_SIZE}, buffer_batch=${BUFFER_BATCH_SIZE}, sidecar_w=$(sidecar_width_label)"
  log "STGS     : guided=${USE_STGS_GUIDED_SAMPLING}, mode=${SFM_TRACK_GUIDED_MODE}, inter_frame=${USE_STGS_INTER_FRAME_LOSS}, apply=${SFM_TRACK_INTER_FRAME_APPLY_TO}, w=${SFM_TRACK_INTER_FRAME_WEIGHT}, schedule=${SFM_TRACK_INTER_FRAME_START_RATIO}/${SFM_TRACK_INTER_FRAME_DECAY_LAST_RATIO}, drop=${SFM_TRACK_INTER_FRAME_DROPOUT}, balanced_replace f=${SFM_TRACK_GUIDED_FRACTION}, source/target patch_center, offset<=${SFM_TRACK_MAX_ANCHOR_PATCH_OFFSET_PX}/${SFM_TRACK_MAX_TARGET_PATCH_OFFSET_PX}px"
  printf 'scene\tgpu\tkeyframe_channel\trun_dir\tbest_file\n' > "${TRAIN_MANIFEST}"

  local pids=()
  local labels=()
  local gpu_index=0
  for scene in "${SCENES[@]}"; do
    local gpu="${GPUS[$gpu_index]}"
    run_scene "${gpu}" "${scene}" &
    pids+=("$!")
    labels+=("${scene}:gpu${gpu}")
    gpu_index=$(((gpu_index + 1) % ${#GPUS[@]}))
    if [ "${#pids[@]}" -ge "${#GPUS[@]}" ]; then
      wait_batch pids labels
      pids=()
      labels=()
    fi
  done
  if [ "${#pids[@]}" -gt 0 ]; then
    wait_batch pids labels
  fi
  log "Done. Manifest: ${TRAIN_MANIFEST}"
  if ! bool_true "${DRY_RUN}" && ! bool_true "${PREP_ONLY}"; then
    log "Aggregating metric-wise best results under ${RUN_ROOT}"
    if bash "${REPO_ROOT}/scripts/aggregate_lmc_metricwise_best.sh" "${RUN_ROOT}" > "${RUN_ROOT}/metricwise_best.aggregate.log" 2>&1; then
      log "Metric-wise best: ${RUN_ROOT}/metricwise_best.tsv"
    else
      log "[warn] metric-wise aggregation failed; see ${RUN_ROOT}/metricwise_best.aggregate.log"
    fi
  fi
}

main "$@"
