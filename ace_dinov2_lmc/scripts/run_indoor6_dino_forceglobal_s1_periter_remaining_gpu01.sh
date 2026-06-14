#!/usr/bin/env bash
# Complete the Indoor6 DINOv2 ACE-G forceglobal+s1-per-iter configuration.
#
# This script runs the five non-scene2a Indoor6 scenes with the same core
# configuration as the current best scene2a run:
#   K${NUM_LATENT_TOKENS:-64}, it28, res518, buf2.56M/final7.68M, bs10240, spi384,
#   ACE-G fusion in S2, forceglobal, and S1 loss step mode per_iter.
#
# Run from the parent project root:
#   cd /home/xwh/project/ace_depth
#   conda activate mapanything
#   bash ace_dinov2_lmc/scripts/run_indoor6_dino_forceglobal_s1_periter_remaining_gpu01.sh

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/xwh/project/ace_depth}"
REPO_ROOT="${REPO_ROOT:-${ROOT_DIR}/ace_dinov2_lmc}"
DATA_ROOT="${ACE_DATA_ROOT:-/home/xwh/data}"
ACE_ROOT="${ACE_ROOT:-${DATA_ROOT}/indoor6_ace}"
WAI_ROOT="${WAI_ROOT:-${DATA_ROOT}/mapanything-dataset/wai_data/indoor6}"

SCENES_STR="${SCENES_STR:-scene1 scene3 scene4a scene5 scene6}"
GPUS_STR="${GPUS_STR:-0 1}"
PYTHON_BIN="${PYTHON_BIN:-python}"

EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${REPO_ROOT}/04_evaluation/train_compare}"
EXPERIMENT_SUBDIR="${EXPERIMENT_SUBDIR:-indoor6_full_baselines_4090_forceglobal_s1_periter}"
RUN_ROOT="${RUN_ROOT:-${EXPERIMENT_ROOT}/run_logs/forceglobal_s1_periter_remaining_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${RUN_ROOT}/logs"
RESULT_ROOT="${RUN_ROOT}/results"
TRAIN_MANIFEST="${RUN_ROOT}/train_manifest.tsv"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-10240}"
TRAINING_BUFFER_SIZE="${TRAINING_BUFFER_SIZE:-2560000}"
BUFFER_SIZE_FINAL="${BUFFER_SIZE_FINAL:-7680000}"
BUFFER_ON_CPU="${BUFFER_ON_CPU:-false}"
BUFFER_ON_CPU_FINAL="${BUFFER_ON_CPU_FINAL:-true}"
POST_TRAIN_SEEDS_STR="${POST_TRAIN_SEEDS_STR:-1305 2026 4242 7777 9001}"
POST_TRAIN_HYPOTHESES="${POST_TRAIN_HYPOTHESES:-256}"
NUM_LATENT_TOKENS="${NUM_LATENT_TOKENS:-64}"
LMC_MODE="${LMC_MODE:-global}"
LMC_AUTO_MODE_BY_VISIBILITY="${LMC_AUTO_MODE_BY_VISIBILITY:-False}"
S1_LOSS_STEP_MODE="${S1_LOSS_STEP_MODE:-per_iter}"
EVAL_EACH_ITERATION="${EVAL_EACH_ITERATION:-true}"
EVAL_AFTER_TRAIN="${EVAL_AFTER_TRAIN:-true}"
RESUME_EXISTING="${RESUME_EXISTING:-true}"
DRY_RUN="${DRY_RUN:-false}"
LIVE_LOGS="${LIVE_LOGS:-true}"
LIVE_TAIL_LINES="${LIVE_TAIL_LINES:-20}"
HEARTBEAT_INTERVAL="${HEARTBEAT_INTERVAL:-300}"

mkdir -p "$LOG_ROOT" "$RESULT_ROOT"

read -r -a SCENES <<< "$SCENES_STR"
read -r -a GPUS <<< "$GPUS_STR"
read -r -a POST_TRAIN_SEEDS <<< "$POST_TRAIN_SEEDS_STR"

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
      echo "${MEMORY_SCENE2A:-${REPO_ROOT}/memory_extraction/04_evaluation/memory_extract/scene2a/40v_v0.05_bilinear_bse_ut0.02_noaug_asb_adaptive_pool1.0_rgate4_prepair8_sor_gm_l2/20260508_104337/memory_bse.pt}"
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
      echo "ERROR: no memory mapping for scene '$1'" >&2
      return 1
      ;;
  esac
}

require_scene_inputs() {
  local scene="$1"
  local memory_path="$2"
  if [ ! -d "${ACE_ROOT}/${scene}" ]; then
    echo "ERROR: missing ACE scene root: ${ACE_ROOT}/${scene}" >&2
    return 1
  fi
  if [ ! -d "${WAI_ROOT}/${scene}_train" ]; then
    echo "ERROR: missing WAI aux depth root: ${WAI_ROOT}/${scene}_train" >&2
    return 1
  fi
  if [ ! -f "$memory_path" ]; then
    echo "ERROR: missing memory for ${scene}: ${memory_path}" >&2
    return 1
  fi
}

result_is_ok() {
  local result_file="$1"
  [ -f "$result_file" ] || return 1
  local status best_file
  status="$(awk -F '=' '$1 == "status" {print $2}' "$result_file" | tail -n 1)"
  best_file="$(awk -F '=' '$1 == "best_file" {print $2}' "$result_file" | tail -n 1)"
  [ "$status" = "ok" ] && [ -n "$best_file" ] && [ -f "$best_file" ]
}

start_live_log_tail() {
  local label="$1"
  local log_file="$2"
  STARTED_MONITOR_PID=""
  if ! bool_true "$LIVE_LOGS"; then
    return 0
  fi
  touch "$log_file"
  tail -n "$LIVE_TAIL_LINES" -F "$log_file" 2>/dev/null \
    | awk -v label="$label" '{ printf("[%s] %s\n", label, $0); fflush(); }' &
  STARTED_MONITOR_PID="$!"
}

start_heartbeat() {
  local label="$1"
  local job_pid="$2"
  local log_file="$3"
  STARTED_MONITOR_PID=""
  if [ "$HEARTBEAT_INTERVAL" -le 0 ]; then
    return 0
  fi
  (
    local start_ts
    start_ts="$(date +%s)"
    while kill -0 "$job_pid" 2>/dev/null; do
      sleep "$HEARTBEAT_INTERVAL" || exit 0
      if kill -0 "$job_pid" 2>/dev/null; then
        local now elapsed
        now="$(date +%s)"
        elapsed=$((now - start_ts))
        log "[running] ${label} elapsed=${elapsed}s log=${log_file}"
      fi
    done
  ) &
  STARTED_MONITOR_PID="$!"
}

stop_monitor_pid() {
  local pid="${1:-}"
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  fi
}

latest_run_dir_for_scene() {
  local scene="$1"
  local train_root="${EXPERIMENT_ROOT}/${EXPERIMENT_SUBDIR}/indoor6_ace/${scene}/dino_ace_lmc_ace_g"
  find "$train_root" -type f -name "run_config.json" -printf '%T@ %h\n' 2>/dev/null \
    | sort -nr \
    | head -n 1 \
    | cut -d' ' -f2- \
    || true
}

latest_best_file_for_scene() {
  local scene="$1"
  local train_root="${EXPERIMENT_ROOT}/${EXPERIMENT_SUBDIR}/indoor6_ace/${scene}/dino_ace_lmc_ace_g"
  find "$train_root" -type f -name "best_*.pt" -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr \
    | head -n 1 \
    | cut -d' ' -f2- \
    || true
}

run_scene() {
  local gpu="$1"
  local scene="$2"
  local device="cuda:${gpu}"
  local memory_path
  memory_path="$(memory_path_for_scene "$scene")"
  local log_file="${LOG_ROOT}/train_${scene}_gpu${gpu}.log"
  local result_file="${RESULT_ROOT}/train_${scene}.txt"
  local output_name="${scene}_forceglobal_periter_K${NUM_LATENT_TOKENS}_$(date +%Y%m%d_%H%M%S).pt"

  require_scene_inputs "$scene" "$memory_path"
  if bool_true "$RESUME_EXISTING" && result_is_ok "$result_file"; then
    log "[skip] ${scene}: existing successful marker: ${result_file}"
    return 0
  fi

  log "[train start] scene=${scene} gpu=${gpu}"
  log "[train log] ${log_file}"

  if bool_true "$DRY_RUN"; then
    printf 'DRY_RUN scene=%s gpu=%s memory=%s log=%s\n' "$scene" "$gpu" "$memory_path" "$log_file"
    return 0
  fi

  ACE_DATA_ROOT="$DATA_ROOT" \
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  "$PYTHON_BIN" "${REPO_ROOT}/train_ace_dinov2_lmc.py" \
    "${ACE_ROOT}/${scene}" \
    "$output_name" \
    --train_preset memory_compare_ace_g_v1 \
    --data_backend ace \
    --wai_repo_root "${DATA_ROOT}/map-anything" \
    --device "$device" \
    --post_train_eval_device "$device" \
    --use_lmc True \
    --lmc_flow ace_g \
    --lmc_profile legacy \
    --apply_baseline_contract True \
    --vanilla_iterations 1 \
    --memory_path "$memory_path" \
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
    --ace_g_cross_iter_eval True \
    --pe_normalize_input False \
    --lmc_mode "$LMC_MODE" \
    --lmc_fps_start_policy farthest_from_center \
    --lmc_auto_mode_by_visibility "$LMC_AUTO_MODE_BY_VISIBILITY" \
    --lmc_visibility_front_ratio_threshold 0.85 \
    --lmc_visibility_fallback_mode local \
    --lmc_visibility_sample_points 4096 \
    --lmc_key_slice_idx 2 \
    --num_latent_tokens "$NUM_LATENT_TOKENS" \
    --num_attn_layers 4 \
    --s1_batch_size 16 \
    --s1_use_buffer True \
    --s1_buffer_refill_mode full \
    --s1_buffer_keep_ratio 0.5 \
    --use_scale_token True \
    --lmc_iterations 28 \
    --lmc_train_steps 600 \
    --lmc_warmup_steps 2000 \
    --s1_early_stop True \
    --s1_early_stop_min_updates 400 \
    --s1_early_stop_patience 180 \
    --s1_early_stop_rel_improve 0.01 \
    --s1_early_stop_ema_beta 0.9 \
    --s1_loss_mode sample_per_image \
    --s1_loss_step_mode "$S1_LOSS_STEP_MODE" \
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

  local run_dir best_file
  run_dir="$(latest_run_dir_for_scene "$scene")"
  best_file="$(latest_best_file_for_scene "$scene")"
  {
    echo "status=ok"
    echo "scene=${scene}"
    echo "gpu=${gpu}"
    echo "memory_path=${memory_path}"
    echo "run_dir=${run_dir:-unknown}"
    echo "best_file=${best_file:-unknown}"
    echo "log=${log_file}"
  } > "$result_file"
  printf '%s\t%s\t%s\t%s\n' "$scene" "$gpu" "${run_dir:-unknown}" "${best_file:-unknown}" >> "$TRAIN_MANIFEST"
  log "[train done] scene=${scene}: ${run_dir:-unknown}"
}

wait_batch() {
  local -n pids_ref="$1"
  local -n labels_ref="$2"
  local -n tail_ref="$3"
  local -n heartbeat_ref="$4"
  local failed=0

  for i in "${!pids_ref[@]}"; do
    if ! wait "${pids_ref[$i]}"; then
      log "[failed] ${labels_ref[$i]}"
      failed=1
    else
      log "[ok] ${labels_ref[$i]}"
    fi
    stop_monitor_pid "${tail_ref[$i]:-}"
    stop_monitor_pid "${heartbeat_ref[$i]:-}"
  done
  return "$failed"
}

main() {
  if [ "${#GPUS[@]}" -eq 0 ]; then
    echo "ERROR: GPUS_STR is empty" >&2
    exit 1
  fi

  log "Run root        : ${RUN_ROOT}"
  log "Experiment subdir: ${EXPERIMENT_SUBDIR}"
  log "Scenes          : ${SCENES[*]}"
  log "GPUs            : ${GPUS[*]}"
  log "Config          : K${NUM_LATENT_TOKENS} it28 res518 buf=${TRAINING_BUFFER_SIZE}/${BUFFER_SIZE_FINAL} bs=${TRAIN_BATCH_SIZE} spi=384 mode=${LMC_MODE} auto_visibility=${LMC_AUTO_MODE_BY_VISIBILITY} s1_step=${S1_LOSS_STEP_MODE}"
  log "Dry run         : ${DRY_RUN}"
  printf 'scene\tgpu\trun_dir\tbest_file\n' > "$TRAIN_MANIFEST"

  local pids=()
  local labels=()
  local tail_pids=()
  local heartbeat_pids=()
  local gpu_index=0

  for scene in "${SCENES[@]}"; do
    local gpu="${GPUS[$gpu_index]}"
    local label="${scene}:gpu${gpu}"
    local log_file="${LOG_ROOT}/train_${scene}_gpu${gpu}.log"

    run_scene "$gpu" "$scene" &
    local job_pid="$!"
    pids+=("$job_pid")
    labels+=("$label")
    start_live_log_tail "$label" "$log_file"
    tail_pids+=("$STARTED_MONITOR_PID")
    start_heartbeat "$label" "$job_pid" "$log_file"
    heartbeat_pids+=("$STARTED_MONITOR_PID")

    gpu_index=$(((gpu_index + 1) % ${#GPUS[@]}))
    if [ "${#pids[@]}" -ge "${#GPUS[@]}" ]; then
      wait_batch pids labels tail_pids heartbeat_pids
      pids=()
      labels=()
      tail_pids=()
      heartbeat_pids=()
    fi
  done

  if [ "${#pids[@]}" -gt 0 ]; then
    wait_batch pids labels tail_pids heartbeat_pids
  fi

  log "Done. Manifest: ${TRAIN_MANIFEST}"
}

main "$@"
