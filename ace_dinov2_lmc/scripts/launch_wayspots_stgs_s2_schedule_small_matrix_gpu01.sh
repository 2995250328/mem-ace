#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

CONDA_ENV="${CONDA_ENV:-mapanything}"
WAYSPOTS_ROOT="${WAYSPOTS_ROOT:-/data/xwh/Wayspots}"
MEMORY_ROOT="${MEMORY_ROOT:-/data/xwh/ace_dinov2_lmc/04_evaluation/wayspots_ace_fcn_lmc_suite/20260530_181745/memory}"
SIDECAR_ROOT="${SIDECAR_ROOT:-/data/xwh/dataset_staging/wayspots/extracted}"
ACE_ENCODER_PATH="${ACE_ENCODER_PATH:-${ROOT_DIR}/ace_encoder_pretrained.pt}"

SCENES="${SCENES:-wayspots_bears wayspots_cubes}"
CONFIGS="${CONFIGS:-}"
LMC_ITERATIONS="${LMC_ITERATIONS:-4}"
TRAINING_BUFFER_SIZE="${TRAINING_BUFFER_SIZE:-5000000}"
BUFFER_SIZE_FINAL="${BUFFER_SIZE_FINAL:-5000000}"
NUM_LATENT_TOKENS="${NUM_LATENT_TOKENS:-64}"
IMAGE_RESOLUTION="${IMAGE_RESOLUTION:-512}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
EPOCHS="${EPOCHS:-24}"
BEST_METRIC="${BEST_METRIC:-pct25_5}"
POST_TRAIN_EVAL_SEEDS="${POST_TRAIN_EVAL_SEEDS:-1305 2026 4242}"
POST_TRAIN_HYPOTHESES="${POST_TRAIN_HYPOTHESES:-256}"
ITERATION_EVAL_HYPOTHESES="${ITERATION_EVAL_HYPOTHESES:-256}"

GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-/data/xwh/ace_dinov2_lmc/04_evaluation/wayspots/stgs_s2_schedule/$(date +%Y%m%d)_stage1_it${LMC_ITERATIONS}_buf${TRAINING_BUFFER_SIZE}_anchor_v2_s2schedule_gpu01}"
LOG_DIR="${RUN_ROOT}/launcher_logs"
DRY_RUN="${DRY_RUN:-false}"

mkdir -p "${LOG_DIR}"
SUPERVISOR_LOG="${LOG_DIR}/supervisor_${STAMP}.log"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${SUPERVISOR_LOG}"
}

bool_true() {
  case "${1:-}" in
    1|true|TRUE|True|yes|YES|y|Y) return 0 ;;
    *) return 1 ;;
  esac
}

sidecar_path() {
  local scene="$1"
  echo "${SIDECAR_ROOT}/${scene}/train/colmap_keyframe_channel_v1_sp_strict/keyframe_channel.npz"
}

sidecar_summary_path() {
  local scene="$1"
  echo "${SIDECAR_ROOT}/${scene}/train/colmap_keyframe_channel_v1_sp_strict/summary.json"
}

memory_path() {
  local scene="$1"
  echo "${MEMORY_ROOT}/${scene}/memory_ace_fcn_sparse_sp_r4.pt"
}

check_scene_inputs() {
  local scene="$1"
  local scene_root="${WAYSPOTS_ROOT}/${scene}"
  local memory
  local sidecar
  local summary
  memory="$(memory_path "${scene}")"
  sidecar="$(sidecar_path "${scene}")"
  summary="$(sidecar_summary_path "${scene}")"

  [[ -d "${scene_root}" ]] || { log "ERROR missing scene root: ${scene_root}"; exit 1; }
  [[ -f "${memory}" ]] || { log "ERROR missing memory: ${memory}"; exit 1; }
  [[ -f "${sidecar}" ]] || { log "ERROR missing STGS keyframe_channel: ${sidecar}"; exit 1; }
  [[ -f "${summary}" ]] || { log "ERROR missing STGS summary: ${summary}"; exit 1; }

  local rows
  rows="$(python3 - "${summary}" <<'PYCHECK'
import json, sys
path = sys.argv[1]
with open(path, 'r', encoding='utf-8') as f:
    data = json.load(f)
for key in ('rows', 'num_rows', 'n_rows'):
    if key in data:
        print(int(data[key]))
        break
else:
    print(0)
PYCHECK
)"
  if [[ "${rows}" -le 0 ]]; then
    log "ERROR STGS sidecar rows <= 0 for ${scene}: ${summary} rows=${rows}"
    exit 1
  fi
  log "inputs ok scene=${scene} memory=${memory} sidecar_rows=${rows} sidecar=${sidecar}"
}

write_plan() {
  cat >"${RUN_ROOT}/matrix_plan_${STAMP}.txt" <<EOFPLAN
purpose: Wayspots ACE-FCN-LMC Stage1-only STGS anchor-v2 small matrix for internal S1/S2-G schedule validation
run_root: ${RUN_ROOT}
timestamp: ${STAMP}
scenes: ${SCENES}
configs:
  alt_r2:    ace_g_s2_schedule=every_iter, ace_g_fusion_in_s2=True,  ace_g_cross_iter_eval=True
  s1_only:   ace_g_s2_schedule=none,       ace_g_fusion_in_s2=False, ace_g_cross_iter_eval=False
  final_r1:  ace_g_s2_schedule=final_only, ace_g_fusion_in_s2=False, ace_g_cross_iter_eval=False
  final_r2:  ace_g_s2_schedule=final_only, ace_g_fusion_in_s2=True,  ace_g_cross_iter_eval=False
stgs_anchor_v2:
  use_sfm_track_guided_sampling: True
  sfm_track_guided_batch_size: 512
  sfm_track_guided_sampling_strategy: balanced_replace
  sfm_track_guided_fraction: 0.10
  sfm_track_guided_mode: anchor_only
  sfm_track_guided_main_loss_mode: include
  sfm_track_guided_source_target_mode: patch_center
  sfm_track_guided_aux_normalizer: full_batch
  sfm_track_anchor_self_weight: 0.5
  sfm_track_anchor_use_alignment_weight: True
  sfm_track_inter_frame_weight: 0.0
training:
  lmc_iterations: ${LMC_ITERATIONS}
  training_buffer_size: ${TRAINING_BUFFER_SIZE}
  buffer_size_final: ${BUFFER_SIZE_FINAL}
  num_latent_tokens: ${NUM_LATENT_TOKENS}
  image_resolution: ${IMAGE_RESOLUTION}
  batch_size: ${BATCH_SIZE}
  epochs: ${EPOCHS}
  best_metric: ${BEST_METRIC}
eval:
  deterministic: True
  iteration_eval_seed: 1305
  iteration_eval_hypotheses: ${ITERATION_EVAL_HYPOTHESES}
  post_train_eval_seeds: ${POST_TRAIN_EVAL_SEEDS}
  post_train_hypotheses: ${POST_TRAIN_HYPOTHESES}
gpu_policy:
  default_pair_a: alt_r2 on physical GPU ${GPU0}, s1_only on physical GPU ${GPU1}
  default_pair_b: final_r1 on physical GPU ${GPU0}, final_r2 on physical GPU ${GPU1}
  custom_configs: ${CONFIGS:-<default full schedule matrix>}
EOFPLAN
}

common_args() {
  local scene="$1"
  local scene_root="${WAYSPOTS_ROOT}/${scene}"
  local memory
  memory="$(memory_path "${scene}")"

  printf '%s\0' \
    --model_backend ace_fcn_lmc \
    --data_backend ace \
    --use_lmc True \
    --lmc_flow ace_g \
    --memory_path "${memory}" \
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
    --iteration_eval_hypotheses "${ITERATION_EVAL_HYPOTHESES}" \
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
    --best_metric "${BEST_METRIC}" \
    --post_train_hypotheses "${POST_TRAIN_HYPOTHESES}"
}

stgs_args() {
  local scene="$1"
  local keyframe_channel
  keyframe_channel="$(sidecar_path "${scene}")"

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

schedule_args() {
  local config="$1"
  case "${config}" in
    alt_r2)
      printf '%s\0' --ace_g_s2_schedule every_iter --ace_g_fusion_in_s2 True --ace_g_cross_iter_eval True
      ;;
    s1_only)
      printf '%s\0' --ace_g_s2_schedule none --ace_g_fusion_in_s2 False --ace_g_cross_iter_eval False
      ;;
    final_r1)
      printf '%s\0' --ace_g_s2_schedule final_only --ace_g_fusion_in_s2 False --ace_g_cross_iter_eval False
      ;;
    final_r2)
      printf '%s\0' --ace_g_s2_schedule final_only --ace_g_fusion_in_s2 True --ace_g_cross_iter_eval False
      ;;
    *)
      log "ERROR unknown config: ${config}"
      exit 1
      ;;
  esac
}

collect_null_args() {
  local fn="$1"
  shift
  local -n out_ref="$1"
  shift
  local item
  while IFS= read -r -d '' item; do
    out_ref+=("${item}")
  done < <("${fn}" "$@")
}

run_train() {
  local scene="$1"
  local config="$2"
  local gpu="$3"
  local scene_root="${WAYSPOTS_ROOT}/${scene}"
  local tag="${scene}_stgs_${config}_it${LMC_ITERATIONS}_buf${TRAINING_BUFFER_SIZE}_${STAMP}"
  local exp_root="${RUN_ROOT}/${scene}/${config}"
  local exp_subdir="stage1_stgs_${config}_it${LMC_ITERATIONS}_buf${TRAINING_BUFFER_SIZE}"
  local log_file="${LOG_DIR}/${tag}_gpu${gpu}.log"
  local -a cmd=(
    env -u CUDA_VISIBLE_DEVICES
    conda run --no-capture-output -n "${CONDA_ENV}" python ace_dinov2_lmc/train_ace_dinov2_lmc.py
    "${scene_root}" "${tag}.pt"
    --run_name "${tag}"
    --device "cuda:${gpu}"
    --post_train_eval_device "cuda:${gpu}"
    --experiment_root "${exp_root}"
    --experiment_subdir "${exp_subdir}"
  )

  collect_null_args common_args cmd "${scene}"
  collect_null_args stgs_args cmd "${scene}"
  collect_null_args schedule_args cmd "${config}"
  cmd+=(--post_train_eval_seeds)
  local seed
  for seed in ${POST_TRAIN_EVAL_SEEDS}; do
    cmd+=("${seed}")
  done

  {
    echo "[$(date)] scene=${scene} config=${config} physical_gpu=${gpu} run_root=${RUN_ROOT}"
    printf '%q ' "${cmd[@]}"
    printf '\n'
  } >>"${log_file}"

  if bool_true "${DRY_RUN}"; then
    log "DRY_RUN ${scene} ${config} gpu${gpu}; command logged to ${log_file}"
    return 0
  fi

  log "start ${scene} ${config} on gpu${gpu}; log=${log_file}"
  OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" OMP_DYNAMIC="${OMP_DYNAMIC:-FALSE}" "${cmd[@]}" >>"${log_file}" 2>&1
  log "done ${scene} ${config} on gpu${gpu}"
}

launch_pair() {
  local scene="$1"
  local config_a="$2"
  local gpu_a="$3"
  local config_b="$4"
  local gpu_b="$5"
  log "launch pair scene=${scene}: ${config_a} gpu${gpu_a}, ${config_b} gpu${gpu_b}"
  run_train "${scene}" "${config_a}" "${gpu_a}" &
  local pid_a=$!
  run_train "${scene}" "${config_b}" "${gpu_b}" &
  local pid_b=$!

  local rc_a=0
  local rc_b=0
  wait "${pid_a}" || rc_a=$?
  wait "${pid_b}" || rc_b=$?

  if [[ "${rc_a}" -ne 0 || "${rc_b}" -ne 0 ]]; then
    log "pair failed scene=${scene}: ${config_a}_rc=${rc_a}, ${config_b}_rc=${rc_b}"
    exit 1
  fi
  log "pair complete scene=${scene}: ${config_a}, ${config_b}"
}

write_plan
log "matrix plan: ${RUN_ROOT}/matrix_plan_${STAMP}.txt"
log "dry_run=${DRY_RUN} scenes=${SCENES} configs=${CONFIGS:-<default>} it=${LMC_ITERATIONS} buffer=${TRAINING_BUFFER_SIZE} final=${BUFFER_SIZE_FINAL} best_metric=${BEST_METRIC}"
log "GPUs: ${GPU0}, ${GPU1} (physical ids; CUDA_VISIBLE_DEVICES is unset for train commands)"

if ! bool_true "${DRY_RUN}" && [[ -x ace_dinov2_lmc/scripts/agent_safe_snapshot.sh ]]; then
  bash ace_dinov2_lmc/scripts/agent_safe_snapshot.sh run_provenance "wayspots_stgs_s2_schedule_${STAMP}" "${RUN_ROOT}" \
    >>"${SUPERVISOR_LOG}" 2>&1 || log "WARN run_provenance snapshot failed; continuing"
fi

for scene in ${SCENES}; do
  check_scene_inputs "${scene}"
  if [[ -n "${CONFIGS}" ]]; then
    gpu_index=0
    for config in ${CONFIGS}; do
      if [[ "${gpu_index}" -eq 0 ]]; then
        run_train "${scene}" "${config}" "${GPU0}"
        gpu_index=1
      else
        run_train "${scene}" "${config}" "${GPU1}"
        gpu_index=0
      fi
    done
  else
    launch_pair "${scene}" alt_r2 "${GPU0}" s1_only "${GPU1}"
    launch_pair "${scene}" final_r1 "${GPU0}" final_r2 "${GPU1}"
  fi
done

if ! bool_true "${DRY_RUN}"; then
  log "all training jobs complete; aggregating metric-wise best"
  bash ace_dinov2_lmc/scripts/aggregate_lmc_metricwise_best.sh "${RUN_ROOT}" | tee -a "${SUPERVISOR_LOG}"
fi

log "matrix complete: ${RUN_ROOT}"
