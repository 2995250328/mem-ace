#!/usr/bin/env bash
set -euo pipefail

# Indoor6 + Wayspots Stage2 R2 big-buffer probe launcher.
#
# Purpose:
#   Prepare and launch a small first-round probe, not a full scene sweep.
#   GPU0: Indoor6 scene1 -> scene5
#   GPU1: Wayspots bears -> squarebench
#
# Default action is preflight only; it does not start training.
#
# From /home/xwh/project/ace_depth:
#   ACTION=preflight bash ace_dinov2_lmc/scripts/launch_indoor6_wayspots_stage2_r2_bigbuf_probe_gpu01.sh
#   ACTION=launch    bash ace_dinov2_lmc/scripts/launch_indoor6_wayspots_stage2_r2_bigbuf_probe_gpu01.sh
#
# After ACTION=launch, verify:
#   tmux capture-pane -pt indoor6_wayspots_stage2_r2_bigbuf_20260626_gpu01:indoor6 -S -80
#   tmux capture-pane -pt indoor6_wayspots_stage2_r2_bigbuf_20260626_gpu01:wayspots -S -80
#   nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

ACTION="${ACTION:-preflight}"  # preflight | launch | worker
DATASET="${DATASET:-}"         # indoor6 | wayspots, only for ACTION=worker
CONDA_ENV="${CONDA_ENV:-mapanything}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATE_TAG="${DATE_TAG:-20260626}"
SESSION="${SESSION:-indoor6_wayspots_stage2_r2_bigbuf_${DATE_TAG}_gpu01}"
DRY_RUN="${DRY_RUN:-false}"
SKIP_EXISTING="${SKIP_EXISTING:-true}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-true}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OMP_DYNAMIC="${OMP_DYNAMIC:-FALSE}"

# Shared Stage2 R2-big-buffer protocol.
NUM_LATENT_TOKENS="${NUM_LATENT_TOKENS:-64}"
LMC_ITERATIONS="${LMC_ITERATIONS:-10}"
IMAGE_RESOLUTION="${IMAGE_RESOLUTION:-512}"
TRAINING_BUFFER_SIZE="${TRAINING_BUFFER_SIZE:-10000000}"
BUFFER_SIZE_FINAL="${BUFFER_SIZE_FINAL:-12000000}"
BATCH_SIZE="${BATCH_SIZE:-8192}"
SAMPLES_PER_IMAGE="${SAMPLES_PER_IMAGE:-512}"
LMC_TRAIN_STEPS="${LMC_TRAIN_STEPS:-600}"
LMC_WARMUP_STEPS="${LMC_WARMUP_STEPS:-2000}"
ACE_G_FUSION_LR_RATIO="${ACE_G_FUSION_LR_RATIO:-0.01}"
POST_TRAIN_HYPOTHESES="${POST_TRAIN_HYPOTHESES:-256}"
POST_TRAIN_EVAL_SEEDS=(${POST_TRAIN_EVAL_SEEDS:-1305 2026 4242})
ITERATION_EVAL_HYPOTHESES="${ITERATION_EVAL_HYPOTHESES:-256}"
ITERATION_EVAL_SEED="${ITERATION_EVAL_SEED:-1305}"
EVAL_DETERMINISTIC="${EVAL_DETERMINISTIC:-True}"
EVAL_DSACSTAR_SEED="${EVAL_DSACSTAR_SEED:-1305}"
EVAL_DSACSTAR_SEED_PER_FRAME="${EVAL_DSACSTAR_SEED_PER_FRAME:-True}"
ACE_G_CROSS_ITER_EVAL="${ACE_G_CROSS_ITER_EVAL:-False}"

# Keep old ACE-FCN Stage2 head contract explicit.
ACE_ENCODER_PATH="${ACE_ENCODER_PATH:-${ROOT_DIR}/ace_encoder_pretrained.pt}"
GLACE_ROOT="${GLACE_ROOT:-/home/xwh/project/glace}"
GLACE_FEAT_NAME="${GLACE_FEAT_NAME:-features.npy}"
GLACE_HEAD_CHANNELS="${GLACE_HEAD_CHANNELS:-512}"
GLACE_MLP_RATIO="${GLACE_MLP_RATIO:-1.0}"
NUM_HEAD_BLOCKS="${NUM_HEAD_BLOCKS:-4}"
ACE_LMC_GLOBAL_NORMALIZE="${ACE_LMC_GLOBAL_NORMALIZE:-True}"
ACE_LMC_GLOBAL_NOISE_STD="${ACE_LMC_GLOBAL_NOISE_STD:-0.1}"
LMC_FUSION_REFINEMENT_MODE="${LMC_FUSION_REFINEMENT_MODE:-single}"
LMC_FUSION_CASCADE_LAYERS="${LMC_FUSION_CASCADE_LAYERS:-4}"
LMC_FUSION_ASSEMBLY_GAMMA_INIT="${LMC_FUSION_ASSEMBLY_GAMMA_INIT:-0.0}"
NUM_DATA_LOADER_WORKERS="${NUM_DATA_LOADER_WORKERS:-12}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-6}"

# Indoor6 source/reference paths.
INDOOR6_ACE_ROOT="${INDOOR6_ACE_ROOT:-/home/xwh/data/indoor6_ace}"
INDOOR6_WAI_ROOT="${INDOOR6_WAI_ROOT:-/home/xwh/data/mapanything-dataset/wai_data/indoor6}"
INDOOR6_SOURCE_RUN_ROOT="${INDOOR6_SOURCE_RUN_ROOT:-/data/xwh/ace_dinov2_lmc/04_evaluation/indoor6_ace_fcn_glace_lmc_all_4gpu}"
INDOOR6_SOURCE_MEMORY_DIRNAME="${INDOOR6_SOURCE_MEMORY_DIRNAME:-memory}"
INDOOR6_SOURCE_STAGE1_SUBDIR="${INDOOR6_SOURCE_STAGE1_SUBDIR:-stage1_local_ace_memory_it12}"
INDOOR6_SCENES=(${INDOOR6_SCENES:-scene1 scene5})
INDOOR6_GPU="${INDOOR6_GPU:-0}"
INDOOR6_BEST_METRIC="${INDOOR6_BEST_METRIC:-pct25_5}"
INDOOR6_RUN_ROOT="${INDOOR6_RUN_ROOT:-/data/xwh/ace_dinov2_lmc/04_evaluation/indoor6/stage2/stage2_r2_fusion/${DATE_TAG}_scene1_scene5_it10_buf10m_final12m_h256_gpu0}"

# Wayspots source/reference paths.
WAYSPOTS_ROOT="${WAYSPOTS_ROOT:-/data/xwh/Wayspots}"
WAYSPOTS_SOURCE_RUN_ROOT="${WAYSPOTS_SOURCE_RUN_ROOT:-/data/xwh/ace_dinov2_lmc/04_evaluation/wayspots_ace_fcn_lmc_suite/20260530_181745}"
WAYSPOTS_SOURCE_MEMORY_DIRNAME="${WAYSPOTS_SOURCE_MEMORY_DIRNAME:-memory}"
WAYSPOTS_SOURCE_STAGE1_SUBDIR="${WAYSPOTS_SOURCE_STAGE1_SUBDIR:-stage1_local_ace_memory_it12}"
WAYSPOTS_SCENES=(${WAYSPOTS_SCENES:-wayspots_bears wayspots_squarebench})
WAYSPOTS_GPU="${WAYSPOTS_GPU:-1}"
WAYSPOTS_BEST_METRIC="${WAYSPOTS_BEST_METRIC:-pct50_5}"
WAYSPOTS_RUN_ROOT="${WAYSPOTS_RUN_ROOT:-/data/xwh/ace_dinov2_lmc/04_evaluation/wayspots/stage2/stage2_r2_fusion/${DATE_TAG}_bears_sq_it10_buf10m_final12m_h256_gpu1}"

print_cmd() {
  printf "%q " "$@"
  printf "\n"
}

json_quote() {
  local s="$1"
  s="${s//\\/\\\\}"
  s="${s//\"/\\\"}"
  s="${s//$'\n'/\\n}"
  printf '"%s"' "${s}"
}

join_by_space() {
  local IFS=" "
  printf "%s" "$*"
}

ensure_run_root() {
  local dataset="$1"
  local track="$2"
  local method="$3"
  local scope="$4"
  local protocol="$5"
  local gpu_tag="$6"
  local scenes="$7"
  local run_root="$8"
  mkdir -p "${run_root}/logs" "${run_root}/summaries" "${run_root}/artifacts" "${run_root}/scripts"
  cp "$0" "${run_root}/scripts/$(basename "$0")"
  cat > "${run_root}/manifest.json" <<EOF
{
  "dataset": $(json_quote "${dataset}"),
  "track": $(json_quote "${track}"),
  "method": $(json_quote "${method}"),
  "scope": $(json_quote "${scope}"),
  "protocol": $(json_quote "${protocol}"),
  "gpu_tag": $(json_quote "${gpu_tag}"),
  "scenes": $(json_quote "${scenes}"),
  "launch_script": "ace_dinov2_lmc/scripts/$(basename "$0")",
  "session": $(json_quote "${SESSION}"),
  "metric_policy": $(json_quote "Indoor6: Acc25+median, best_metric=${INDOOR6_BEST_METRIC}; Wayspots: Acc50/Acc25, best_metric=${WAYSPOTS_BEST_METRIC}; aggregate metric-wise best including post-train."),
  "stage2_contract": {
    "feature_source": "raw_backbone",
    "fusion_in_s2": true,
    "fusion_lr_ratio": ${ACE_G_FUSION_LR_RATIO},
    "global_head_mode": "glace_concat",
    "training_buffer_size": ${TRAINING_BUFFER_SIZE},
    "buffer_size_final": ${BUFFER_SIZE_FINAL},
    "batch_size": ${BATCH_SIZE},
    "lmc_iterations": ${LMC_ITERATIONS},
    "post_train_hypotheses": ${POST_TRAIN_HYPOTHESES}
  },
  "status": "planned"
}
EOF
  if [[ ! -f "${run_root}/matrix.tsv" ]]; then
    printf "dataset\tscene\tgpu\tstage2_variant\tbest_metric\tmemory_path\tstage1_checkpoint\taux_depth_root\tlog\n" > "${run_root}/matrix.tsv"
  fi
  if [[ ! -f "${run_root}/status.tsv" ]]; then
    printf "timestamp\tdataset\tscene\tstage\tstatus\texit_code\tdevice\tlog\n" > "${run_root}/status.tsv"
  fi
}

log_status() {
  local run_root="$1"
  local dataset="$2"
  local scene="$3"
  local stage="$4"
  local status="$5"
  local exit_code="$6"
  local device="$7"
  local log_file="$8"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$(date +%Y-%m-%dT%H:%M:%S)" "${dataset}" "${scene}" "${stage}" "${status}" "${exit_code}" "${device}" "${log_file}" >> "${run_root}/status.tsv"
}

find_stage1_ckpt() {
  local source_root="$1"
  local stage1_subdir="$2"
  local scene="$3"
  local pattern="best_K${NUM_LATENT_TOKENS}_it12_ace_fcn_local_stage1.pt"
  find "${source_root}/${stage1_subdir}" -type f -name "${pattern}" -path "*${scene}*" 2>/dev/null | sort | tail -n 1 || true
}

preflight_common_scene() {
  local dataset="$1"
  local scene="$2"
  local scene_root="$3"
  local memory_path="$4"
  local stage1_ckpt="$5"
  local aux_depth_root="$6"
  local aux_depth_kind="$7"
  local log_file="$8"

  if [[ ! -d "${scene_root}/train" || ! -d "${scene_root}/test" ]]; then
    echo "ERROR: missing scene train/test root for ${dataset}/${scene}: ${scene_root}" | tee -a "${log_file}" >&2
    return 2
  fi
  for split in train test; do
    if [[ ! -s "${scene_root}/${split}/${GLACE_FEAT_NAME}" ]]; then
      echo "ERROR: missing GLACE features: ${scene_root}/${split}/${GLACE_FEAT_NAME}" | tee -a "${log_file}" >&2
      return 2
    fi
  done
  if [[ ! -s "${memory_path}" ]]; then
    echo "ERROR: missing memory: ${memory_path}" | tee -a "${log_file}" >&2
    return 2
  fi
  if [[ -z "${stage1_ckpt}" || ! -s "${stage1_ckpt}" ]]; then
    echo "ERROR: missing stage1 checkpoint for ${dataset}/${scene}" | tee -a "${log_file}" >&2
    return 2
  fi
  if [[ ! -d "${aux_depth_root}/${aux_depth_kind}" && ! -d "${aux_depth_root}" ]]; then
    echo "ERROR: missing aux depth root: ${aux_depth_root} kind=${aux_depth_kind}" | tee -a "${log_file}" >&2
    return 2
  fi
}

build_train_cmd() {
  local dataset="$1"
  local scene_root="$2"
  local output_suffix="$3"
  local memory_path="$4"
  local stage1_ckpt="$5"
  local aux_depth_root="$6"
  local aux_depth_kind="$7"
  local gpu="$8"
  local run_root="$9"
  local experiment_subdir="${10}"
  local best_metric="${11}"

  TRAIN_CMD=(
    conda run --no-capture-output -n "${CONDA_ENV}" "${PYTHON_BIN}" ace_dinov2_lmc/train_ace_dinov2_lmc.py
    "${scene_root}" "${output_suffix}"
    --model_backend ace_fcn_lmc
    --data_backend ace
    --post_train_eval_scene "${scene_root}"
    --use_lmc True
    --lmc_flow ace_g
    --memory_path "${memory_path}"
    --use_scale_token False
    --ace_encoder_path "${ACE_ENCODER_PATH}"
    --ace_lmc_global_head_mode glace_concat
    --ace_lmc_local_checkpoint_path "${stage1_ckpt}"
    --ace_lmc_freeze_local_stack True
    --ace_lmc_stage2_feature_source raw_backbone
    --ace_lmc_global_feature_mode glace
    --ace_lmc_global_gate_init 1.0
    --ace_lmc_global_gate_learnable False
    --ace_lmc_global_normalize "${ACE_LMC_GLOBAL_NORMALIZE}"
    --ace_lmc_global_noise_std "${ACE_LMC_GLOBAL_NOISE_STD}"
    --glace_root "${GLACE_ROOT}"
    --glace_feat_name "${GLACE_FEAT_NAME}"
    --glace_head_channels "${GLACE_HEAD_CHANNELS}"
    --glace_mlp_ratio "${GLACE_MLP_RATIO}"
    --num_head_blocks "${NUM_HEAD_BLOCKS}"
    --best_metric "${best_metric}"
    --device "cuda:${gpu}"
    --post_train_eval_device "cuda:${gpu}"
    --experiment_root "${run_root}"
    --experiment_subdir "${experiment_subdir}"
    --lmc_iterations "${LMC_ITERATIONS}"
    --num_latent_tokens "${NUM_LATENT_TOKENS}"
    --lmc_fusion_refinement_mode "${LMC_FUSION_REFINEMENT_MODE}"
    --lmc_fusion_cascade_layers "${LMC_FUSION_CASCADE_LAYERS}"
    --lmc_fusion_assembly_gamma_init "${LMC_FUSION_ASSEMBLY_GAMMA_INIT}"
    --lmc_train_steps "${LMC_TRAIN_STEPS}"
    --lmc_warmup_steps "${LMC_WARMUP_STEPS}"
    --s1_early_stop False
    --num_data_loader_workers "${NUM_DATA_LOADER_WORKERS}"
    --eval_num_workers "${EVAL_NUM_WORKERS}"
    --eval_deterministic "${EVAL_DETERMINISTIC}"
    --eval_dsacstar_seed "${EVAL_DSACSTAR_SEED}"
    --eval_dsacstar_seed_per_frame "${EVAL_DSACSTAR_SEED_PER_FRAME}"
    --iteration_eval_seed "${ITERATION_EVAL_SEED}"
    --iteration_eval_hypotheses "${ITERATION_EVAL_HYPOTHESES}"
    --ace_g_fusion_in_s2 True
    --ace_g_fusion_lr_ratio "${ACE_G_FUSION_LR_RATIO}"
    --ace_g_cross_iter_eval "${ACE_G_CROSS_ITER_EVAL}"
    --s1_use_buffer True
    --s1_loss_mode sample_per_image
    --s1_buffer_refill_mode full
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
    --c1_aux_depth_root "${aux_depth_root}"
    --c1_aux_depth_kind "${aux_depth_kind}"
    --post_train_eval_seeds "${POST_TRAIN_EVAL_SEEDS[@]}"
    --post_train_hypotheses "${POST_TRAIN_HYPOTHESES}"
  )
}

run_one_scene() {
  local dataset="$1"
  local scene="$2"
  local gpu="$3"
  local run_root="$4"
  local scene_root memory_path stage1_ckpt aux_depth_root aux_depth_kind best_metric experiment_subdir output_suffix log_file checkpoint_pattern existing_ckpt

  if [[ "${dataset}" == "indoor6" ]]; then
    scene_root="${INDOOR6_ACE_ROOT}/${scene}"
    memory_path="${INDOOR6_SOURCE_RUN_ROOT}/${INDOOR6_SOURCE_MEMORY_DIRNAME}/${scene}/memory_ace_fcn_colmap_depth_sp_r4.pt"
    stage1_ckpt="$(find_stage1_ckpt "${INDOOR6_SOURCE_RUN_ROOT}" "${INDOOR6_SOURCE_STAGE1_SUBDIR}" "${scene}")"
    aux_depth_root="${INDOOR6_WAI_ROOT}/${scene}_train"
    aux_depth_kind="colmap_depth"
    best_metric="${INDOOR6_BEST_METRIC}"
  elif [[ "${dataset}" == "wayspots" ]]; then
    scene_root="${WAYSPOTS_ROOT}/${scene}"
    memory_path="${WAYSPOTS_SOURCE_RUN_ROOT}/${WAYSPOTS_SOURCE_MEMORY_DIRNAME}/${scene}/memory_ace_fcn_sparse_sp_r4.pt"
    stage1_ckpt="$(find_stage1_ckpt "${WAYSPOTS_SOURCE_RUN_ROOT}" "${WAYSPOTS_SOURCE_STAGE1_SUBDIR}" "${scene}")"
    aux_depth_root="${scene_root}/train/sparse_depth"
    aux_depth_kind="sparse_depth"
    best_metric="${WAYSPOTS_BEST_METRIC}"
  else
    echo "ERROR: unsupported dataset=${dataset}" >&2
    return 2
  fi

  experiment_subdir="stage2_r2_bigbuf_it${LMC_ITERATIONS}_buf10m_final12m_bs${BATCH_SIZE}_${best_metric}"
  output_suffix="ace_fcn_stage2_r2_bigbuf_${dataset}_${scene}.pt"
  log_file="${run_root}/logs/${dataset}_${scene}_gpu${gpu}.log"
  mkdir -p "$(dirname "${log_file}")"
  checkpoint_pattern="best_K${NUM_LATENT_TOKENS}_it${LMC_ITERATIONS}_${output_suffix}"
  existing_ckpt="$(find "${run_root}/${experiment_subdir}" -type f -name "${checkpoint_pattern}" -path "*${scene}*" 2>/dev/null | sort | tail -n 1 || true)"

  if ! preflight_common_scene "${dataset}" "${scene}" "${scene_root}" "${memory_path}" "${stage1_ckpt}" "${aux_depth_root}" "${aux_depth_kind}" "${log_file}"; then
    log_status "${run_root}" "${dataset}" "${scene}" "preflight" "failed" 2 "cuda:${gpu}" "${log_file}"
    return 2
  fi

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${dataset}" "${scene}" "cuda:${gpu}" "r2_bigbuf" "${best_metric}" "${memory_path}" "${stage1_ckpt}" "${aux_depth_root}/${aux_depth_kind}" "${log_file}" >> "${run_root}/matrix.tsv"

  build_train_cmd "${dataset}" "${scene_root}" "${output_suffix}" "${memory_path}" "${stage1_ckpt}" "${aux_depth_root}" "${aux_depth_kind}" "${gpu}" "${run_root}" "${experiment_subdir}" "${best_metric}"

  {
    printf "[%s] dataset=%s scene=%s gpu=%s\n" "$(date)" "${dataset}" "${scene}" "${gpu}"
    printf "run_root=%s\nmemory=%s\nstage1=%s\naux_depth=%s kind=%s\n" "${run_root}" "${memory_path}" "${stage1_ckpt}" "${aux_depth_root}" "${aux_depth_kind}"
    print_cmd "${TRAIN_CMD[@]}"
  } | tee -a "${log_file}"

  if [[ "${ACTION}" == "preflight" || "${DRY_RUN}" == "true" ]]; then
    log_status "${run_root}" "${dataset}" "${scene}" "train" "dry_run" 0 "cuda:${gpu}" "${log_file}"
    return 0
  fi

  if [[ -n "${existing_ckpt}" && "${SKIP_EXISTING}" == "true" ]]; then
    log_status "${run_root}" "${dataset}" "${scene}" "train" "skipped_existing" 0 "cuda:${gpu}" "${log_file}"
    return 0
  fi

  set +e
  "${TRAIN_CMD[@]}" 2>&1 | tee -a "${log_file}"
  local exit_code=${PIPESTATUS[0]}
  set -e

  if [[ ${exit_code} -eq 0 ]]; then
    log_status "${run_root}" "${dataset}" "${scene}" "train" "ok" 0 "cuda:${gpu}" "${log_file}"
  else
    log_status "${run_root}" "${dataset}" "${scene}" "train" "failed" "${exit_code}" "cuda:${gpu}" "${log_file}"
    [[ "${CONTINUE_ON_ERROR}" == "true" ]] || exit "${exit_code}"
  fi
  return "${exit_code}"
}

prepare_roots() {
  ensure_run_root \
    "indoor6" "stage2" "stage2_r2_fusion" "scene1_scene5" "it10_buf10m_final12m_h256" "gpu0" \
    "$(join_by_space "${INDOOR6_SCENES[@]}")" "${INDOOR6_RUN_ROOT}"
  ensure_run_root \
    "wayspots" "stage2" "stage2_r2_fusion" "bears_sq" "it10_buf10m_final12m_h256" "gpu1" \
    "$(join_by_space "${WAYSPOTS_SCENES[@]}")" "${WAYSPOTS_RUN_ROOT}"
}

run_dataset_worker() {
  local dataset="$1"
  local failed=0
  if [[ "${dataset}" == "indoor6" ]]; then
    for scene in "${INDOOR6_SCENES[@]}"; do
      run_one_scene "indoor6" "${scene}" "${INDOOR6_GPU}" "${INDOOR6_RUN_ROOT}" || failed=1
      [[ ${failed} -eq 0 || "${CONTINUE_ON_ERROR}" == "true" ]] || exit 1
    done
  elif [[ "${dataset}" == "wayspots" ]]; then
    for scene in "${WAYSPOTS_SCENES[@]}"; do
      run_one_scene "wayspots" "${scene}" "${WAYSPOTS_GPU}" "${WAYSPOTS_RUN_ROOT}" || failed=1
      [[ ${failed} -eq 0 || "${CONTINUE_ON_ERROR}" == "true" ]] || exit 1
    done
  else
    echo "ERROR: ACTION=worker requires DATASET=indoor6 or DATASET=wayspots" >&2
    exit 2
  fi
  return "${failed}"
}

launch_tmux() {
  if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "ERROR: tmux session already exists: ${SESSION}" >&2
    exit 2
  fi
  local script_path="ace_dinov2_lmc/scripts/$(basename "$0")"
  tmux new-session -d -s "${SESSION}" -n "indoor6" \
    "cd '${ROOT_DIR}' && ACTION=worker DATASET=indoor6 bash '${script_path}' 2>&1 | tee '${INDOOR6_RUN_ROOT}/logs/indoor6_worker.tmux.log'"
  tmux new-window -t "${SESSION}" -n "wayspots" \
    "cd '${ROOT_DIR}' && ACTION=worker DATASET=wayspots bash '${script_path}' 2>&1 | tee '${WAYSPOTS_RUN_ROOT}/logs/wayspots_worker.tmux.log'"
  echo "tmux session launched: ${SESSION}"
  echo "Indoor6 run_root : ${INDOOR6_RUN_ROOT}"
  echo "Wayspots run_root: ${WAYSPOTS_RUN_ROOT}"
}

case "${ACTION}" in
  preflight)
    prepare_roots
    echo "ACTION=preflight: checking inputs and printing commands; no training will start."
    run_dataset_worker "indoor6"
    run_dataset_worker "wayspots"
    echo "Preflight complete."
    echo "Indoor6 run_root : ${INDOOR6_RUN_ROOT}"
    echo "Wayspots run_root: ${WAYSPOTS_RUN_ROOT}"
    ;;
  launch)
    prepare_roots
    launch_tmux
    ;;
  worker)
    prepare_roots
    run_dataset_worker "${DATASET}"
    ;;
  *)
    echo "ERROR: unsupported ACTION=${ACTION}; use preflight | launch | worker" >&2
    exit 2
    ;;
esac
