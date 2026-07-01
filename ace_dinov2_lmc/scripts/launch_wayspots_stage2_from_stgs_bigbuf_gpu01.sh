#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

CONDA_ENV="${CONDA_ENV:-mapanything}"
PYTHON_BIN="${PYTHON_BIN:-python}"
WAYSPOTS_ROOT="${WAYSPOTS_ROOT:-/data/xwh/Wayspots}"
MEMORY_ROOT="${MEMORY_ROOT:-/data/xwh/ace_dinov2_lmc/04_evaluation/wayspots_ace_fcn_lmc_suite/20260530_181745/memory}"
SOURCE_STAGE1_ROOT="${SOURCE_STAGE1_ROOT:-/data/xwh/ace_dinov2_lmc/04_evaluation/wayspots/stgs_guided_bigbuf/20260627_stage1_it6_buf10m_baseline_anchor_gpu23}"
RUN_ROOT="${RUN_ROOT:-/data/xwh/ace_dinov2_lmc/04_evaluation/wayspots/stage2/stgs_stage1_transfer/20260628_bears_cubes_stage2_r2_it10_buf10m_final12m_gpu01}"
LOG_DIR="${RUN_ROOT}/logs"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"

GPU_BASELINE="${GPU_BASELINE:-0}"
GPU_ANCHOR="${GPU_ANCHOR:-1}"
STAGE2_SERIAL_PAIRS="${STAGE2_SERIAL_PAIRS:-False}"
CONTINUE_ON_FAILURE="${CONTINUE_ON_FAILURE:-False}"

ACE_ENCODER_PATH="${ACE_ENCODER_PATH:-${ROOT_DIR}/ace_encoder_pretrained.pt}"
GLACE_ROOT="${GLACE_ROOT:-/home/xwh/project/glace}"
GLACE_FEAT_NAME="${GLACE_FEAT_NAME:-features.npy}"

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
BEST_METRIC="${BEST_METRIC:-pct50_5}"

ACE_LMC_GLOBAL_NORMALIZE="${ACE_LMC_GLOBAL_NORMALIZE:-True}"
ACE_LMC_GLOBAL_NOISE_STD="${ACE_LMC_GLOBAL_NOISE_STD:-0.1}"
GLACE_HEAD_CHANNELS="${GLACE_HEAD_CHANNELS:-512}"
GLACE_MLP_RATIO="${GLACE_MLP_RATIO:-1.0}"
NUM_HEAD_BLOCKS="${NUM_HEAD_BLOCKS:-4}"
LMC_FUSION_REFINEMENT_MODE="${LMC_FUSION_REFINEMENT_MODE:-single}"
LMC_FUSION_CASCADE_LAYERS="${LMC_FUSION_CASCADE_LAYERS:-4}"
LMC_FUSION_ASSEMBLY_GAMMA_INIT="${LMC_FUSION_ASSEMBLY_GAMMA_INIT:-0.0}"
NUM_DATA_LOADER_WORKERS="${NUM_DATA_LOADER_WORKERS:-12}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-6}"

SCENES=(${SCENES:-wayspots_bears wayspots_cubes})

mkdir -p "${LOG_DIR}"
SUPERVISOR_LOG="${LOG_DIR}/supervisor_${STAMP}.log"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${SUPERVISOR_LOG}"
}

print_cmd() {
  printf "%q " "$@"
  printf "\n"
}

stage1_variant_dir() {
  local variant="$1"
  if [[ "${variant}" == "baseline" ]]; then
    printf "baseline/stage1_local_ace_memory_it6_bigbuf"
  else
    printf "anchor_v2_f010_w05/stage1_local_stgs_anchor_bigbuf_it6"
  fi
}

find_stage1_ckpt() {
  local scene="$1"
  local variant="$2"
  local variant_dir
  variant_dir="$(stage1_variant_dir "${variant}")"
  find "${SOURCE_STAGE1_ROOT}/${scene}/${variant_dir}" -type f -name "best_K64_it6_*.pt" 2>/dev/null | sort | tail -n 1 || true
}

preflight_scene_variant() {
  local scene="$1"
  local variant="$2"
  local scene_root="${WAYSPOTS_ROOT}/${scene}"
  local memory_path="${MEMORY_ROOT}/${scene}/memory_ace_fcn_sparse_sp_r4.pt"
  local stage1_ckpt="$3"

  [[ -d "${scene_root}/train" ]] || { log "missing train split: ${scene_root}/train"; return 2; }
  [[ -d "${scene_root}/test" ]] || { log "missing test split: ${scene_root}/test"; return 2; }
  [[ -s "${scene_root}/train/${GLACE_FEAT_NAME}" ]] || { log "missing GLACE train feature: ${scene_root}/train/${GLACE_FEAT_NAME}"; return 2; }
  [[ -s "${scene_root}/test/${GLACE_FEAT_NAME}" ]] || { log "missing GLACE test feature: ${scene_root}/test/${GLACE_FEAT_NAME}"; return 2; }
  [[ -s "${memory_path}" ]] || { log "missing memory: ${memory_path}"; return 2; }
  [[ -s "${stage1_ckpt}" ]] || { log "missing stage1 checkpoint for ${scene}/${variant}: ${stage1_ckpt}"; return 2; }
  [[ -d "${scene_root}/train/sparse_depth" ]] || { log "missing sparse depth: ${scene_root}/train/sparse_depth"; return 2; }
}

build_train_cmd() {
  local scene="$1"
  local variant="$2"
  local gpu="$3"
  local stage1_ckpt="$4"
  local scene_root="${WAYSPOTS_ROOT}/${scene}"
  local memory_path="${MEMORY_ROOT}/${scene}/memory_ace_fcn_sparse_sp_r4.pt"
  local tag="${scene}_${variant}_stage2_r2_it${LMC_ITERATIONS}_buf10m_${STAMP}"
  local output_suffix="${tag}.pt"
  local experiment_root="${RUN_ROOT}/${scene}/${variant}"
  local experiment_subdir="stage2_r2_bigbuf_from_${variant}_it${LMC_ITERATIONS}_buf10m_final12m_bs${BATCH_SIZE}_${BEST_METRIC}"

  TRAIN_CMD=(
    conda run --no-capture-output -n "${CONDA_ENV}" "${PYTHON_BIN}" ace_dinov2_lmc/train_ace_dinov2_lmc.py
    "${scene_root}" "${output_suffix}"
    --run_name "${tag}"
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
    --best_metric "${BEST_METRIC}"
    --device "cuda:${gpu}"
    --post_train_eval_device "cuda:${gpu}"
    --experiment_root "${experiment_root}"
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
    --c1_aux_depth_root "${scene_root}/train/sparse_depth"
    --c1_aux_depth_kind sparse_depth
    --post_train_eval_seeds "${POST_TRAIN_EVAL_SEEDS[@]}"
    --post_train_hypotheses "${POST_TRAIN_HYPOTHESES}"
  )
}

run_train() {
  local scene="$1"
  local variant="$2"
  local gpu="$3"
  local stage1_ckpt="$4"
  local log_file="${LOG_DIR}/${scene}_${variant}_stage2_gpu${gpu}_${STAMP}.log"

  build_train_cmd "${scene}" "${variant}" "${gpu}" "${stage1_ckpt}"
  {
    printf "[%s] scene=%s variant=%s gpu=%s\n" "$(date)" "${scene}" "${variant}" "${gpu}"
    printf "stage1=%s\n" "${stage1_ckpt}"
    print_cmd "${TRAIN_CMD[@]}"
  } >> "${log_file}"

  log "start ${scene} ${variant} Stage2 on gpu${gpu}; log=${log_file}"
  env -u CUDA_VISIBLE_DEVICES "${TRAIN_CMD[@]}" >> "${log_file}" 2>&1
  log "done ${scene} ${variant} Stage2 on gpu${gpu}"
}

launch_pair() {
  local scene="$1"
  local baseline_ckpt anchor_ckpt
  baseline_ckpt="$(find_stage1_ckpt "${scene}" baseline)"
  anchor_ckpt="$(find_stage1_ckpt "${scene}" anchor)"
  preflight_scene_variant "${scene}" baseline "${baseline_ckpt}"
  preflight_scene_variant "${scene}" anchor "${anchor_ckpt}"

  log "launch pair ${scene}: baseline gpu${GPU_BASELINE}, anchor-v2 gpu${GPU_ANCHOR}"
  local rc_baseline=0
  local rc_anchor=0
  if [[ "${STAGE2_SERIAL_PAIRS}" == "True" || "${GPU_BASELINE}" == "${GPU_ANCHOR}" ]]; then
    log "serial pair mode for ${scene}"
    run_train "${scene}" baseline "${GPU_BASELINE}" "${baseline_ckpt}" || rc_baseline=$?
    run_train "${scene}" anchor_v2_f010_w05 "${GPU_ANCHOR}" "${anchor_ckpt}" || rc_anchor=$?
  else
    run_train "${scene}" baseline "${GPU_BASELINE}" "${baseline_ckpt}" &
    local pid_baseline=$!
    run_train "${scene}" anchor_v2_f010_w05 "${GPU_ANCHOR}" "${anchor_ckpt}" &
    local pid_anchor=$!

    wait "${pid_baseline}" || rc_baseline=$?
    wait "${pid_anchor}" || rc_anchor=$?
  fi
  if [[ "${rc_baseline}" -ne 0 || "${rc_anchor}" -ne 0 ]]; then
    log "pair failed ${scene}: baseline_rc=${rc_baseline}, anchor_rc=${rc_anchor}"
    if [[ "${CONTINUE_ON_FAILURE}" == "True" ]]; then
      return 0
    fi
    exit 1
  fi
  log "pair complete ${scene}"
}

cat >"${RUN_ROOT}/matrix_plan_${STAMP}.txt" <<EOF
purpose: Stage2 transfer check from completed Stage1 big-buffer baseline vs anchor-v2
source_stage1_root: ${SOURCE_STAGE1_ROOT}
run_root: ${RUN_ROOT}
scenes: ${SCENES[*]}
stage2_variant: r2_bigbuf_glace_concat
lmc_iterations: ${LMC_ITERATIONS}
training_buffer_size: ${TRAINING_BUFFER_SIZE}
buffer_size_final: ${BUFFER_SIZE_FINAL}
batch_size: ${BATCH_SIZE}
best_metric: ${BEST_METRIC}
gpu_baseline: ${GPU_BASELINE}
gpu_anchor: ${GPU_ANCHOR}
stage2_serial_pairs: ${STAGE2_SERIAL_PAIRS}
continue_on_failure: ${CONTINUE_ON_FAILURE}
post_train_eval_seeds: ${POST_TRAIN_EVAL_SEEDS[*]}
post_train_hypotheses: ${POST_TRAIN_HYPOTHESES}
EOF

log "plan: ${RUN_ROOT}/matrix_plan_${STAMP}.txt"
for scene in "${SCENES[@]}"; do
  launch_pair "${scene}"
done
log "all Stage2 transfer jobs complete"
