#!/usr/bin/env bash
set -euo pipefail

OLD_ROOT="${OLD_ROOT:-/data/xwh/.tmp/ace_depth_6578984_repro}"
OLD_COMMIT="${OLD_COMMIT:-65789847cb0c8f3b0f5cc5450c6f19844b275b90}"
CURRENT_ROOT="${CURRENT_ROOT:-/home/xwh/project/ace_depth}"
RUN_ROOT="${RUN_ROOT:-/data/xwh/ace_dinov2_lmc/04_evaluation/legacy_single_k64_repro_20260622_old6578984_quick_gpu01}"
MEMORY_ROOT="${MEMORY_ROOT:-/data/xwh/ace_dinov2_lmc/04_evaluation/wayspots_ace_fcn_lmc_suite/20260530_181745/memory}"
DATA_ROOT="${DATA_ROOT:-/data/xwh/Wayspots}"
CONDA_ENV="${CONDA_ENV:-mapanything}"
PYTHON_BIN="${PYTHON_BIN:-python}"

LMC_ITERATIONS="${LMC_ITERATIONS:-12}"
NUM_LATENT_TOKENS="${NUM_LATENT_TOKENS:-64}"
ITERATION_EVAL_HYPOTHESES="${ITERATION_EVAL_HYPOTHESES:-256}"
POST_TRAIN_EVAL_SEEDS=(${POST_TRAIN_EVAL_SEEDS:-1305 2026 4242})
POST_TRAIN_HYPOTHESES="${POST_TRAIN_HYPOTHESES:-256}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OMP_DYNAMIC="${OMP_DYNAMIC:-FALSE}"

log() {
  echo "[$(date +%Y-%m-%dT%H:%M:%S)] $*"
}

check_old_root() {
  local actual
  actual="$(git -C "${OLD_ROOT}" rev-parse HEAD)"
  if [[ "${actual}" != "${OLD_COMMIT}" ]]; then
    echo "old worktree commit mismatch: root=${OLD_ROOT} actual=${actual} expected=${OLD_COMMIT}" >&2
    return 2
  fi
}

run_scene() {
  local gpu="$1"
  local scene="$2"
  local log_dir="${RUN_ROOT}/logs/gpu${gpu}"
  local scene_root="${DATA_ROOT}/${scene}"
  local memory_path="${MEMORY_ROOT}/${scene}/memory_ace_fcn_sparse_sp_r4.pt"
  local aux_depth_dir="${scene_root}/train/sparse_depth"

  mkdir -p "${log_dir}"
  log "old6578984 quick scene=${scene} gpu=${gpu} run_root=${RUN_ROOT}"

  (
    cd "${OLD_ROOT}"
    conda run --no-capture-output -n "${CONDA_ENV}" "${PYTHON_BIN}" \
      ace_dinov2_lmc/train_ace_dinov2_lmc.py \
      "${scene_root}" ace_fcn_local_stage1.pt \
      --model_backend ace_fcn_lmc \
      --data_backend ace \
      --use_lmc True \
      --lmc_flow ace_g \
      --memory_path "${memory_path}" \
      --use_scale_token False \
      --ace_encoder_path "${CURRENT_ROOT}/ace_encoder_pretrained.pt" \
      --ace_lmc_global_head_mode none \
      --device "cuda:${gpu}" \
      --post_train_eval_device "cuda:${gpu}" \
      --experiment_root "${RUN_ROOT}" \
      --experiment_subdir stage1_local_ace_memory_it12 \
      --lmc_iterations "${LMC_ITERATIONS}" \
      --num_latent_tokens "${NUM_LATENT_TOKENS}" \
      --iteration_eval_hypotheses "${ITERATION_EVAL_HYPOTHESES}" \
      --ace_g_fusion_in_s2 True \
      --ace_g_cross_iter_eval True \
      --s1_use_buffer True \
      --s1_loss_mode sample_per_image \
      --s1_buffer_refill_mode full \
      --image_resolution 512 \
      --batch_size 4096 \
      --training_buffer_size 2800000 \
      --buffer_size_final 7600000 \
      --buffer_on_cpu True \
      --buffer_on_cpu_final True \
      --samples_per_image 512 \
      --buffer_sample_valid_coords True \
      --buffer_valid_coord_sample_ratio 1.0 \
      --buffer_valid_coord_neighbor_radius 1 \
      --buffer_valid_coord_neighbor_mode cross \
      --c1_aux_depth_root "${aux_depth_dir}" \
      --c1_aux_depth_kind sparse_depth \
      --post_train_eval_seeds "${POST_TRAIN_EVAL_SEEDS[@]}" \
      --post_train_hypotheses "${POST_TRAIN_HYPOTHESES}"
  ) 2>&1 | tee "${log_dir}/${scene}.log"
}

main() {
  check_old_root
  mkdir -p "${RUN_ROOT}/logs"
  log "run_root=${RUN_ROOT}"
  log "old_root=${OLD_ROOT}"
  log "old_commit=${OLD_COMMIT}"
  log "scenes=gpu0:wayspots_bears gpu1:wayspots_squarebench"

  run_scene 0 wayspots_bears &
  pid0=$!
  run_scene 1 wayspots_squarebench &
  pid1=$!

  rc=0
  wait "${pid0}" || rc=1
  wait "${pid1}" || rc=1
  if [[ "${rc}" != 0 ]]; then
    log "quick old6578984 failed"
    return "${rc}"
  fi
  log "quick old6578984 done"
}

main "$@"
