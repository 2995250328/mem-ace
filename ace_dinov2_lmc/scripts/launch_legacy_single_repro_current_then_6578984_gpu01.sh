#!/usr/bin/env bash
set -euo pipefail

CURRENT_ROOT="${CURRENT_ROOT:-/home/xwh/project/ace_depth}"
OLD_COMMIT="${OLD_COMMIT:-65789847cb0c8f3b0f5cc5450c6f19844b275b90}"
OLD_ROOT="${OLD_ROOT:-/data/xwh/.tmp/ace_depth_6578984_repro}"
RUN_ROOT_BASE="${RUN_ROOT_BASE:-/data/xwh/ace_dinov2_lmc/04_evaluation/legacy_single_k64_repro_20260622_gpu01}"
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

SCENES=(
  wayspots_bears
  wayspots_squarebench
  wayspots_cubes
  wayspots_tendrils
  wayspots_inscription
  wayspots_lawn
  wayspots_map
  wayspots_therock
)

log() {
  echo "[$(date +%Y-%m-%dT%H:%M:%S)] $*"
}

ensure_old_worktree() {
  if [[ ! -e "${OLD_ROOT}" ]]; then
    log "creating old worktree root=${OLD_ROOT} commit=${OLD_COMMIT}"
    git -C "${CURRENT_ROOT}" worktree add --detach "${OLD_ROOT}" "${OLD_COMMIT}"
  fi

  local actual
  actual="$(git -C "${OLD_ROOT}" rev-parse HEAD)"
  if [[ "${actual}" != "${OLD_COMMIT}" ]]; then
    echo "old worktree commit mismatch: root=${OLD_ROOT} actual=${actual} expected=${OLD_COMMIT}" >&2
    return 2
  fi
}

run_scene() {
  local project_root="$1"
  local phase="$2"
  local gpu="$3"
  local scene="$4"
  local run_root="${RUN_ROOT_BASE}/${phase}"
  local log_dir="${run_root}/logs/gpu${gpu}"
  local scene_root="${DATA_ROOT}/${scene}"
  local memory_path="${MEMORY_ROOT}/${scene}/memory_ace_fcn_sparse_sp_r4.pt"
  local aux_depth_dir="${scene_root}/train/sparse_depth"
  local output_suffix="ace_fcn_local_stage1.pt"

  if [[ ! -d "${scene_root}" ]]; then
    echo "missing scene root: ${scene_root}" >&2
    return 2
  fi
  if [[ ! -f "${memory_path}" ]]; then
    echo "missing memory: ${memory_path}" >&2
    return 2
  fi
  if [[ ! -d "${aux_depth_dir}" ]]; then
    echo "missing sparse depth dir: ${aux_depth_dir}" >&2
    return 2
  fi

  mkdir -p "${log_dir}"
  log "phase=${phase} scene=${scene} gpu=${gpu} root=${project_root}"

  (
    cd "${project_root}"
    conda run --no-capture-output -n "${CONDA_ENV}" "${PYTHON_BIN}" \
      ace_dinov2_lmc/train_ace_dinov2_lmc.py \
      "${scene_root}" "${output_suffix}" \
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
      --experiment_root "${run_root}" \
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

run_worker() {
  local project_root="$1"
  local phase="$2"
  local gpu="$3"
  shift 3
  local scene
  for scene in "$@"; do
    run_scene "${project_root}" "${phase}" "${gpu}" "${scene}"
  done
}

run_phase() {
  local project_root="$1"
  local phase="$2"
  mkdir -p "${RUN_ROOT_BASE}/${phase}/logs"
  log "phase begin ${phase} root=${project_root} run_root=${RUN_ROOT_BASE}/${phase}"
  log "legacy flags: single by default, eval_deterministic default False, s1_early_stop default True"

  local gpu0_scenes=()
  local gpu1_scenes=()
  local i
  for i in "${!SCENES[@]}"; do
    if (( i % 2 == 0 )); then
      gpu0_scenes+=("${SCENES[$i]}")
    else
      gpu1_scenes+=("${SCENES[$i]}")
    fi
  done

  local rc=0
  run_worker "${project_root}" "${phase}" 0 "${gpu0_scenes[@]}" &
  local pid0=$!
  run_worker "${project_root}" "${phase}" 1 "${gpu1_scenes[@]}" &
  local pid1=$!

  wait "${pid0}" || rc=1
  wait "${pid1}" || rc=1
  if [[ "${rc}" != 0 ]]; then
    log "phase failed ${phase}"
    return "${rc}"
  fi
  log "phase done ${phase}"
}

main() {
  mkdir -p "${RUN_ROOT_BASE}/logs"
  log "run_root_base=${RUN_ROOT_BASE}"
  log "scenes=${SCENES[*]}"
  log "gpu_policy=only cuda:0,cuda:1"

  run_phase "${CURRENT_ROOT}" "current_code_legacy_k64"
  ensure_old_worktree
  run_phase "${OLD_ROOT}" "original_6578984_legacy_k64"
  log "all phases done"
}

main "$@"
