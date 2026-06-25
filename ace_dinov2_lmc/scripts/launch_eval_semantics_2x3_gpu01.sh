#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/xwh/project/ace_depth"
EVAL_SCRIPT="${ROOT_DIR}/ace_dinov2_lmc/test_ace_dinov2_lmc.py"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/data/xwh/ace_dinov2_lmc/04_evaluation/pmrf_common_scale_matrix_20260620_gpu0123/centered_s0}"
RUN_ROOT="${RUN_ROOT:-/data/xwh/ace_dinov2_lmc/04_evaluation/pmrf_eval_semantics_2x3_20260620_gpu01}"
DRY_RUN="${DRY_RUN:-false}"

SCENES=(wayspots_squarebench wayspots_cubes)
GPUS=(0 1)
SEEDS=(1305 2026 4242)
CONFIGS=(
  "c0_cap0:0.0:0.0"
  "c0_cap006:0.0:0.006"
  "c0_cap008:0.0:0.008"
  "c1_cap0:1.0:0.0"
  "c1_cap006:1.0:0.006"
  "c1_cap008:1.0:0.008"
)

find_checkpoint() {
  local scene="$1"
  find "${CHECKPOINT_ROOT}/${scene}" -type f \
    -name 'best_K64_it12_ace_fcn_local_stage1.pt' | sort | tail -n 1
}

run_eval() {
  local scene="$1"
  local gpu="$2"
  local checkpoint="$3"
  local config="$4"
  local common_scale="$5"
  local cap="$6"
  local seed="$7"
  local output_dir="${RUN_ROOT}/${config}/${scene}"
  local log_dir="${RUN_ROOT}/logs/${config}"
  local session="override_${config}_seed${seed}"

  echo "[$(date +%Y-%m-%dT%H:%M:%S)] scene=${scene} gpu=${gpu} config=${config} common=${common_scale} cap=${cap} seed=${seed}"
  if [[ "${DRY_RUN}" == "true" ]]; then
    return 0
  fi

  mkdir -p "${output_dir}" "${log_dir}"
  OMP_NUM_THREADS=8 OMP_DYNAMIC=FALSE \
    conda run --no-capture-output -n mapanything python "${EVAL_SCRIPT}" \
      "/data/xwh/Wayspots/${scene}" \
      "${checkpoint}" \
      --data_backend ace \
      --ace_encoder_path "${ROOT_DIR}/ace_encoder_pretrained.pt" \
      --device "cuda:${gpu}" \
      --output_dir "${output_dir}" \
      --session "${session}" \
      --image_resolution 512 \
      --hypotheses 256 \
      --eval_deterministic True \
      --dsacstar_seed "${seed}" \
      --dsacstar_seed_per_frame True \
      --eval_num_workers 6 \
      --log_per_frame False \
      --lmc_log_runtime_stats True \
      --lmc_runtime_stats_interval 100 \
      --lmc_runtime_stats_max_pixels 4096 \
      --eval_lmc_fusion_reread_common_scale_override "${common_scale}" \
      --eval_lmc_fusion_reread_effective_ratio_cap_override "${cap}" \
      2>&1 | tee "${log_dir}/${scene}_seed${seed}.log"
}

run_scene_matrix() {
  local scene="$1"
  local gpu="$2"
  local checkpoint
  checkpoint="$(find_checkpoint "${scene}")"
  if [[ -z "${checkpoint}" || ! -f "${checkpoint}" ]]; then
    echo "missing checkpoint for ${scene} under ${CHECKPOINT_ROOT}" >&2
    return 2
  fi
  echo "[$(date +%Y-%m-%dT%H:%M:%S)] scene begin ${scene} gpu=${gpu} checkpoint=${checkpoint}"
  local spec config common_scale cap seed
  for spec in "${CONFIGS[@]}"; do
    IFS=: read -r config common_scale cap <<< "${spec}"
    for seed in "${SEEDS[@]}"; do
      run_eval "${scene}" "${gpu}" "${checkpoint}" "${config}" "${common_scale}" "${cap}" "${seed}"
    done
  done
  echo "[$(date +%Y-%m-%dT%H:%M:%S)] scene done ${scene}"
}

if [[ "${DRY_RUN}" != "true" ]]; then
  mkdir -p "${RUN_ROOT}/logs"
fi
cd "${ROOT_DIR}"
pids=()
for idx in "${!SCENES[@]}"; do
  run_scene_matrix "${SCENES[$idx]}" "${GPUS[$idx]}" &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if [[ "${failed}" != 0 ]]; then
  echo "eval semantics matrix failed" >&2
  exit 1
fi

echo "[$(date +%Y-%m-%dT%H:%M:%S)] all done: ${RUN_ROOT}"
