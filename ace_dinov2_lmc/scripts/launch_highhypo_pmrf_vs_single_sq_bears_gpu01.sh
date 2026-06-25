#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/xwh/project/ace_depth"
EVAL_SCRIPT="${ROOT_DIR}/ace_dinov2_lmc/test_ace_dinov2_lmc.py"
RUN_ROOT="${RUN_ROOT:-/data/xwh/ace_dinov2_lmc/04_evaluation/highhypo_pmrf_vs_single_sq_bears_20260622_gpu01}"
HYPOTHESES="${HYPOTHESES:-512}"
SEEDS=(${SEEDS:-1305})
DRY_RUN="${DRY_RUN:-false}"

SCENE_GPU_PAIRS=(
  "wayspots_squarebench:0"
  "wayspots_bears:1"
)

# Variant roots are completed training runs. Keep this small: baseline + two PMRF-family edits.
VARIANTS=(
  "single_current:/data/xwh/ace_dinov2_lmc/04_evaluation/legacy_single_k64_repro_20260622_gpu01/current_code_legacy_k64"
  "pmrf_base:/data/xwh/ace_dinov2_lmc/04_evaluation/pmrf_repro_base_vs_centered_c0_all8_20260622_gpu01/pmrf_base"
  "centered_c0:/data/xwh/ace_dinov2_lmc/04_evaluation/pmrf_repro_base_vs_centered_c0_all8_20260622_gpu01/centered_c0"
)

find_checkpoint() {
  local root="$1"
  local scene="$2"
  find "${root}" -type f \
    -path "*${scene}*" \
    -name 'best_K64_it12_ace_fcn_local_stage1.pt' \
    | sort | tail -n 1
}

run_eval() {
  local variant="$1"
  local checkpoint_root="$2"
  local scene="$3"
  local gpu="$4"
  local seed="$5"
  local checkpoint
  checkpoint="$(find_checkpoint "${checkpoint_root}" "${scene}")"
  if [[ -z "${checkpoint}" || ! -f "${checkpoint}" ]]; then
    echo "missing checkpoint variant=${variant} scene=${scene} root=${checkpoint_root}" >&2
    return 2
  fi

  local output_dir="${RUN_ROOT}/hyp${HYPOTHESES}/${variant}/${scene}"
  local log_dir="${RUN_ROOT}/logs/hyp${HYPOTHESES}/${variant}"
  local session="${variant}_hyp${HYPOTHESES}_seed${seed}"
  mkdir -p "${output_dir}" "${log_dir}"

  echo "[$(date +%Y-%m-%dT%H:%M:%S)] eval variant=${variant} scene=${scene} gpu=${gpu} hyp=${HYPOTHESES} seed=${seed}"
  echo "checkpoint=${checkpoint}"
  if [[ "${DRY_RUN}" == "true" ]]; then
    return 0
  fi

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
      --hypotheses "${HYPOTHESES}" \
      --eval_deterministic True \
      --dsacstar_seed "${seed}" \
      --dsacstar_seed_per_frame True \
      --eval_num_workers 6 \
      --log_per_frame False \
      --lmc_log_runtime_stats True \
      --lmc_runtime_stats_interval 100 \
      --lmc_runtime_stats_max_pixels 4096 \
      2>&1 | tee "${log_dir}/${scene}_seed${seed}.log"
}

run_scene() {
  local scene="$1"
  local gpu="$2"
  local spec variant root seed
  for spec in "${VARIANTS[@]}"; do
    variant="${spec%%:*}"
    root="${spec#*:}"
    for seed in "${SEEDS[@]}"; do
      run_eval "${variant}" "${root}" "${scene}" "${gpu}" "${seed}"
    done
  done
}

mkdir -p "${RUN_ROOT}/logs/hyp${HYPOTHESES}"
cd "${ROOT_DIR}"

pids=()
for item in "${SCENE_GPU_PAIRS[@]}"; do
  scene="${item%%:*}"
  gpu="${item##*:}"
  run_scene "${scene}" "${gpu}" &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done

if [[ "${failed}" != 0 ]]; then
  echo "high-hypo eval failed" >&2
  exit 1
fi

echo "[$(date +%Y-%m-%dT%H:%M:%S)] all done: ${RUN_ROOT}"
