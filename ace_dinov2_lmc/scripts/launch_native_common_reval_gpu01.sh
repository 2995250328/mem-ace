#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/xwh/project/ace_depth"
EVAL_SCRIPT="${ROOT_DIR}/ace_dinov2_lmc/test_ace_dinov2_lmc.py"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/data/xwh/ace_dinov2_lmc/04_evaluation/pmrf_common_scale_matrix_20260620_gpu0123}"
RUN_ROOT="${RUN_ROOT:-/data/xwh/ace_dinov2_lmc/04_evaluation/pmrf_native_common_reval_20260621_gpu01}"
DRY_RUN="${DRY_RUN:-false}"

SCENES=(wayspots_squarebench wayspots_cubes wayspots_bears wayspots_tendrils)
SEEDS=(1305 2026 4242)
VARIANTS=(
  "c0_cap0:centered_s0:0.0"
  "c05_cap0:centered_s05:0.5"
)

find_checkpoint() {
  local source_variant="$1"
  local scene="$2"
  find "${CHECKPOINT_ROOT}/${source_variant}/${scene}" -type f \
    -name 'best_K64_it12_ace_fcn_local_stage1.pt' | sort | tail -n 1
}

run_eval() {
  local config="$1"
  local source_variant="$2"
  local common_scale="$3"
  local scene="$4"
  local gpu="$5"
  local seed="$6"
  local checkpoint="$7"
  local output_dir="${RUN_ROOT}/${config}/${scene}"
  local log_dir="${RUN_ROOT}/logs/${config}"
  local session="native_${config}_seed${seed}"

  echo "[$(date +%Y-%m-%dT%H:%M:%S)] scene=${scene} gpu=${gpu} config=${config} source=${source_variant} common=${common_scale} cap=0 seed=${seed}"
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
      --eval_lmc_fusion_reread_effective_ratio_cap_override 0.0 \
      2>&1 | tee "${log_dir}/${scene}_seed${seed}.log"
}

run_variant_scene() {
  local spec="$1"
  local scene="$2"
  local gpu="$3"
  local config source_variant common_scale checkpoint seed
  IFS=: read -r config source_variant common_scale <<< "${spec}"
  checkpoint="$(find_checkpoint "${source_variant}" "${scene}")"
  if [[ -z "${checkpoint}" || ! -f "${checkpoint}" ]]; then
    echo "missing checkpoint for ${source_variant}/${scene}" >&2
    return 2
  fi
  echo "[$(date +%Y-%m-%dT%H:%M:%S)] job begin config=${config} scene=${scene} gpu=${gpu} checkpoint=${checkpoint}"
  for seed in "${SEEDS[@]}"; do
    run_eval "${config}" "${source_variant}" "${common_scale}" "${scene}" "${gpu}" "${seed}" "${checkpoint}"
  done
  echo "[$(date +%Y-%m-%dT%H:%M:%S)] job done config=${config} scene=${scene} gpu=${gpu}"
}

cd "${ROOT_DIR}"
if [[ "${DRY_RUN}" != "true" ]]; then
  mkdir -p "${RUN_ROOT}/logs"
fi

for scene in "${SCENES[@]}"; do
  run_variant_scene "${VARIANTS[0]}" "${scene}" 0 &
  pid0="$!"
  run_variant_scene "${VARIANTS[1]}" "${scene}" 1 &
  pid1="$!"
  failed=0
  wait "${pid0}" || failed=1
  wait "${pid1}" || failed=1
  if [[ "${failed}" != 0 ]]; then
    echo "native common re-eval failed for ${scene}" >&2
    exit 1
  fi
done

echo "[$(date +%Y-%m-%dT%H:%M:%S)] all done: ${RUN_ROOT}"
