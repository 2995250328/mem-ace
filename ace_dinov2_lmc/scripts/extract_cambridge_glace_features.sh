#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

CONDA_ENV="${CONDA_ENV:-mapanything}"
CAMBRIDGE_ROOT="${CAMBRIDGE_ROOT:-/data/xwh/Cambridge}"
SCENES_STR="${SCENES_STR:-Cambridge_GreatCourt Cambridge_KingsCollege Cambridge_OldHospital Cambridge_ShopFacade Cambridge_StMarysChurch}"
GLACE_ROOT="${GLACE_ROOT:-/home/xwh/project/glace}"
GLACE_CHECKPOINT="${GLACE_CHECKPOINT:-/home/xwh/project/glace/CVPR23_DeitS_Rerank.pth}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DRY_RUN="${DRY_RUN:-false}"
SKIP_EXISTING="${SKIP_EXISTING:-true}"

read -r -a SCENES <<< "${SCENES_STR}"

for scene in "${SCENES[@]}"; do
  scene_root="${CAMBRIDGE_ROOT}/${scene}"
  if [[ ! -d "${scene_root}/train/rgb" || ! -d "${scene_root}/test/rgb" ]]; then
    echo "ERROR: missing Cambridge ACE train/test rgb dirs: ${scene_root}" >&2
    exit 2
  fi
  if [[ "${SKIP_EXISTING}" == "true" && -s "${scene_root}/train/features.npy" && -s "${scene_root}/test/features.npy" ]]; then
    echo "[skip] ${scene}: train/test features.npy already exist."
    continue
  fi

  CMD=(
    conda run --no-capture-output -n "${CONDA_ENV}" python "${GLACE_ROOT}/datasets/extract_features.py"
    "${scene_root}"
    --checkpoint "${GLACE_CHECKPOINT}"
    --batch_size "${BATCH_SIZE}"
    --num_workers "${NUM_WORKERS}"
  )
  printf "CAMBRIDGE_GLACE_FEATURE_CMD[%s]: " "${scene}"
  printf "%q " "${CMD[@]}"
  printf "\n"
  if [[ "${DRY_RUN}" != "true" ]]; then
    "${CMD[@]}"
  fi
done

if [[ "${DRY_RUN}" != "true" ]]; then
  python ace_dinov2_lmc/tools/audit_cambridge_data.py "${CAMBRIDGE_ROOT}" --scenes "${SCENES[@]}" --require-features
fi
