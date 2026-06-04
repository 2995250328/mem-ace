#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

CAMBRIDGE_ROOT="${CAMBRIDGE_ROOT:-/data/xwh/Cambridge}"
SCENES_STR="${SCENES_STR:-Cambridge_GreatCourt Cambridge_KingsCollege Cambridge_OldHospital Cambridge_ShopFacade Cambridge_StMarysChurch}"
CONDA_ENV="${CONDA_ENV:-mapanything}"
MAX_DEPTH_M="${MAX_DEPTH_M:-1000}"
DRY_RUN="${DRY_RUN:-false}"

read -r -a SCENES <<< "${SCENES_STR}"

CMD=(
  conda run --no-capture-output -n "${CONDA_ENV}" python -m ace_dinov2_lmc.tools.project_cambridge_nvm_sparse_depth
  "${CAMBRIDGE_ROOT}"
  --scenes "${SCENES[@]}"
  --max-depth-m "${MAX_DEPTH_M}"
)

printf "CAMBRIDGE_SPARSE_DEPTH_CMD: "
printf "%q " "${CMD[@]}"
printf "\n"

if [[ "${DRY_RUN}" == "true" ]]; then
  exit 0
fi

"${CMD[@]}"
conda run --no-capture-output -n "${CONDA_ENV}" python ace_dinov2_lmc/tools/audit_cambridge_data.py "${CAMBRIDGE_ROOT}" --scenes "${SCENES[@]}" --require-sparse-depth
