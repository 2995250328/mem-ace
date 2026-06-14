#!/usr/bin/env bash
set -euo pipefail

# Preflight for paper-facing GPU0/1 experiment scripts.
# This does not start real training/evaluation; it only validates syntax,
# required MAREPO files/imports, and dry-run command expansion.
# Run from /home/xwh/project/ace_depth:
#   bash ace_dinov2_lmc/scripts/verify_paper_experiment_scripts_gpu01.sh

ROOT_DIR="${ROOT_DIR:-/home/xwh/project/ace_depth}"
REPO_ROOT="${REPO_ROOT:-${ROOT_DIR}/ace_dinov2_lmc}"
MAREPO_ROOT="${MAREPO_ROOT:-/home/xwh/project/marepo}"
CONDA_ENV="${CONDA_ENV:-mapanything}"
TMP_ROOT="${TMP_ROOT:-/tmp/ace_paper_scripts_preflight_$(date +%Y%m%d_%H%M%S)}"

cd "${REPO_ROOT}"
mkdir -p "${TMP_ROOT}"

log() { printf '[preflight] %s\n' "$*"; }
require_file() {
  local path="$1" label="$2"
  if [[ ! -s "${path}" ]]; then
    echo "ERROR: missing ${label}: ${path}" >&2
    exit 2
  fi
}
require_dir() {
  local path="$1" label="$2"
  if [[ ! -d "${path}" ]]; then
    echo "ERROR: missing ${label}: ${path}" >&2
    exit 2
  fi
}

log "bash syntax"
bash -n \
  scripts/run_wayspots_marepo_s_gpu01.sh \
  scripts/run_wayspots_aceg25_baseline_gpu01.sh \
  scripts/run_wayspots_glace_lmc_stage2_film_gpu01.sh \
  scripts/run_indoor6_dino_k_sweep_gpu01.sh \
  scripts/run_indoor6_dino_forceglobal_s1_periter_remaining_gpu01.sh

log "MAREPO files"
require_dir "${MAREPO_ROOT}" "MAREPO root"
require_file "${MAREPO_ROOT}/test_marepo.py" "MAREPO test_marepo.py"
require_file "${MAREPO_ROOT}/ace_encoder_pretrained.pt" "MAREPO ACE encoder"
require_file "${MAREPO_ROOT}/transformer/config/nerf_focal_12T1R_256_homo.json" "MAREPO transformer config"
require_file "${MAREPO_ROOT}/logs/paper_model/marepo_s_wayspots_bears/marepo_s_wayspots_bears.pt" "MAREPO-S bears model"
require_file "${MAREPO_ROOT}/logs/wayspots_pretrain/test/wayspots_bears/wayspots_bears.pt" "Wayspots bears ACE head"
require_dir "/data/xwh/Wayspots/wayspots_bears/test/rgb" "Wayspots bears test/rgb"
require_dir "/data/xwh/Wayspots/wayspots_bears/test/poses" "Wayspots bears test/poses"
require_dir "/data/xwh/Wayspots/wayspots_bears/test/calibration" "Wayspots bears test/calibration"

log "MAREPO import smoke"
(cd "${MAREPO_ROOT}" && conda run --no-capture-output -n "${CONDA_ENV}" python - <<'INNER_PY'
import torch
import dataset
from marepo.marepo_network import Regressor
import pytorch3d.transforms as T
print('torch', torch.__version__)
print('rot6d', tuple(T.rotation_6d_to_matrix(torch.randn(2, 6)).shape))
print('marepo_import_ok')
INNER_PY
)

log "MAREPO-S dry run"
DRY_RUN=true \
SCENES_STR="wayspots_bears wayspots_cubes" \
RUN_ROOT="${TMP_ROOT}/marepo_s" \
bash scripts/run_wayspots_marepo_s_gpu01.sh

log "Wayspots ACE-G25 dry run"
DRY_RUN=true \
SCENES="wayspots_bears wayspots_cubes" \
RUN_ROOT="${TMP_ROOT}/aceg25" \
bash scripts/run_wayspots_aceg25_baseline_gpu01.sh

log "Wayspots Stage2 FiLM dry run"
DRY_RUN=true \
SCENES_STR="wayspots_squarebench wayspots_bears" \
KS_STR="64" \
FEATURE_MODES_STR="glace zero" \
RUN_TAG="preflight_stage2_film" \
bash scripts/run_wayspots_glace_lmc_stage2_film_gpu01.sh

log "Indoor6 K sweep dry run"
DRY_RUN=true \
KS_STR="256" \
SCENES_STR="scene2a scene5" \
GPUS_STR="0 1" \
RUN_ROOT="${TMP_ROOT}/indoor6_k" \
bash scripts/run_indoor6_dino_k_sweep_gpu01.sh

log "ok: ${TMP_ROOT}"
