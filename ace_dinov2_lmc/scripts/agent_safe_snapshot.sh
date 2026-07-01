#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
LABEL="${2:-agent}"
RUN_ROOT="${3:-}"
ROOT_DIR="${ROOT_DIR:-/home/xwh/project/ace_depth}"
REPO_ROOT="${REPO_ROOT:-${ROOT_DIR}/ace_dinov2_lmc}"
SNAPSHOT_BASE="${SNAPSHOT_BASE:-/data/xwh/.tmp/ace_dinov2_lmc_agent_snapshots}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
SAFE_LABEL="$(printf '%s' "${LABEL}" | tr -c 'A-Za-z0-9_.-' '_')"

usage() {
  cat >&2 <<EOF
Usage:
  $0 pre_edit <label>
  $0 run_provenance <label> <run_root>
EOF
}

if [[ -z "${MODE}" ]]; then
  usage
  exit 2
fi

cd "${ROOT_DIR}"
mkdir -p "${SNAPSHOT_BASE}"

write_state() {
  local out_dir="$1"
  mkdir -p "${out_dir}"
  git -C "${ROOT_DIR}" rev-parse HEAD > "${out_dir}/head.txt" 2>/dev/null || true
  git -C "${ROOT_DIR}" status --short > "${out_dir}/status_short.txt" 2>/dev/null || true
  git -C "${ROOT_DIR}" diff --binary > "${out_dir}/tracked_diff.patch" 2>/dev/null || true
  git -C "${ROOT_DIR}" ls-files --others --exclude-standard > "${out_dir}/untracked_files.txt" 2>/dev/null || true
  {
    echo "timestamp=${STAMP}"
    echo "label=${LABEL}"
    echo "root_dir=${ROOT_DIR}"
    echo "repo_root=${REPO_ROOT}"
    echo "mode=${MODE}"
    echo "run_root=${RUN_ROOT}"
  } > "${out_dir}/snapshot_meta.txt"
}

case "${MODE}" in
  pre_edit)
    OUT_DIR="${SNAPSHOT_BASE}/${STAMP}_${SAFE_LABEL}_pre_edit"
    write_state "${OUT_DIR}"
    echo "agent_snapshot=${OUT_DIR}"
    ;;
  run_provenance)
    if [[ -z "${RUN_ROOT}" ]]; then
      usage
      exit 2
    fi
    OUT_DIR="${RUN_ROOT}/agent_provenance"
    write_state "${OUT_DIR}"
    echo "agent_provenance=${OUT_DIR}"
    ;;
  *)
    usage
    exit 2
    ;;
esac
