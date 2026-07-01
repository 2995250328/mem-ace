#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <run_root> [summary_name]" >&2
  exit 2
fi

RUN_ROOT="$1"
SUMMARY_NAME="${2:-summary.tsv}"
ROOT_DIR="${ROOT_DIR:-/home/xwh/project/ace_depth}"
REPO_ROOT="${REPO_ROOT:-${ROOT_DIR}/ace_dinov2_lmc}"
PYTHON_BIN="${PYTHON_BIN:-/home/xwh/miniforge3/envs/mapanything/bin/python}"
OUT_TSV="${OUT_TSV:-${RUN_ROOT}/metricwise_best.tsv}"
OUT_MD="${OUT_MD:-${RUN_ROOT}/metricwise_best.md}"

mkdir -p "$(dirname "${OUT_TSV}")" "$(dirname "${OUT_MD}")"
cd "${ROOT_DIR}"

"${PYTHON_BIN}" "${REPO_ROOT}/.codex_skills/lmc-experiment-standard-workflow/scripts/best_metric_summary.py" \
  "${RUN_ROOT}" \
  --summary-name "${SUMMARY_NAME}" \
  --sources \
  --tsv "${OUT_TSV}" \
  > "${OUT_MD}"

echo "metricwise_best_tsv=${OUT_TSV}"
echo "metricwise_best_md=${OUT_MD}"
