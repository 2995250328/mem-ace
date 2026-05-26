#!/usr/bin/env bash
set -euo pipefail

# Download InLoc raw archives only.
# This script does not extract anything.
# Use the official CIIRC mirror; it exposes the raw archives directly and
# avoids the 403/HTML issues seen with the original host and Dropbox mirrors.

OUT_DIR="${OUT_DIR:-/data/xwh/dataset_staging/inloc/raw}"
mkdir -p "$OUT_DIR"

BASE_URL="https://data.ciirc.cvut.cz/public/projects/2020VisualLocalization/InLoc"

# The mirror exposes the actual archive names directly.
CUTOUTS_URL="$BASE_URL/cutouts.tar.gz"
SCANS_URL="$BASE_URL/scans.tar.gz"
QUERY_URL="$BASE_URL/queries/iphone7.tar.gz"

# The historical repository scripts also download alignment zips from the
# original project host. The CIIRC mirror lists only the big archive bundles
# directly, so if alignments are needed later, use the original repository
# scripts after the mirror sanity-check or fetch them from a different source.
# For now we keep this downloader focused on the archives the mirror exposes.

download() {
  local url="$1"
  local out="$2"
  if command -v curl >/dev/null 2>&1; then
    curl -L -A 'Mozilla/5.0' --retry 5 --retry-all-errors --connect-timeout 20 --max-time 0 --fail -o "$out" "$url"
  else
    wget --user-agent='Mozilla/5.0' --tries=5 --timeout=20 --content-disposition -O "$out" "$url"
  fi
  if [[ ! -s "$out" ]]; then
    echo "[InLoc] Download failed or produced empty file: $out" >&2
    exit 1
  fi
}

echo "[InLoc] Downloading raw archives into: $OUT_DIR"

download "$CUTOUTS_URL" "$OUT_DIR/cutouts.tar.gz"
download "$SCANS_URL" "$OUT_DIR/scans.tar.gz"
download "$QUERY_URL" "$OUT_DIR/iphone7.tar.gz"

echo "[InLoc] Done."
