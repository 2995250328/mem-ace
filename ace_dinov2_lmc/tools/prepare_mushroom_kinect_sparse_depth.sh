#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MUSHROOM_ROOT="${MUSHROOM_ROOT:-/data/xwh/MuSHRoom/kinect}"
ACE_ROOT="${ACE_ROOT:-/data/xwh/MuSHRoom_ace}"
WAI_ROOT="${WAI_ROOT:-/data/xwh/MuSHRoom_wai/kinect}"
if [ "$#" -gt 0 ]; then
    ROOMS=("$@")
else
    ROOMS=(coffee_room classroom vr_room)
fi

FRAME_STRIDE="${FRAME_STRIDE:-1}"
NEIGHBOR_RADIUS="${NEIGHBOR_RADIUS:-3}"
PIXEL_STEP="${PIXEL_STEP:-16}"
TEXTURE_PERCENTILE="${TEXTURE_PERCENTILE:-0.70}"
MIN_VIEWS="${MIN_VIEWS:-3}"
MIN_DEPTH="${MIN_DEPTH:-0.1}"
MAX_DEPTH="${MAX_DEPTH:-20.0}"
DEPTH_ABS_TOL="${DEPTH_ABS_TOL:-0.03}"
DEPTH_REL_TOL="${DEPTH_REL_TOL:-0.02}"
WRITE_OVERLAYS="${WRITE_OVERLAYS:-false}"
SKIP_WAI="${SKIP_WAI:-false}"

CMD=(
    python
    "$SCRIPT_DIR/extract_mushroom_kinect_sparse_depth.py"
    --mushroom-root "$MUSHROOM_ROOT"
    --ace-root "$ACE_ROOT"
    --wai-root "$WAI_ROOT"
    --rooms "${ROOMS[@]}"
    --frame-stride "$FRAME_STRIDE"
    --neighbor-radius "$NEIGHBOR_RADIUS"
    --pixel-step "$PIXEL_STEP"
    --texture-percentile "$TEXTURE_PERCENTILE"
    --min-views "$MIN_VIEWS"
    --min-depth "$MIN_DEPTH"
    --max-depth "$MAX_DEPTH"
    --depth-abs-tol "$DEPTH_ABS_TOL"
    --depth-rel-tol "$DEPTH_REL_TOL"
)

if [ "$WRITE_OVERLAYS" = "true" ]; then
    CMD+=(--write-overlays)
fi
if [ "$SKIP_WAI" = "true" ]; then
    CMD+=(--skip-wai)
fi

printf 'Running:'
printf ' %q' "${CMD[@]}"
printf '\n'
"${CMD[@]}"
