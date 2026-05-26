#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MUSHROOM_ROOT="${MUSHROOM_ROOT:-/data/xwh/MuSHRoom/kinect}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/xwh/MuSHRoom_wai/kinect}"
COPY_FILES="${COPY_FILES:-false}"
SKIP_INCOMPLETE="${SKIP_INCOMPLETE:-true}"
if [ "$#" -gt 0 ]; then
    ROOMS=("$@")
else
    ROOMS=(coffee_room classroom vr_room)
fi

CMD=(
    python
    "$SCRIPT_DIR/convert_mushroom_kinect_to_wai.py"
    --mushroom-root "$MUSHROOM_ROOT"
    --output-root "$OUTPUT_ROOT"
    --rooms "${ROOMS[@]}"
)

if [ "$COPY_FILES" = "true" ]; then
    CMD+=(--copy-files)
fi
if [ "$SKIP_INCOMPLETE" = "true" ]; then
    CMD+=(--skip-incomplete)
fi

printf 'Running:'
printf ' %q' "${CMD[@]}"
printf '\n'
"${CMD[@]}"
