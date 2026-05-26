#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

IPHONE_ROOT="${IPHONE_ROOT:-/data/xwh/MuSHRoom/iphone}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/xwh/MuSHRoom_ace_iphone}"
SPLIT="${SPLIT:-short_capture}"
TRANSFORM_FILE="${TRANSFORM_FILE:-transformations_colmap.json}"
COPY_FILES="${COPY_FILES:-false}"
REQUIRE_DEPTH="${REQUIRE_DEPTH:-false}"
SKIP_INCOMPLETE="${SKIP_INCOMPLETE:-true}"
if [ "$#" -gt 0 ]; then
    ROOMS=("$@")
else
    ROOMS=(coffee_room classroom vr_room)
fi

CMD=(
    python
    "$SCRIPT_DIR/convert_mushroom_iphone_to_ace.py"
    --iphone-root "$IPHONE_ROOT"
    --output-root "$OUTPUT_ROOT"
    --split "$SPLIT"
    --transform-file "$TRANSFORM_FILE"
    --rooms "${ROOMS[@]}"
)

if [ "$COPY_FILES" = "true" ]; then
    CMD+=(--copy-files)
fi
if [ "$REQUIRE_DEPTH" = "true" ]; then
    CMD+=(--require-depth)
fi
if [ "$SKIP_INCOMPLETE" = "true" ]; then
    CMD+=(--skip-incomplete)
fi

printf 'Running:'
printf ' %q' "${CMD[@]}"
printf '\n'
"${CMD[@]}"
