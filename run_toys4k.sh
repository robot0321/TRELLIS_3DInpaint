#!/bin/bash
set -euo pipefail

GPU="${1:-0}"
TAG="${2:-eval_toys4k}"
LIMIT="${3:-}"
OFFSET="${4:-0}"
shift $(( $# >= 4 ? 4 : $# )) || true
EXTRA_ARGS=("$@")

CMD=(python example_toys4k.py --tag "$TAG" --offset "$OFFSET")
if [[ -n "$LIMIT" ]]; then
  CMD+=(--limit "$LIMIT")
fi
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

echo "[run_toys4k] GPU=$GPU TAG=$TAG OFFSET=$OFFSET LIMIT=${LIMIT:-all}"
echo "[run_toys4k] CMD: ${CMD[*]}"

CUDA_VISIBLE_DEVICES="$GPU" "${CMD[@]}"
