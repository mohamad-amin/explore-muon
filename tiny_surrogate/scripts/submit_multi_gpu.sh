#!/usr/bin/env bash
set -euo pipefail
STUDY_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if (( $# < 3 )); then
  echo 'Usage: submit_multi_gpu.sh {1|2|4} CONFIG OUTPUT_DIR [--time HH:MM:SS] [--node NODE] [--dry-run]' >&2
  exit 2
fi
GPU_COUNT=$1
CONFIG_PATH=$(realpath -- "$2")
OUTPUT_PATH=$(realpath -m -- "$3")
shift 3
exec "$STUDY_ROOT/run" -m research.tiny_spectra.multi_gpu submit --gpus "$GPU_COUNT" --config "$CONFIG_PATH" --out "$OUTPUT_PATH" "$@"
