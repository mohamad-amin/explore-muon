#!/usr/bin/env bash
set -euo pipefail
STUDY_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if (( $# != 3 )); then
  echo 'Usage on an allocated GPU node: run_multi_gpu.sh {1|2|4} CONFIG OUTPUT_DIR' >&2
  exit 2
fi
CONFIG_PATH=$(realpath -- "$2")
OUTPUT_PATH=$(realpath -m -- "$3")
exec "$STUDY_ROOT/run" -m research.tiny_spectra.multi_gpu run --gpus "$1" --config "$CONFIG_PATH" --out "$OUTPUT_PATH"
