#!/usr/bin/env bash
set -euo pipefail
exec "$(dirname -- "${BASH_SOURCE[0]}")/submit_multi_gpu.sh" 2 "$@"
