#!/bin/bash
set -uo pipefail
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 RAYON_NUM_THREADS=8 PYTHONUNBUFFERED=1
# Outer timeout also interrupts a long native tokenizer call.
timeout --signal=TERM --kill-after=30s 7200 /share/data/dl-theory/amin/projects/explore_muon/tiny_surrogate/run /share/data/dl-theory/amin/projects/explore_muon/tiny_surrogate/logs/tiny_spectra/fineweb_d_preparation_20260928/driver.py
PREPARATION_EXIT_CODE=$?
/share/data/dl-theory/amin/projects/explore_muon/tiny_surrogate/run /share/data/dl-theory/amin/projects/explore_muon/tiny_surrogate/logs/tiny_spectra/fineweb_d_preparation_20260928/record_exit.py "$PREPARATION_EXIT_CODE"
exit "$PREPARATION_EXIT_CODE"
