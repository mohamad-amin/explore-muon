#!/bin/bash
set -euo pipefail
export CC=/usr/bin/gcc CXX=/usr/bin/g++ OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONUNBUFFERED=1
exec /share/data/dl-theory/amin/projects/explore_muon/tiny_surrogate/run /share/data/dl-theory/amin/projects/explore_muon/tiny_surrogate/logs/tiny_spectra/fineweb_d_smoke_20260928/frozen/research/tiny_spectra/cohort.py task /share/data/dl-theory/amin/projects/explore_muon/tiny_surrogate/logs/tiny_spectra/fineweb_d_smoke_20260928 --index "$SLURM_ARRAY_TASK_ID"
