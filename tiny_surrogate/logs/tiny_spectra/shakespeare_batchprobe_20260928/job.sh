#!/bin/bash
set -euo pipefail
export CC=/usr/bin/gcc CXX=/usr/bin/g++ OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONUNBUFFERED=1
exec /share/data/dl-theory/amin/projects/explore_muon/.venv/bin/python /share/data/dl-theory/amin/projects/explore_muon/logs/tiny_spectra/shakespeare_batchprobe_20260928/frozen/research/tiny_spectra/cohort.py task /share/data/dl-theory/amin/projects/explore_muon/logs/tiny_spectra/shakespeare_batchprobe_20260928 --index "$SLURM_ARRAY_TASK_ID"
