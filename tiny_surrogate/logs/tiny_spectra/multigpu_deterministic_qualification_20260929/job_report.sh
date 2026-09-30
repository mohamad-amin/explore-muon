#!/bin/bash
set -euo pipefail
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
exec /share/data/dl-theory/amin/projects/explore_muon/tiny_surrogate/run /share/data/dl-theory/amin/projects/explore_muon/tiny_surrogate/logs/tiny_spectra/multigpu_deterministic_qualification_20260929/parallel_report.py /share/data/dl-theory/amin/projects/explore_muon/tiny_surrogate/logs/tiny_spectra/multigpu_deterministic_qualification_20260929
