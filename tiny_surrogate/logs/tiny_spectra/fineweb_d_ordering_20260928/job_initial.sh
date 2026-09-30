#!/bin/bash
set -uo pipefail
export CC=/usr/bin/gcc CXX=/usr/bin/g++ OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1
timeout --signal=TERM --kill-after=15s 3540 /share/data/dl-theory/amin/projects/explore_muon/tiny_surrogate/run /share/data/dl-theory/amin/projects/explore_muon/tiny_surrogate/logs/tiny_spectra/fineweb_d_ordering_20260928/ordering_driver.py worker /share/data/dl-theory/amin/projects/explore_muon/tiny_surrogate/logs/tiny_spectra/fineweb_d_ordering_20260928 --stage initial --index "$SLURM_ARRAY_TASK_ID"
ORDERING_EXIT=$?
/share/data/dl-theory/amin/projects/explore_muon/tiny_surrogate/run -c 'import json,sys,time; from pathlib import Path; p=Path(sys.argv[1]); p.write_text(json.dumps(dict(exit_code=int(sys.argv[2]),finished_unix=time.time()),indent=2)+'"'"'\n'"'"')' /share/data/dl-theory/amin/projects/explore_muon/tiny_surrogate/logs/tiny_spectra/fineweb_d_ordering_20260928/workers/initial_exit_"$SLURM_ARRAY_TASK_ID".json "$ORDERING_EXIT"
exit "$ORDERING_EXIT"
