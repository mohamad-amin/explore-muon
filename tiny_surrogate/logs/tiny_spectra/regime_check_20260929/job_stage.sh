#!/bin/bash
#SBATCH --job-name=tiny-regime-stage
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_rtx_a4000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/share/data/dl-theory/amin/projects/explore_muon/tiny_surrogate/logs/tiny_spectra/regime_check_20260929/console/stage_%j.log
# Usage: sbatch job_stage.sh <regime_probe.py arguments...>
set -uo pipefail
export CC=/usr/bin/gcc CXX=/usr/bin/g++ OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
ROOT=/share/data/dl-theory/amin/projects/explore_muon/tiny_surrogate
echo "args $* host $(hostname) gpu $CUDA_VISIBLE_DEVICES start $(date)"
$ROOT/run $ROOT/logs/tiny_spectra/regime_check_20260929/regime_probe.py "$@"
EXIT=$?
echo "exit $EXIT end $(date)"
exit $EXIT
