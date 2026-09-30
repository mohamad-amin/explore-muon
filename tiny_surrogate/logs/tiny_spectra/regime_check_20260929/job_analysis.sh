#!/bin/bash
#SBATCH --job-name=tiny-regime-probe
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_rtx_a4000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/share/data/dl-theory/amin/projects/explore_muon/tiny_surrogate/logs/tiny_spectra/regime_check_20260929/console/probe_%j.log
# Usage: sbatch --dependency=afterok:<train task> job_analysis.sh <run_id>
set -uo pipefail
export CC=/usr/bin/gcc CXX=/usr/bin/g++ OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
ROOT=/share/data/dl-theory/amin/projects/explore_muon/tiny_surrogate
echo "run $1 host $(hostname) gpu $CUDA_VISIBLE_DEVICES start $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$ROOT/run $ROOT/logs/tiny_spectra/regime_check_20260929/regime_probe.py all "$1"
EXIT=$?
echo "exit $EXIT end $(date)"
exit $EXIT
