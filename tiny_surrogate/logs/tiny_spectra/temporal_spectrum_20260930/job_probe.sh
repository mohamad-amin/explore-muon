#!/bin/bash
#SBATCH --job-name=tiny-temporal
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_rtx_a4000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/share/data/dl-theory/amin/projects/explore_muon/tiny_surrogate/logs/tiny_spectra/temporal_spectrum_20260930/console/%x_%j.log
set -uo pipefail
export CC=/usr/bin/gcc CXX=/usr/bin/g++ OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
ROOT=/share/data/dl-theory/amin/projects/explore_muon/tiny_surrogate
echo "run $1 host $(hostname) start $(date)"
$ROOT/run $ROOT/logs/tiny_spectra/temporal_spectrum_20260930/temporal_probe.py "$1"
echo "exit $? end $(date)"
