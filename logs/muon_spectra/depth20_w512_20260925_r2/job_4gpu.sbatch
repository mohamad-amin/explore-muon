#!/bin/bash
#SBATCH --job-name=muon-d20-w512
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --constraint=96g
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/share/data/dl-theory/amin/projects/last_layer/logs/muon_spectra/depth20_w512_20260925_r2/slurm-%j.log
#SBATCH --error=/share/data/dl-theory/amin/projects/last_layer/logs/muon_spectra/depth20_w512_20260925_r2/slurm-%j.log
#SBATCH --chdir=/share/data/dl-theory/amin/projects/last_layer
set -euo pipefail
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=2
export TORCHINDUCTOR_COMPILE_THREADS=1
export PYTHONUNBUFFERED=1
exec /share/data/dl-theory/amin/projects/last_layer/.venv/bin/python -u /share/data/dl-theory/amin/projects/last_layer/logs/muon_spectra/depth20_w512_20260925_r2/run_muon.py --root /share/data/dl-theory/amin/projects/last_layer/logs/muon_spectra/depth20_w512_20260925_r2 --world-size 4
