#!/bin/bash
#SBATCH --job-name=adamw-d12-w512
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --constraint=48g
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/share/data/dl-theory/amin/projects/last_layer/logs/adamw_spectra/depth12_w512_20260924/slurm-%j.log
#SBATCH --error=/share/data/dl-theory/amin/projects/last_layer/logs/adamw_spectra/depth12_w512_20260924/slurm-%j.log
#SBATCH --chdir=/share/data/dl-theory/amin/projects/last_layer
set -euo pipefail
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=2
export TORCHINDUCTOR_COMPILE_THREADS=1
export PYTHONUNBUFFERED=1
exec /share/data/dl-theory/amin/projects/last_layer/.venv/bin/python -u /share/data/dl-theory/amin/projects/last_layer/logs/adamw_spectra/depth12_w512_20260924/run_depth.py --root /share/data/dl-theory/amin/projects/last_layer/logs/adamw_spectra/depth12_w512_20260924
