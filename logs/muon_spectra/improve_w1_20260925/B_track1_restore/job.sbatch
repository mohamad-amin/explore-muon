#!/bin/bash
#SBATCH --job-name=muon-imp-B_track1_restore
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --constraint=48g
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/improve_w1_20260925/B_track1_restore/slurm-%j.log
#SBATCH --error=/share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/improve_w1_20260925/B_track1_restore/slurm-%j.log
#SBATCH --chdir=/share/data/dl-theory/amin/projects/explore_muon
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 PYTHONUNBUFFERED=1
exec /share/data/dl-theory/amin/projects/explore_muon/.venv/bin/python -u /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/improve_w1_20260925/frozen/adamw_spectra/run_arm.py --root /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/improve_w1_20260925/B_track1_restore --world-size 4
