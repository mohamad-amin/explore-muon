#!/bin/bash
#SBATCH --job-name=muon-headcenter16m-PDhwc0.5_a0.5_b16M_lr0.028_mom0.9_s260925_l40s
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_l40s:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_headcenter16m_20260928/PDhwc0.5_a0.5_b16M_lr0.028_mom0.9_s260925_l40s/slurm-%j.log
#SBATCH --error=/share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_headcenter16m_20260928/PDhwc0.5_a0.5_b16M_lr0.028_mom0.9_s260925_l40s/slurm-%j.log
#SBATCH --chdir=/share/data/dl-theory/amin/projects/explore_muon
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 PYTHONUNBUFFERED=1
exec /share/data/dl-theory/amin/projects/explore_muon/.venv/bin/python -u /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_headcenter16m_20260928/frozen/adamw_spectra/run_arm.py --root /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_headcenter16m_20260928/PDhwc0.5_a0.5_b16M_lr0.028_mom0.9_s260925_l40s --world-size 4
