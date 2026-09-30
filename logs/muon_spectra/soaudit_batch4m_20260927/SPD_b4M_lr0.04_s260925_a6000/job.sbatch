#!/bin/bash
#SBATCH --job-name=muon-batch4m-SPD_b4M_lr0.04_s260925_a6000
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_rtx_a6000:4
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_batch4m_20260927/SPD_b4M_lr0.04_s260925_a6000/slurm-%j.log
#SBATCH --error=/share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_batch4m_20260927/SPD_b4M_lr0.04_s260925_a6000/slurm-%j.log
#SBATCH --chdir=/share/data/dl-theory/amin/projects/explore_muon
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 PYTHONUNBUFFERED=1
exec /share/data/dl-theory/amin/projects/explore_muon/.venv/bin/python -u /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_batch4m_20260927/frozen/adamw_spectra/run_arm.py --root /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_batch4m_20260927/SPD_b4M_lr0.04_s260925_a6000 --world-size 4
