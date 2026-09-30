#!/bin/bash
#SBATCH --job-name=so-eoslin-smoke
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/second_order_audit_20260926/job_eoslin_smoke_%j.log
#SBATCH --chdir=/share/data/dl-theory/amin/projects/explore_muon
set -euo pipefail
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
exec .venv/bin/python -u /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/second_order_audit_20260926/eos_linearized.py /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/second_order_audit_20260926/eos_linearized_smoke.json \
  /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_traj_20260926/M_lr0.007_s260925_l40s:500:muon /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_traj_20260926/PD_a0.25_lr0.01_s260925_ada:500:pd:0.25 --steps 8 --curvature-sequences 16
