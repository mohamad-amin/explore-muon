#!/bin/bash
#SBATCH --job-name=so-coupling
#SBATCH --partition=dev-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/second_order_audit_20260926/job_coupling_%j.log
#SBATCH --chdir=/share/data/dl-theory/amin/projects/explore_muon
set -euo pipefail
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
/share/data/dl-theory/amin/projects/explore_muon/.venv/bin/python -u logs/muon_spectra/second_order_audit_20260926/measure_step_coupling.py logs/muon_spectra/second_order_audit_20260926/coupling /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_traj_20260926/M_lr0.007_s260925_l40s:500 /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_traj_20260926/PD_a0.25_lr0.01_s260925_ada:500
