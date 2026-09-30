#!/bin/bash
#SBATCH --job-name=so-marg_SPD
#SBATCH --partition=dev-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/second_order_audit_20260926/job_marg_SPD_%j.log
#SBATCH --chdir=/share/data/dl-theory/amin/projects/explore_muon
set -euo pipefail
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
exec /share/data/dl-theory/amin/projects/explore_muon/.venv/bin/python -u logs/muon_spectra/second_order_audit_20260926/measure_marginals.py logs/muon_spectra/second_order_audit_20260926/marginals logs/muon_spectra/soaudit_traj_20260926/SPD_a0.25_lr0.01_s260925_ada:10 logs/muon_spectra/soaudit_traj_20260926/SPD_a0.25_lr0.01_s260925_ada:50 logs/muon_spectra/soaudit_traj_20260926/SPD_a0.25_lr0.01_s260925_ada:100 logs/muon_spectra/soaudit_traj_20260926/SPD_a0.25_lr0.01_s260925_ada:200 logs/muon_spectra/soaudit_traj_20260926/SPD_a0.25_lr0.01_s260925_ada:500 logs/muon_spectra/soaudit_traj_20260926/SPD_a0.25_lr0.01_s260925_ada:900 logs/muon_spectra/soaudit_traj_20260926/SPD_a0.25_lr0.01_s260925_ada:1300 logs/muon_spectra/soaudit_traj_20260926/SPD_a0.25_lr0.01_s260925_ada:final 
