#!/bin/bash
#SBATCH --job-name=so-marg_PDf
#SBATCH --partition=dev-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/second_order_audit_20260926/job_marg_PDf_%j.log
#SBATCH --chdir=/share/data/dl-theory/amin/projects/explore_muon
set -euo pipefail
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
exec /share/data/dl-theory/amin/projects/explore_muon/.venv/bin/python -u logs/muon_spectra/second_order_audit_20260926/measure_marginals.py logs/muon_spectra/second_order_audit_20260926/marginals logs/muon_spectra/soaudit_traj_20260926/PD_a0.25_lr0.01_s260925_ada:final
