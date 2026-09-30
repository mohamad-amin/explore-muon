#!/bin/bash
#SBATCH --job-name=so-gn-smoke
#SBATCH --partition=gpu
#SBATCH --nodelist=g4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/second_order_audit_20260926/job_gn_smoke_%j.log
#SBATCH --chdir=/share/data/dl-theory/amin/projects/explore_muon
set -euo pipefail
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
exec /share/data/dl-theory/amin/projects/explore_muon/.venv/bin/python -u logs/muon_spectra/second_order_audit_20260926/one_step_gn.py logs/muon_spectra/second_order_audit_20260926/gn_smoke /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_traj_20260926/PD_a0.25_lr0.01_s260925_ada:500 --inputs g1M,momentum --krylov 8 --curvature-sequences 16 --eval-sequences 16 --micro 4 --gram --gram-sequences 4
