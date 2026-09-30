#!/bin/bash
#SBATCH --job-name=so-gram-saved
#SBATCH --partition=dev-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/second_order_audit_20260926/job_gram_saved_%j.log
#SBATCH --chdir=/share/data/dl-theory/amin/projects/explore_muon
set -euo pipefail
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
A=logs/muon_spectra/second_order_audit_20260926
T=logs/muon_spectra/soaudit_traj_20260926
.venv/bin/python -u $A/gram_saved.py $A/gn/M_lr0.007_s260925_l40s_step000500_directions.pt $T/M_lr0.007_s260925_l40s/scientific/kept/step000500.pt --inputs g4M
.venv/bin/python -u $A/gram_saved.py $A/gn/PD_a0.25_lr0.01_s260925_ada_step000500_directions.pt $T/PD_a0.25_lr0.01_s260925_ada/scientific/kept/step000500.pt --inputs g4M
