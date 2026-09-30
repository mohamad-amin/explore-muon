#!/bin/bash
#SBATCH --job-name=so-eos
#SBATCH --partition=dev-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/second_order_audit_20260926/job_eos_%j.log
#SBATCH --chdir=/share/data/dl-theory/amin/projects/explore_muon
set -euo pipefail
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
exec .venv/bin/python -u logs/muon_spectra/second_order_audit_20260926/eos_invariant.py logs/muon_spectra/second_order_audit_20260926/eos_invariant.json /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_traj_20260926/M_lr0.007_s260925_l40s:500:0 /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_traj_20260926/PD_a0.25_lr0.01_s260925_ada:500:0.25 /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_batch4m_20260927/M_b4M_lr0.014_s260925_l40s:183:0 /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_batch4m_20260927/PD_b4M_lr0.02_s260925_ada:183:0.25 /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_alpha4m_20260927/PD_a0.5_b4M_lr0.02_s260925_ada:183:0.5 /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_traj_20260926/M_lr0.007_s260925_l40s:900:0 /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_traj_20260926/PD_a0.25_lr0.01_s260925_ada:900:0.25 /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_traj_20260926/M_lr0.007_s260925_l40s:200:0 /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_traj_20260926/PD_a0.25_lr0.01_s260925_ada:200:0.25
