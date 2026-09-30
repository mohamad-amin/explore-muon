#!/bin/bash
#SBATCH --job-name=so-eoslin-a
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/second_order_audit_20260926/job_eoslin_a_%j.log
#SBATCH --chdir=/share/data/dl-theory/amin/projects/explore_muon
set -euo pipefail
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
exec .venv/bin/python -u /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/second_order_audit_20260926/eos_linearized.py /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/second_order_audit_20260926/eos_linearized_a.json /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_traj_20260926/M_lr0.007_s260925_l40s:500:muon /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_traj_20260926/PD_a0.25_lr0.01_s260925_ada:500:pd:0.25 /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_batch4m_20260927/M_b4M_lr0.014_s260925_l40s:183:muon /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_batch4m_20260927/PD_b4M_lr0.02_s260925_ada:183:pd:0.25 /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_traj_20260926/M_lr0.007_s260925_l40s:900:muon --steps 40
