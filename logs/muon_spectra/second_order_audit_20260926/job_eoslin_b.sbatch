#!/bin/bash
#SBATCH --job-name=so-eoslin-b
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/second_order_audit_20260926/job_eoslin_b_%j.log
#SBATCH --chdir=/share/data/dl-theory/amin/projects/explore_muon
set -euo pipefail
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
exec .venv/bin/python -u /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/second_order_audit_20260926/eos_linearized.py /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/second_order_audit_20260926/eos_linearized_b.json /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_mom4m_20260927/M_b4M_lr0.014_mom0.9_s260925_l40s:183:muon /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_mom4m_20260927/M_b4M_lr0.014_mom0.81_s260925_l40s:183:muon /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_mom4m_20260927/M_b4M_lr0.014_nesterov_s260925_l40s:183:muon /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_mom4m3_20260927/M_b4M_lr0.02_mom0.9_s260925_l40s:183:muon /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_mom4m3_20260927/PD_b4M_lr0.02_mom0.9_s260925_ada:183:pd:0.25 --steps 40
