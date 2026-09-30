#!/bin/bash
#SBATCH --job-name=so-eos-mom2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/second_order_audit_20260926/job_eos_mom2_%j.log
#SBATCH --chdir=/share/data/dl-theory/amin/projects/explore_muon
set -euo pipefail
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
exec .venv/bin/python -u /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/second_order_audit_20260926/eos_invariant.py /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/second_order_audit_20260926/eos_invariant_mom2.json \
  /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_mom4m3_20260927/PD_b4M_lr0.02_mom0.9_s260925_ada:183:0.25 \
  /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_mom4m2_20260927/PD_b4M_lr0.02_mom0.81_s260925_ada:183:0.25 \
  /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_mom4m_20260927/M_b4M_lr0.014_mom0.81_s260925_l40s:330:0 \
  /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_mom4m_20260927/M_b4M_lr0.014_mom0.9_s260925_l40s:330:0
