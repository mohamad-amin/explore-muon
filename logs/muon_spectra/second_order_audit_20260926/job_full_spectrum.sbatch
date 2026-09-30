#!/bin/bash
#SBATCH --job-name=so-fullspec
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --constraint=24g
#SBATCH --exclude=g10,g12,g13
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=01:30:00
#SBATCH --output=/share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/second_order_audit_20260926/figures_momentum/full_spectrum.slurm_%j.log
#SBATCH --chdir=/share/data/dl-theory/amin/projects/explore_muon
# Where the top of the full-parameter GN spectrum lives (MUON_CASE 2026-09-30 16:0x CDT, after arXiv 2607.21716).
set -uo pipefail
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
A=logs/muon_spectra/second_order_audit_20260926
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 $A/full_spectrum_probe.py $A/figures_momentum/full_spectrum.json   /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_mom16m_20260928/PD_a0.5_b16M_lr0.028_mom0.8_s260925_l40s:9   /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_mom16m_20260928/PD_a0.5_b16M_lr0.028_mom0.8_s260925_l40s:46   /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_mom16m_20260928/PD_a0.5_b16M_lr0.028_mom0.8_s260925_l40s:83   /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_prefilter16m_20260928/SPD_a0.5_b16M_lr0.028_mom0.8_s260925_ada:46   /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_mom16m_20260928/M_b16M_lr0.02_mom0.8_s260925_l40s:46   /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_mom4m4_20260927/PD_a0.5_b4M_lr0.02_mom0.9_s260925_ada:183   /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_strength4m_20260927/SPD_a0.5_b4M_lr0.02_mom0.9_s260925_l40s:183   /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_traj_20260926/SPD_a0.25_lr0.01_s260925_ada:900
