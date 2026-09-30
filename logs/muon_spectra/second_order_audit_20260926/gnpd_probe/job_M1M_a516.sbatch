#!/bin/bash
#SBATCH --job-name=soaudit-gnpd-M1M_a516
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/second_order_audit_20260926/gnpd_probe/M1M_a516.log
#SBATCH --chdir=/share/data/dl-theory/amin/projects/explore_muon
set -euo pipefail
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
exec .venv/bin/python logs/muon_spectra/second_order_audit_20260926/gnpd_probe.py logs/muon_spectra/second_order_audit_20260926/gnpd_probe/M1M_a516.json /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/second_order_audit_20260926/valley/M_lr0.007_s260925_l40s_a500:516 --inputs momentum_g1M,g4M
