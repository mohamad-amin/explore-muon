#!/bin/bash
#SBATCH --job-name=so-blockgn-M1M
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=05:00:00
#SBATCH --output=/share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/second_order_audit_20260926/job_blockgn_M1M_%j.log
#SBATCH --chdir=/share/data/dl-theory/amin/projects/explore_muon
set -euo pipefail
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
exec .venv/bin/python -u logs/muon_spectra/second_order_audit_20260926/one_step_blockgn.py logs/muon_spectra/second_order_audit_20260926/blockgn/M1M_g4M.json logs/muon_spectra/soaudit_traj_20260926/M_lr0.007_s260925_l40s:500 --input g4M
