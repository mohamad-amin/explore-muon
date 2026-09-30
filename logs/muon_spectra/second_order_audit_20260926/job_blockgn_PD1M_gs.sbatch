#!/bin/bash
#SBATCH --job-name=so-bgn-PD1M-gs
#SBATCH --partition=dev-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/second_order_audit_20260926/job_blockgn_PD1M_gs_%j.log
#SBATCH --chdir=/share/data/dl-theory/amin/projects/explore_muon
set -euo pipefail
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
exec .venv/bin/python -u logs/muon_spectra/second_order_audit_20260926/one_step_blockgn.py logs/muon_spectra/second_order_audit_20260926/blockgn/PD1M_g4M_gs.json logs/muon_spectra/soaudit_traj_20260926/PD_a0.25_lr0.01_s260925_ada:500 --input g4M --partitions gs --curvature-sequences 128 --block-steps 16 --gs-damping 0.001
