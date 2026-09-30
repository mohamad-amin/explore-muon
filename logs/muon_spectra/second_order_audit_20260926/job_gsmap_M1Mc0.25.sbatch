#!/bin/bash
#SBATCH --job-name=so-gsmap-M1Mc0.25
#SBATCH --partition=dev-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/second_order_audit_20260926/job_gsmap_M1Mc0.25_%j.log
#SBATCH --chdir=/share/data/dl-theory/amin/projects/explore_muon
set -euo pipefail
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
exec .venv/bin/python -u logs/muon_spectra/second_order_audit_20260926/one_step_blockgn.py logs/muon_spectra/second_order_audit_20260926/blockgn/gsmap_M1Mc0.25.json logs/muon_spectra/soaudit_traj_20260926/M_lr0.007_s260925_l40s:500 --input momentum1M --momentum-c 0.25 --partitions gsmap --curvature-sequences 128 --block-steps 16 --gs-damping 0.001
