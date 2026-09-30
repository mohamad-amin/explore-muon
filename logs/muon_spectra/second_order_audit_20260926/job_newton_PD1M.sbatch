#!/bin/bash
#SBATCH --job-name=so-newton-PD1M
#SBATCH --partition=dev-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:59:00
#SBATCH --output=/share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/second_order_audit_20260926/job_newton_PD1M_%j.log
#SBATCH --chdir=/share/data/dl-theory/amin/projects/explore_muon
set -uo pipefail
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
# resumes from newton_ref/PD1M_from500/checkpoint.pt when present (chained 1-hour dev jobs)
timeout 3420 .venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/second_order_audit_20260926/newton_train.py /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/second_order_audit_20260926/newton_ref/PD1M_from500_v2 /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_traj_20260926/PD_a0.25_lr0.01_s260925_ada:500 --validation-every 25 --checkpoint-every 25
