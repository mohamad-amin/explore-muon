#!/bin/bash
#SBATCH --job-name=so-newton-M4M
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_rtx_a6000:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=/share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/second_order_audit_20260926/job_newton_M4M_%j.log
#SBATCH --chdir=/share/data/dl-theory/amin/projects/explore_muon
set -uo pipefail
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/second_order_audit_20260926/newton_train.py /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/second_order_audit_20260926/newton_ref/M4M_from183_v2 /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_batch4m_20260927/M_b4M_lr0.014_s260925_l40s:183 --curvature-sequences 64 --validation-every 25 --checkpoint-every 25
