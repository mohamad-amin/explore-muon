#!/bin/bash
#SBATCH --job-name=so-staged-big
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_rtx_a6000:4
#SBATCH --exclude=g10,g12,g13
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --output=/share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/second_order_audit_20260926/staged_branch/staged_big_256.slurm_%j.log
#SBATCH --chdir=/share/data/dl-theory/amin/projects/explore_muon
# Low-noise layer-staged PD (MUON_CASE 2026-09-29 08:0x CDT): curvature and PD root from 256 sequences per rank (8x the
# earlier staged runs), 16M @9 to 92, sharpness every 10 steps; staged arm.
set -uo pipefail
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
A=logs/muon_spectra/second_order_audit_20260926
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 $A/newton_train.py $A/staged_branch/staged_big_256 /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_batch16m_20260927/PD_a0.5_b16M_lr0.028_mom0.9_s260925_l40s:9 --normalize 0 --momentum 0.9 --clip 1.0 --staged-pd 0.5  --curvature-sequences 256 --validation-every 10 --sharpness-every 10 --checkpoint-every 1000 --keep 46
