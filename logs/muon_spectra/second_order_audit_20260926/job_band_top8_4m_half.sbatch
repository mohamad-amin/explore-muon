#!/bin/bash
#SBATCH --job-name=so-top8-4m-half
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --constraint=48g
#SBATCH --exclude=g10,g12,g13
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=02:30:00
#SBATCH --output=/share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/second_order_audit_20260926/band/band_top8_4m_half.slurm_%j.log
#SBATCH --chdir=/share/data/dl-theory/amin/projects/explore_muon
# Dose-matched 4M: the 8-direction band at half strength (MUON_CASE 2026-09-29 14:4x CDT).
set -uo pipefail
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
A=logs/muon_spectra/second_order_audit_20260926
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 $A/newton_train.py $A/band/band_top8_4m_half /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_mom4m4_20260927/PD_a0.5_b4M_lr0.02_mom0.9_s260925_ada:37 --normalize 0 --momentum 0.9 --clip 1.0 --staged-pd 0.5 --stage-band top8 --stage-scale 0.5 --validation-every 20 --sharpness-every 40 --checkpoint-every 1000
