#!/bin/bash
#SBATCH --job-name=so-ctl2x-ema
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --constraint=24g
#SBATCH --exclude=g10,g12,g13
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --output=/share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/second_order_audit_20260926/band/spd_control_c8ema_2x_from0.slurm_%j.log
#SBATCH --chdir=/share/data/dl-theory/amin/projects/explore_muon
# Matched 2x control for the cheap version (design review (a) at 2x): S∘PD beta 0.8 from initialization, no coordination,
# R from the same 8-sequence EMA (0.9) of C as spd_sublayer_c8ema_2x_from0 (MUON_CASE 2026-09-30 05:02 caveat).
set -uo pipefail
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
A=logs/muon_spectra/second_order_audit_20260926
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 $A/newton_train.py $A/band/spd_control_c8ema_2x_from0 /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/second_order_audit_20260926/band/init16m2x_spd_b08:0 --normalize 0 --momentum 0.8 --clip 1.0 --staged-pd 0.5 --soap --stage-off --curvature-sequences 8 --stage-c-ema 0.9 --validation-every 20 --sharpness-every 20 --checkpoint-every 1000
