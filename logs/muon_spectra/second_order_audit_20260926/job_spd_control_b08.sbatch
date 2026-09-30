#!/bin/bash
#SBATCH --job-name=so-spd-ctrl-b08
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --constraint=48g
#SBATCH --exclude=g10,g12,g13
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=02:30:00
#SBATCH --output=/share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/second_order_audit_20260926/band/spd_control_b08.slurm_%j.log
#SBATCH --chdir=/share/data/dl-theory/amin/projects/explore_muon
# S∘PD harness control at beta 0.8 (--soap --stage-off), from the real S∘PD beta 0.8 run @9 (MUON_CASE 14:44 CDT).
set -uo pipefail
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
A=logs/muon_spectra/second_order_audit_20260926
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 $A/newton_train.py $A/band/spd_control_b08 /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_prefilter16m_20260928/SPD_a0.5_b16M_lr0.028_mom0.8_s260925_ada:9 --normalize 0 --momentum 0.8 --clip 1.0 --staged-pd 0.5 --soap --stage-off --validation-every 10 --sharpness-every 10 --checkpoint-every 1000
