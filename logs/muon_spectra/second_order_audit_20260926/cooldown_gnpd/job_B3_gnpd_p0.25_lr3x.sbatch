#!/bin/bash
#SBATCH --job-name=soaudit-cool-B3_gnpd_lr3x
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_rtx_a6000:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/second_order_audit_20260926/cooldown_gnpd/B3_gnpd_p0.25_lr3x.log
#SBATCH --chdir=/share/data/dl-theory/amin/projects/explore_muon
set -euo pipefail
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
exec .venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/second_order_audit_20260926/newton_train.py /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/second_order_audit_20260926/cooldown_gnpd/B3_gnpd_p0.25_lr3x /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_traj_20260926/M_lr0.007_s260925_l40s:1300 --normalize 0.001 --gnpd 0.25 --cg 64 --lr-scale 3 --stop-after 169 --momentum 0.95 --validation-every 25 --checkpoint-every 25
