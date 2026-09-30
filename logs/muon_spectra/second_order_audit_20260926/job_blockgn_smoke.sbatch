#!/bin/bash
#SBATCH --job-name=so-bgn-smoke
#SBATCH --partition=dev-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=/share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/second_order_audit_20260926/job_blockgn_smoke_%j.log
#SBATCH --chdir=/share/data/dl-theory/amin/projects/explore_muon
set -euo pipefail
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
exec .venv/bin/python -u logs/muon_spectra/second_order_audit_20260926/one_step_blockgn.py /home-nfs/mohamadamin/.tmp/claude-1457/-share-data-dl-theory-amin-projects-explore-muon/b1c5bfda-8bf5-4dd2-9cc8-4fa3608299f9/scratchpad/blockgn_smoke.json logs/muon_spectra/soaudit_traj_20260926/M_lr0.007_s260925_l40s:500 --input g1M --full-steps 8 --block-steps 3 --curvature-sequences 16 --eval-sequences 32 --partitions full,kind,matrix,gs
