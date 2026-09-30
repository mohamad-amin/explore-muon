#!/bin/bash
#SBATCH --job-name=so-gntrain-sweep
#SBATCH --partition=dev-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:55:00
#SBATCH --output=/share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/second_order_audit_20260926/job_gntrain_sweep_%j.log
#SBATCH --chdir=/share/data/dl-theory/amin/projects/explore_muon
set -uo pipefail
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
for pair in "0.002 0.0005" "0.001 0.00025" "0.0005 0.000125"; do
  set -- $pair
  echo "=== inner_lr $1 aux $2"
  .venv/bin/torchrun --standalone --nproc_per_node=2 /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/second_order_audit_20260926/gn_train.py /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/second_order_audit_20260926/gntrain_sweep/lr$1 /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_traj_20260926/M_lr0.007_s260925_l40s:100 --stop-after 8 --inner-sequences 64 --inner-lr $1 --inner-aux-lr $2 --validation-every 4 2>&1 | grep -v Warning
done
