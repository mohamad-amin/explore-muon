#!/bin/bash
#SBATCH --job-name=so-gntrain-pair
#SBATCH --partition=dev-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:58:00
#SBATCH --output=/share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/second_order_audit_20260926/job_gntrain_pair_%j.log
#SBATCH --chdir=/share/data/dl-theory/amin/projects/explore_muon
set -uo pipefail
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
for arm in gn control; do
  extra=""; [ $arm = control ] && extra="--no-linearize"
  echo "=== $arm"
  .venv/bin/torchrun --standalone --nproc_per_node=2 /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/second_order_audit_20260926/gn_train.py /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/second_order_audit_20260926/gntrain_pair/$arm /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_traj_20260926/M_lr0.007_s260925_l40s:100 --stop-after 80 --inner-sequences 64 --inner-lr 0.002 --inner-aux-lr 0.0005 --validation-every 20 $extra 2>&1 | grep -v Warning | grep validation_nll
done
