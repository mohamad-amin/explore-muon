#!/bin/bash
#SBATCH --job-name=so-bandprobe
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --constraint=48g
#SBATCH --exclude=g10,g12,g13
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=00:45:00
#SBATCH --output=/share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/second_order_audit_20260926/band/probe_%j.log
#SBATCH --chdir=/share/data/dl-theory/amin/projects/explore_muon
# What band-restricted coordination changes in one step, at the control and staged @46 harness states (MUON_CASE 08:45).
set -uo pipefail
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
A=logs/muon_spectra/second_order_audit_20260926
RUN=/share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_batch16m_20260927/PD_a0.5_b16M_lr0.028_mom0.9_s260925_l40s
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for st in control staged; do
  .venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 $A/band_probe.py $A/band/probe_${st}_46.json $RUN:9 --harness $A/floor_ledger/${st}_46/kept/step000046.pt
  echo "$st exit $?"
done
