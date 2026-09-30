#!/bin/bash
#SBATCH --job-name=so-secant
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --exclude=g10,g12,g13
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/second_order_audit_20260926/secant/job_%j.log
#SBATCH --chdir=/share/data/dl-theory/amin/projects/explore_muon
set -uo pipefail
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
L=/share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra
A=$L/second_order_audit_20260926
P16=$L/soaudit_batch16m_20260927/PD_a0.5_b16M_lr0.028_mom0.9_s260925_l40s
P4=$L/soaudit_mom4m4_20260927/PD_a0.5_b4M_lr0.02_mom0.9_s260925_ada
P1=$L/soaudit_traj_20260926/PD_a0.25_lr0.01_s260925_ada
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for item in PD16M_46=$P16:46 PD4M_183=$P4:183 PD1M_900=$P1:900 M16M_46=$L/soaudit_batch16m_20260927/M_b16M_lr0.02_mom0.9_s260925_l40s:46 M1M_900=$L/soaudit_traj_20260926/M_lr0.007_s260925_l40s:900 PD16M_9=$P16:9 PD16M_83=$P16:83 PD4M_37=$P4:37 PD4M_330=$P4:330 PD1M_50=$P1:50 PD1M_200=$P1:200; do
  tag=${item%%=*}; target=${item#*=}
  .venv/bin/python -u $A/secant_probe.py $A/secant/$tag.json $target > $A/secant/$tag.log 2>&1
  echo "$tag exit $?"
done
