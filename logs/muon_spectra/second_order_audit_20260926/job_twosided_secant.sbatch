#!/bin/bash
#SBATCH --job-name=so-2secant
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --exclude=g10,g12,g13
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/second_order_audit_20260926/twosided_secant/job_%j.log
#SBATCH --chdir=/share/data/dl-theory/amin/projects/explore_muon
set -uo pipefail
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
L=/share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra
A=$L/second_order_audit_20260926
B16=$L/soaudit_batch16m_20260927
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for item in M16M_46=$B16/M_b16M_lr0.02_mom0.9_s260925_l40s:46 PD16M_46=$B16/PD_a0.5_b16M_lr0.028_mom0.9_s260925_l40s:46 SPD16M_46=$B16/SPD_a0.5_b16M_lr0.028_mom0.9_s260925_ada:46 TS16M_46=$B16/TS_a0.5_b0.5gn_b16M_lr0.028_mom0.9_s260925_l40s:46 M4M_183=$L/soaudit_batch4m_20260927/M_b4M_lr0.014_s260925_l40s:183 PD4M_183=$L/soaudit_mom4m4_20260927/PD_a0.5_b4M_lr0.02_mom0.9_s260925_ada:183 SPD4M_183=$L/soaudit_strength4m_20260927/SPD_a0.5_b4M_lr0.02_mom0.9_s260925_l40s:183 M1M_900=$L/soaudit_traj_20260926/M_lr0.007_s260925_l40s:900 PD1M_900=$L/soaudit_traj_20260926/PD_a0.25_lr0.01_s260925_ada:900 SPD1M_900=$L/soaudit_traj_20260926/SPD_a0.25_lr0.01_s260925_ada:900 S1M_900=$L/soaudit_traj_20260926/S_lr0.007_s260925_l40s:900 M16M_83=$B16/M_b16M_lr0.02_mom0.9_s260925_l40s:83 PD16M_83=$B16/PD_a0.5_b16M_lr0.028_mom0.9_s260925_l40s:83 SPD16M_83=$B16/SPD_a0.5_b16M_lr0.028_mom0.9_s260925_ada:83 TS16M_83=$B16/TS_a0.5_b0.5gn_b16M_lr0.028_mom0.9_s260925_l40s:83; do
  tag=${item%%=*}; target=${item#*=}
  .venv/bin/python -u $A/twosided_secant_probe.py $A/twosided_secant/$tag.json $target > $A/twosided_secant/$tag.log 2>&1
  echo "$tag exit $?"
done
