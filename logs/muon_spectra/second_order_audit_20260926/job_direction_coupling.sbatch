#!/bin/bash
#SBATCH --job-name=so-coupling
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --exclude=g10,g12,g13
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/second_order_audit_20260926/direction_coupling/job_%j.log
#SBATCH --chdir=/share/data/dl-theory/amin/projects/explore_muon
set -uo pipefail
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
L=/share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra
A=$L/second_order_audit_20260926
P16=$L/soaudit_batch16m_20260927/PD_a0.5_b16M_lr0.028_mom0.9_s260925_l40s
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for item in PD16M_46=$P16:46 PD4M_183=$L/soaudit_mom4m4_20260927/PD_a0.5_b4M_lr0.02_mom0.9_s260925_ada:183 M16M_46=$L/soaudit_batch16m_20260927/M_b16M_lr0.02_mom0.9_s260925_l40s:46 PD1M_900=$L/soaudit_traj_20260926/PD_a0.25_lr0.01_s260925_ada:900; do
  tag=${item%%=*}; target=${item#*=}
  .venv/bin/python -u $A/direction_coupling_probe.py $A/direction_coupling/$tag.json $target > $A/direction_coupling/$tag.log 2>&1
  echo "$tag exit $?"
done
