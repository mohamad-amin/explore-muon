#!/bin/bash
#SBATCH --job-name=so-drift
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --exclude=g10,g12,g13
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/second_order_audit_20260926/direction_drift/job_%j.log
#SBATCH --chdir=/share/data/dl-theory/amin/projects/explore_muon
set -uo pipefail
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
L=/share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra
A=$L/second_order_audit_20260926
P16=$L/soaudit_batch16m_20260927/PD_a0.5_b16M_lr0.028_mom0.9_s260925_l40s
P16b=$L/soaudit_mom16m_20260928/PD_a0.5_b16M_lr0.028_mom0.8_s260925_l40s
P4=$L/soaudit_mom4m4_20260927/PD_a0.5_b4M_lr0.02_mom0.9_s260925_ada
P1=$L/soaudit_traj_20260926/PD_a0.25_lr0.01_s260925_ada
M1=$L/soaudit_traj_20260926/M_lr0.007_s260925_l40s
M16=$L/soaudit_batch16m_20260927/M_b16M_lr0.02_mom0.9_s260925_l40s
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for item in PD16M_46=$P16:46 PD1M_900=$P1:900 PD4M_183=$P4:183 PD16M_9=$P16:9 PD16M_83=$P16:83 \
  PD16Mb08_9=$P16b:9 PD16Mb08_46=$P16b:46 PD16Mb08_83=$P16b:83 M16M_46=$M16:46 PD4M_37=$P4:37 PD4M_330=$P4:330 \
  PD1M_50=$P1:50 PD1M_200=$P1:200 M1M_200=$M1:200 M1M_900=$M1:900; do
  tag=${item%%=*}; target=${item#*=}
  .venv/bin/python -u $A/direction_drift_probe.py $A/direction_drift/$tag.json $target > $A/direction_drift/$tag.log 2>&1
  echo "$tag exit $?"
done
