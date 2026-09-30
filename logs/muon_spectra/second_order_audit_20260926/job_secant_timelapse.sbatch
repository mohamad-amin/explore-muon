#!/bin/bash
#SBATCH --job-name=so-secantT
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --exclude=g10,g12,g13
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:30:00
#SBATCH --output=/share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/second_order_audit_20260926/secant_timelapse/job_%j.log
#SBATCH --chdir=/share/data/dl-theory/amin/projects/explore_muon
set -uo pipefail
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
L=/share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra
A=$L/second_order_audit_20260926
P16=$L/soaudit_batch16m_20260927/PD_a0.5_b16M_lr0.028_mom0.9_s260925_l40s
P4=$L/soaudit_mom4m4_20260927/PD_a0.5_b4M_lr0.02_mom0.9_s260925_ada
P1=$L/soaudit_traj_20260926/PD_a0.25_lr0.01_s260925_ada
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for item in M1M_10=$L/soaudit_traj_20260926/M_lr0.007_s260925_l40s:10 M1M_50=$L/soaudit_traj_20260926/M_lr0.007_s260925_l40s:50 M1M_100=$L/soaudit_traj_20260926/M_lr0.007_s260925_l40s:100 M1M_200=$L/soaudit_traj_20260926/M_lr0.007_s260925_l40s:200 M1M_500=$L/soaudit_traj_20260926/M_lr0.007_s260925_l40s:500 M1M_900=$L/soaudit_traj_20260926/M_lr0.007_s260925_l40s:900 M1M_1300=$L/soaudit_traj_20260926/M_lr0.007_s260925_l40s:1300 PD1M_10=$L/soaudit_traj_20260926/PD_a0.25_lr0.01_s260925_ada:10 PD1M_50=$L/soaudit_traj_20260926/PD_a0.25_lr0.01_s260925_ada:50 PD1M_100=$L/soaudit_traj_20260926/PD_a0.25_lr0.01_s260925_ada:100 PD1M_200=$L/soaudit_traj_20260926/PD_a0.25_lr0.01_s260925_ada:200 PD1M_500=$L/soaudit_traj_20260926/PD_a0.25_lr0.01_s260925_ada:500 PD1M_900=$L/soaudit_traj_20260926/PD_a0.25_lr0.01_s260925_ada:900 PD1M_1300=$L/soaudit_traj_20260926/PD_a0.25_lr0.01_s260925_ada:1300 SPD1M_10=$L/soaudit_traj_20260926/SPD_a0.25_lr0.01_s260925_ada:10 SPD1M_50=$L/soaudit_traj_20260926/SPD_a0.25_lr0.01_s260925_ada:50 SPD1M_100=$L/soaudit_traj_20260926/SPD_a0.25_lr0.01_s260925_ada:100 SPD1M_200=$L/soaudit_traj_20260926/SPD_a0.25_lr0.01_s260925_ada:200 SPD1M_500=$L/soaudit_traj_20260926/SPD_a0.25_lr0.01_s260925_ada:500 SPD1M_900=$L/soaudit_traj_20260926/SPD_a0.25_lr0.01_s260925_ada:900 SPD1M_1300=$L/soaudit_traj_20260926/SPD_a0.25_lr0.01_s260925_ada:1300 S1M_10=$L/soaudit_traj_20260926/S_lr0.007_s260925_l40s:10 S1M_50=$L/soaudit_traj_20260926/S_lr0.007_s260925_l40s:50 S1M_100=$L/soaudit_traj_20260926/S_lr0.007_s260925_l40s:100 S1M_200=$L/soaudit_traj_20260926/S_lr0.007_s260925_l40s:200 S1M_500=$L/soaudit_traj_20260926/S_lr0.007_s260925_l40s:500 S1M_900=$L/soaudit_traj_20260926/S_lr0.007_s260925_l40s:900 S1M_1300=$L/soaudit_traj_20260926/S_lr0.007_s260925_l40s:1300; do
  tag=${item%%=*}; target=${item#*=}
  .venv/bin/python -u $A/secant_probe.py $A/secant_timelapse/$tag.json $target > $A/secant_timelapse/$tag.log 2>&1
  echo "$tag exit $?"
done
