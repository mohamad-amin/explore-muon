#!/bin/bash
#SBATCH --job-name=so-transport-smoke
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=/share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/second_order_audit_20260926/job_transport_smoke_%j.log
#SBATCH --chdir=/share/data/dl-theory/amin/projects/explore_muon
set -euo pipefail
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
exec .venv/bin/python -u logs/muon_spectra/second_order_audit_20260926/transport_test.py logs/muon_spectra/second_order_audit_20260926/transport_smoke logs/muon_spectra/soaudit_traj_20260926/M_lr0.007_s260925_l40s:500 --smoke --replay 3 --transport-sequences 16 --curvature-sequences 16 --eval-sequences 16 --krylov 4 --no-save
