#!/bin/bash
#SBATCH --job-name=so-lags-PD
#SBATCH --partition=dev-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/second_order_audit_20260926/job_lags_PD_%j.log
#SBATCH --chdir=/share/data/dl-theory/amin/projects/explore_muon
set -euo pipefail
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
exec /share/data/dl-theory/amin/projects/explore_muon/.venv/bin/python -u logs/muon_spectra/second_order_audit_20260926/measure_persistence.py logs/muon_spectra/second_order_audit_20260926/persistence_lags /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_denselag_20260927/PD:501:next,503,w504,507,w508,515,w516,531,w532
