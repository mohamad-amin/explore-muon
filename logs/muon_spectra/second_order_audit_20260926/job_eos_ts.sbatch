#!/bin/bash
#SBATCH --job-name=so-eos-ts
#SBATCH --partition=dev-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/second_order_audit_20260926/job_eos_ts_%j.log
#SBATCH --chdir=/share/data/dl-theory/amin/projects/explore_muon
set -euo pipefail
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
exec .venv/bin/python -u logs/muon_spectra/second_order_audit_20260926/eos_invariant.py logs/muon_spectra/second_order_audit_20260926/eos_invariant_ts.json /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_twosided_20260927/TS_a0.25_b0.25gn_lr0.01_s260925_a4000:500:0.25:0.25 /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_twosided_20260927/PD_a0.25_lr0.01_s260925_a4000:500:0.25 /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_twosided_20260927/TS_a0.25_b0.25gn_lr0.01_s260925_a4000:900:0.25:0.25 /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_twosided_20260927/PD_a0.25_lr0.01_s260925_a4000:900:0.25
