#!/bin/bash
#SBATCH --job-name=soaudit-profile-mom
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --exclude=g10,g12,g13
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --array=1-9
#SBATCH --output=/share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/second_order_audit_20260926/step_profile_mom/slurm-%A_%a.log
#SBATCH --chdir=/share/data/dl-theory/amin/projects/explore_muon
set -euo pipefail
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
A=logs/muon_spectra/second_order_audit_20260926
item=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $A/step_profile_mom/items.txt)
tag=$(echo "$item" | sed 's#.*/muon_spectra/##; s#/#__#g; s#:#__#')
exec .venv/bin/python $A/step_profile_probe.py "$A/step_profile_mom/$tag.json" "$item"
