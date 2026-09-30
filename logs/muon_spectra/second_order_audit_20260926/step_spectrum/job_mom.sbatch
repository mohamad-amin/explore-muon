#!/bin/bash
#SBATCH --job-name=soaudit-spectrum-mom
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --array=1-6
#SBATCH --output=/share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/second_order_audit_20260926/step_spectrum/slurm-%A_%a.log
#SBATCH --chdir=/share/data/dl-theory/amin/projects/explore_muon
set -euo pipefail
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
A=logs/muon_spectra/second_order_audit_20260926
item=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $A/step_spectrum/items_mom.txt)
tag=$(echo "$item" | sed 's#.*/muon_spectra/##; s#/#__#g; s#:#__#')
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
exec .venv/bin/python $A/step_spectrum_probe.py "$A/step_spectrum/$tag.json" "$item"
