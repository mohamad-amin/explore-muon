#!/bin/bash
#SBATCH --job-name=so-denselag-M
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_rtx_a4000:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_denselag_20260927/job_M_%j.log
#SBATCH --chdir=/share/data/dl-theory/amin/projects/explore_muon
set -euo pipefail
export OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1 PYTHONPATH=/share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_traj_20260926/frozen TORCHINDUCTOR_COMPILE_THREADS=1
exec /share/data/dl-theory/amin/projects/explore_muon/.venv/bin/python -u -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 --max_restarts=0 \
  -m adamw_spectra.distributed --config /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_denselag_20260927/M_config.json --out /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_denselag_20260927/M --stop-after 532
