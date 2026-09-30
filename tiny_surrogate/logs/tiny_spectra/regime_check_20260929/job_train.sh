#!/bin/bash
#SBATCH --job-name=tiny-regime-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_rtx_a4000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --array=0-3
#SBATCH --output=/share/data/dl-theory/amin/projects/explore_muon/tiny_surrogate/logs/tiny_spectra/regime_check_20260929/console/train_%A_%a.log
set -uo pipefail
export CC=/usr/bin/gcc CXX=/usr/bin/g++ OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1
ROOT=/share/data/dl-theory/amin/projects/explore_muon/tiny_surrogate
C=$ROOT/logs/tiny_spectra/regime_check_20260929
RUNS=(muon_b262144_lr0.01_s20261001 pd_b262144_lr0.01_s20261001 muon_b1048576_lr0.01_s20261001 pd_b1048576_lr0.02_s20261001)
ID=${RUNS[$SLURM_ARRAY_TASK_ID]}
echo "task $SLURM_ARRAY_TASK_ID run $ID host $(hostname) gpu $CUDA_VISIBLE_DEVICES start $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$ROOT/run -m research.tiny_spectra.train --config $C/configs/$ID.json --out $C/runs/$ID
EXIT=$?
echo "exit $EXIT end $(date)"
exit $EXIT
