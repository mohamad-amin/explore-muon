#!/bin/bash
#SBATCH --job-name=tiny-oracle
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_rtx_a4000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/share/data/dl-theory/amin/projects/explore_muon/tiny_surrogate/logs/tiny_spectra/staleness_oracle_20260930/console/%x_%j.log
# Usage: sbatch --job-name=<arm> job_run.sh <arm>
set -uo pipefail
export CC=/usr/bin/gcc CXX=/usr/bin/g++ OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
ROOT=/share/data/dl-theory/amin/projects/explore_muon/tiny_surrogate
C=$ROOT/logs/tiny_spectra/staleness_oracle_20260930
echo "arm $1 host $(hostname) gpu $CUDA_VISIBLE_DEVICES start $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$ROOT/run $C/oracle_train.py --config $C/configs/$1.json --out $C/runs/$1
EXIT=$?
echo "exit $EXIT end $(date)"
exit $EXIT
