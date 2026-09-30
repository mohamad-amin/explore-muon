#!/bin/bash
#SBATCH --job-name=tiny-split-par
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/share/data/dl-theory/amin/projects/explore_muon/tiny_surrogate/logs/tiny_spectra/staleness_oracle_20260930/console/%x_%j.log
# Usage: sbatch --job-name=<arm> --gres=gpu:nvidia_rtx_a4000:<N> job_parallel.sh <arm> <N>
set -uo pipefail
ROOT=/share/data/dl-theory/amin/projects/explore_muon/tiny_surrogate
C=$ROOT/logs/tiny_spectra/staleness_oracle_20260930
export CC=/usr/bin/gcc CXX=/usr/bin/g++ OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
echo "arm $1 gpus $2 host $(hostname) visible $CUDA_VISIBLE_DEVICES start $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$ROOT/run -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=$2 --max-restarts=0 $C/split_parallel_v3.py --config $C/configs/$1.json --out $C/runs/$1 --expected-world-size $2
EXIT=$?
echo "exit $EXIT end $(date)"
exit $EXIT
