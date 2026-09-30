#!/bin/bash
#SBATCH --job-name=tiny-regime-follow
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_rtx_a4000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/share/data/dl-theory/amin/projects/explore_muon/tiny_surrogate/logs/tiny_spectra/regime_check_20260929/console/follow_%j.log
# Usage: sbatch job_followup.sh <run_id> <final_step> [ns_diagnostic states...]
# Completes a run's analysis after the 18:05 CDT failure at the final-state momentum check (final.pt load):
# the final momentum state (merged into momentum.json), the gains stage, and, for Muon, the Newton-Schulz
# check repeated with the trainer's four-sequence replay microbatch.
set -uo pipefail
export CC=/usr/bin/gcc CXX=/usr/bin/g++ OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
ROOT=/share/data/dl-theory/amin/projects/explore_muon/tiny_surrogate
P=$ROOT/logs/tiny_spectra/regime_check_20260929/regime_probe.py
RUN=$1; FINAL=$2; shift 2
echo "run $RUN host $(hostname) gpu $CUDA_VISIBLE_DEVICES start $(date)"
$ROOT/run $P momentum "$RUN" --states "$FINAL" || exit 1
$ROOT/run $P gains "$RUN" || exit 1
if [ $# -gt 0 ]; then $ROOT/run $P ns_diagnostic "$RUN" --states "$@" --micro 4 || exit 1; fi
echo "end $(date)"
