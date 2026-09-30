#!/bin/bash
# Run up to four one_step_gn.py task lists on one node, one GPU each (CUDA_VISIBLE_DEVICES), then wait.
# usage: gn_node.sh OUT_DIR LOG_PREFIX "ARGS0" "ARGS1" ...   (each ARGS: space-separated ARM:STEP items and options)
set -u
OUT=$1; PREFIX=$2; shift 2
REPO=/share/data/dl-theory/amin/projects/explore_muon
gpu=0
for list in "$@"; do
  CUDA_VISIBLE_DEVICES=$gpu nohup $REPO/.venv/bin/python -u $REPO/logs/muon_spectra/second_order_audit_20260926/one_step_gn.py \
      $OUT $list > ${PREFIX}_gpu${gpu}.log 2>&1 &
  gpu=$((gpu + 1))
done
wait
