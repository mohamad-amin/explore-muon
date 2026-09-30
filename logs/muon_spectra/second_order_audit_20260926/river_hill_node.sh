#!/bin/bash
# Run river_hill.py over a list of ARM:STEP items on one GPU (CUDA_VISIBLE_DEVICES set by the caller).
REPO=/share/data/dl-theory/amin/projects/explore_muon
OUT=$REPO/logs/muon_spectra/second_order_audit_20260926/river_hill
cd $REPO
for item in "$@"; do
  name=$(basename ${item%:*})_$(echo ${item##*:})
  .venv/bin/python -u logs/muon_spectra/second_order_audit_20260926/river_hill.py $OUT/$name.json $item 2>&1 | grep -v Warning
done
