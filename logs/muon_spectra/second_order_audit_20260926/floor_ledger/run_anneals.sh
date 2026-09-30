#!/bin/bash
# Floor ledger (MUON_CASE 2026-09-29 07:19 CDT): real-trainer 16-step anneals (anneal_branch.py, the run's own optimizer,
# momentum schedule and data order), one per GPU, in parallel. Arguments: LABEL=ARM_DIR:STEP pairs, at most 4.
cd /share/data/dl-theory/amin/projects/explore_muon
L=logs/muon_spectra
A=$L/second_order_audit_20260926
gpu=0
for spec in "$@"; do
  label=${spec%%=*}; item=${spec#*=}
  CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python $A/anneal_branch.py $A/floor_ledger/$label $L/$item --steps 16 --no-save > $A/floor_ledger/$label.log 2>&1 &
  gpu=$((gpu + 1))
done
wait
