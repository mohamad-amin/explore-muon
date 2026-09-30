#!/bin/bash
cd /share/data/dl-theory/amin/projects/explore_muon
A=logs/muon_spectra/second_order_audit_20260926
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
g=0
while read -r item; do
  tag=$(echo "$item" | sed 's#.*/muon_spectra/##; s#/#__#g; s#:#__#')
  CUDA_VISIBLE_DEVICES=$g .venv/bin/python $A/step_profile_probe.py "$A/step_profile_pdtop/$tag.json" "$item" > "$A/step_profile_pdtop/$tag.log" 2>&1 &
  g=$((g+1))
done < $A/step_profile_pdtop/items.txt
wait
