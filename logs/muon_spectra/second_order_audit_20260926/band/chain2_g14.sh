#!/bin/bash
# After the 4M band pair on priv-g14: how concentrated is the useful coupling? The correction kept only on each matrix's
# input-mean direction, then on its 8 largest input eigenvectors (16M @9 -> 92; MUON_CASE 2026-09-29 10:3x CDT).
cd /share/data/dl-theory/amin/projects/explore_muon
A=logs/muon_spectra/second_order_audit_20260926
TR=".venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4"
until [ -f $A/band/band_top_4m/done.json ]; do sleep 20; done
for b in mean top8; do
  $TR $A/newton_train.py $A/band/band_$b /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_batch16m_20260927/PD_a0.5_b16M_lr0.028_mom0.9_s260925_l40s:9 --normalize 0 --momentum 0.9 --clip 1.0 --staged-pd 0.5 --stage-band $b --validation-every 10 --sharpness-every 10 --checkpoint-every 1000 > $A/band/band_$b.log 2>&1
done
