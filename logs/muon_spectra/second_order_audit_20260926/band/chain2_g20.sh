#!/bin/bash
# After the low-noise run on g20: the one-step band probe at the control and staged @46 states, then the above-mean band
# over the 2x horizon (16M PD 2x run @37 -> 184) with its harness control (MUON_CASE 2026-09-29 09:4x CDT).
cd /share/data/dl-theory/amin/projects/explore_muon
A=logs/muon_spectra/second_order_audit_20260926
TR=".venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4"
until [ -f $A/band/staged_big_256/done.json ]; do sleep 20; done
for st in control staged; do
  $TR $A/band_probe.py $A/band/probe_${st}_46.json /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_batch16m_20260927/PD_a0.5_b16M_lr0.028_mom0.9_s260925_l40s:9 --harness $A/floor_ledger/${st}_46/kept/step000046.pt > $A/band/probe_${st}_46.log 2>&1
done
$TR $A/newton_train.py $A/band/band_top_2x /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_warmrep_20260928/PD_a0.5_b16M_T2x_lr0.028_mom0.9_s260925_l40s:37 --normalize 0 --momentum 0.9 --clip 1.0 --staged-pd 0.5 --stage-band top --validation-every 20 --sharpness-every 20 --checkpoint-every 1000 > $A/band/band_top_2x.log 2>&1
$TR $A/newton_train.py $A/band/control_2x /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_warmrep_20260928/PD_a0.5_b16M_T2x_lr0.028_mom0.9_s260925_l40s:37 --normalize 0 --momentum 0.9 --clip 1.0 --staged-pd 0.5 --stage-off --validation-every 20 --sharpness-every 20 --checkpoint-every 1000 > $A/band/control_2x.log 2>&1
