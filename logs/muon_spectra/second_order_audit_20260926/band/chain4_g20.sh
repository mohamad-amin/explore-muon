#!/bin/bash
# After the second-initialization replicate on g20 (MUON_CASE 2026-09-29 13:17 CDT): the correction-vs-shrink probe at the
# control @46 state, then the beta 0.8 composition: above-mean band and 8-direction band from PD beta 0.8 @9 (--momentum 0.8).
cd /share/data/dl-theory/amin/projects/explore_muon
A=logs/muon_spectra/second_order_audit_20260926
TR=".venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4"
COMMON="--normalize 0 --momentum 0.8 --clip 1.0 --staged-pd 0.5 --validation-every 10 --sharpness-every 10 --checkpoint-every 1000"
until [ -f $A/band/band_top_s2/done.json ]; do sleep 20; done
$TR $A/correction_probe.py $A/band/correction_control_46.json /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_batch16m_20260927/PD_a0.5_b16M_lr0.028_mom0.9_s260925_l40s:9 --harness $A/floor_ledger/control_46/kept/step000046.pt > $A/band/correction_control_46.log 2>&1
$TR $A/newton_train.py $A/band/band_top_b08 /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_mom16m_20260928/PD_a0.5_b16M_lr0.028_mom0.8_s260925_l40s:9 $COMMON --stage-band top > $A/band/band_top_b08.log 2>&1
$TR $A/newton_train.py $A/band/band_top8_b08 /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_mom16m_20260928/PD_a0.5_b16M_lr0.028_mom0.8_s260925_l40s:9 $COMMON --stage-band top8 > $A/band/band_top8_b08.log 2>&1
