#!/bin/bash
# S∘PD harness control at beta 0.8 (--soap --stage-off), after the dose-matched 4M run on priv-g14 (MUON_CASE 14:44 CDT).
cd /share/data/dl-theory/amin/projects/explore_muon
A=logs/muon_spectra/second_order_audit_20260926
until [ -f $A/band/band_top_4m_half/done.json ]; do sleep 20; done
.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 $A/newton_train.py $A/band/spd_control_b08 /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_prefilter16m_20260928/SPD_a0.5_b16M_lr0.028_mom0.8_s260925_ada:9 --normalize 0 --momentum 0.8 --clip 1.0 --staged-pd 0.5 --soap --stage-off --validation-every 10 --sharpness-every 10 --checkpoint-every 1000 > $A/band/spd_control_b08.log 2>&1
