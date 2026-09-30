#!/bin/bash
# After the 4M 8-direction band on priv-g14: the beta 0.8 harness control from PD beta 0.8 @9 (MUON_CASE 13:17 CDT).
cd /share/data/dl-theory/amin/projects/explore_muon
A=logs/muon_spectra/second_order_audit_20260926
until [ -f $A/band/band_top8_4m/done.json ]; do sleep 20; done
.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 $A/newton_train.py $A/band/control_b08 /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_mom16m_20260928/PD_a0.5_b16M_lr0.028_mom0.8_s260925_l40s:9 --normalize 0 --momentum 0.8 --clip 1.0 --staged-pd 0.5 --stage-off --validation-every 10 --sharpness-every 10 --checkpoint-every 1000 > $A/band/control_b08.log 2>&1
