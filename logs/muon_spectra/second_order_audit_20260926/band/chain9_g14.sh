#!/bin/bash
# 1M rung: the above-mean band at the 16M dose (kappa ~ 0.77 at 1M, so --stage-scale 0.65 gives ~0.5), after the S∘PD control on priv-g14.
cd /share/data/dl-theory/amin/projects/explore_muon
A=logs/muon_spectra/second_order_audit_20260926
until [ -f $A/band/spd_control_b08/done.json ]; do sleep 20; done
.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 $A/newton_train.py $A/band/band_top_1m_half /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_pdtop1m_20260928/PD_a0.5_lr0.01_s260925_ada:200 --normalize 0 --momentum 0.95 --clip 1.0 --staged-pd 0.5 --stage-band top --stage-scale 0.65 --validation-every 50 --sharpness-every 100 --checkpoint-every 1000 > $A/band/band_top_1m_half.log 2>&1
