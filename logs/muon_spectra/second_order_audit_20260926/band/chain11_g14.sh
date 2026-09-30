#!/bin/bash
# After the oracle-reversal control on priv-g14: the 2x-horizon S∘PD beta 0.8 harness control from the real 2x run @37
# (second review 16:23 CDT; real run final 3.9740), then the 1M above-mean band at the 16M dose (--stage-scale 0.65).
cd /share/data/dl-theory/amin/projects/explore_muon
A=logs/muon_spectra/second_order_audit_20260926
TR=".venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4"
until [ -f $A/band/spd_reversal_b08/done.json ]; do sleep 20; done
$TR $A/newton_train.py $A/band/spd_control_2x_b08 /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_horizon16m_20260928/SPD_a0.5_b16M_T2x_lr0.028_mom0.8_s260925_ada:37 --normalize 0 --momentum 0.8 --clip 1.0 --staged-pd 0.5 --soap --stage-off --validation-every 20 --sharpness-every 20 --checkpoint-every 1000 > $A/band/spd_control_2x_b08.log 2>&1
$TR $A/newton_train.py $A/band/band_top_1m_half /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_pdtop1m_20260928/PD_a0.5_lr0.01_s260925_ada:200 --normalize 0 --momentum 0.95 --clip 1.0 --staged-pd 0.5 --stage-band top --stage-scale 0.65 --validation-every 50 --sharpness-every 100 --checkpoint-every 1000 > $A/band/band_top_1m_half.log 2>&1
