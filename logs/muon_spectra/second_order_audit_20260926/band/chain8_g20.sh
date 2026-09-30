#!/bin/bash
# After the 1M control on g20: the Jacobi arm (S∘PD beta 0.8 + above-mean band, corrections from plain steps) and the
# 2x-horizon S∘PD beta 0.8 + above-mean band from the real 2x run @37 (second review 16:23 CDT).
cd /share/data/dl-theory/amin/projects/explore_muon
A=logs/muon_spectra/second_order_audit_20260926
TR=".venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4"
until [ -f $A/band/control_1m/done.json ]; do sleep 20; done
$TR $A/newton_train.py $A/band/spd_jacobi_b08 /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_prefilter16m_20260928/SPD_a0.5_b16M_lr0.028_mom0.8_s260925_ada:9 --normalize 0 --momentum 0.8 --clip 1.0 --staged-pd 0.5 --soap --stage-band top --stage-jacobi --validation-every 10 --sharpness-every 10 --checkpoint-every 1000 > $A/band/spd_jacobi_b08.log 2>&1
$TR $A/newton_train.py $A/band/spd_top_2x_b08 /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_horizon16m_20260928/SPD_a0.5_b16M_T2x_lr0.028_mom0.8_s260925_ada:37 --normalize 0 --momentum 0.8 --clip 1.0 --staged-pd 0.5 --soap --stage-band top --validation-every 20 --sharpness-every 20 --checkpoint-every 1000 > $A/band/spd_top_2x_b08.log 2>&1
