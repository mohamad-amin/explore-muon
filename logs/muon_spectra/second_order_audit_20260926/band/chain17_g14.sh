#!/bin/bash
# Matched control for the cheap version (design review 03:4x CDT): S∘PD beta 0.8 from initialization, no coordination,
# R from the same 8-sequence EMA(0.9) of C as the cheap coordinated arm (MUON_CASE 09-30 03:4x CDT).
cd /share/data/dl-theory/amin/projects/explore_muon
A=logs/muon_spectra/second_order_audit_20260926
.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 $A/newton_train.py $A/band/spd_control_c8ema_from0 /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/second_order_audit_20260926/band/init16m_spd_b08:0 --normalize 0 --momentum 0.8 --clip 1.0 --staged-pd 0.5 --soap --stage-off --curvature-sequences 8 --stage-c-ema 0.9 --validation-every 10 --sharpness-every 10 --checkpoint-every 1000 > $A/band/spd_control_c8ema_from0.log 2>&1
