#!/bin/bash
# The sublayer sweep over the 2x horizon: S∘PD beta 0.8 + above-mean band, sublayer stages, from the real 2x run @37 to
# 184, against the existing 2x harness control (MUON_CASE 2026-09-29 21:0x CDT).
cd /share/data/dl-theory/amin/projects/explore_muon
A=logs/muon_spectra/second_order_audit_20260926
.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 $A/newton_train.py $A/band/spd_sublayer_2x_b08 /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_horizon16m_20260928/SPD_a0.5_b16M_T2x_lr0.028_mom0.8_s260925_ada:37 --normalize 0 --momentum 0.8 --clip 1.0 --staged-pd 0.5 --soap --stage-band top --stage-granularity sublayer --validation-every 20 --sharpness-every 20 --checkpoint-every 1000 > $A/band/spd_sublayer_2x_b08.log 2>&1
