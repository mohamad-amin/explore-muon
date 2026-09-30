#!/bin/bash
# From initialization over the 2x horizon (184 steps): S∘PD beta 0.8 + above-mean band sublayer sweep (MUON_CASE 09-30 00:3x CDT).
cd /share/data/dl-theory/amin/projects/explore_muon
A=logs/muon_spectra/second_order_audit_20260926
.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 $A/newton_train.py $A/band/spd_sublayer_2x_from0 /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/second_order_audit_20260926/band/init16m2x_spd_b08:0 --normalize 0 --momentum 0.8 --clip 1.0 --staged-pd 0.5 --soap --stage-band top --stage-granularity sublayer --validation-every 20 --sharpness-every 20 --checkpoint-every 1000 > $A/band/spd_sublayer_2x_from0.log 2>&1
