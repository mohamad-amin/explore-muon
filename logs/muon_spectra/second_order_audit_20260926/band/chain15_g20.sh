#!/bin/bash
# After the 1x control from step 0 on g20: the 2x-horizon S∘PD beta 0.8 harness control from initialization.
cd /share/data/dl-theory/amin/projects/explore_muon
A=logs/muon_spectra/second_order_audit_20260926
until [ -f $A/band/spd_control_from0/done.json ]; do sleep 20; done
.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 $A/newton_train.py $A/band/spd_control_2x_from0 /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/second_order_audit_20260926/band/init16m2x_spd_b08:0 --normalize 0 --momentum 0.8 --clip 1.0 --staged-pd 0.5 --soap --stage-off --validation-every 20 --sharpness-every 20 --checkpoint-every 1000 > $A/band/spd_control_2x_from0.log 2>&1
