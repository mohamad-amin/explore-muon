#!/bin/bash
cd /share/data/dl-theory/amin/projects/explore_muon
A=logs/muon_spectra/second_order_audit_20260926
until [ -f $A/band/band_top8_b08/done.json ]; do sleep 20; done
.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 $A/newton_train.py $A/band/band_top_b08_s2 /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_mom16mseed_20260928/SPD_a0.5_b16M_lr0.028_mom0.8_s260926_ada:9 --normalize 0 --momentum 0.8 --clip 1.0 --staged-pd 0.5 --stage-band top --validation-every 10 --sharpness-every 10 --checkpoint-every 1000 > $A/band/band_top_b08_s2.log 2>&1
