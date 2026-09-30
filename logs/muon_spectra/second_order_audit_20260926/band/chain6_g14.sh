#!/bin/bash
# Second-initialization replicate of the beta 0.8 composition: harness control from the seed-260926 S∘PD beta 0.8 run's @9
# state with PD's map (MUON_CASE 2026-09-29 14:4x CDT). The band arm runs on g20.
cd /share/data/dl-theory/amin/projects/explore_muon
A=logs/muon_spectra/second_order_audit_20260926
.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 $A/newton_train.py $A/band/control_b08_s2 /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_mom16mseed_20260928/SPD_a0.5_b16M_lr0.028_mom0.8_s260926_ada:9 --normalize 0 --momentum 0.8 --clip 1.0 --staged-pd 0.5 --stage-off --validation-every 10 --sharpness-every 10 --checkpoint-every 1000 > $A/band/control_b08_s2.log 2>&1
