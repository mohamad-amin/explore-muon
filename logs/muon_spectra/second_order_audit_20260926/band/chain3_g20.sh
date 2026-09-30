#!/bin/bash
# After the 2x harness control on g20: a second-initialization replicate of the above-mean band at 16M 1x. Both arms start
# from the seed-260926 S∘PD run's @9 state (the only other 16M initialization with a kept @9) and use PD's map, as the
# 16M runs do (MUON_CASE 2026-09-29 11:2x CDT).
cd /share/data/dl-theory/amin/projects/explore_muon
A=logs/muon_spectra/second_order_audit_20260926
TR=".venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4"
COMMON="--normalize 0 --momentum 0.9 --clip 1.0 --staged-pd 0.5 --validation-every 10 --sharpness-every 10 --checkpoint-every 1000"
until [ -f $A/band/control_2x/done.json ]; do sleep 20; done
$TR $A/newton_train.py $A/band/control_s2 /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_seed16m_20260928/SPD_a0.5_b16M_lr0.028_mom0.9_s260926_ada:9 $COMMON --stage-off > $A/band/control_s2.log 2>&1
$TR $A/newton_train.py $A/band/band_top_s2 /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_seed16m_20260928/SPD_a0.5_b16M_lr0.028_mom0.9_s260926_ada:9 $COMMON --stage-band top > $A/band/band_top_s2.log 2>&1
