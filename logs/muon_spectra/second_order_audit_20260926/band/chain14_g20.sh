#!/bin/bash
# From initialization: S∘PD beta 0.8 + above-mean band sublayer sweep over the whole 16M schedule (steps 1-92), then its
# harness control (MUON_CASE 2026-09-29 23:3x CDT). Init = the real S∘PD beta 0.8 run's (seed 260925), zero momentum.
cd /share/data/dl-theory/amin/projects/explore_muon
A=logs/muon_spectra/second_order_audit_20260926
TR=".venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4"
COMMON="--normalize 0 --momentum 0.8 --clip 1.0 --staged-pd 0.5 --soap --validation-every 10 --sharpness-every 10 --checkpoint-every 1000"
$TR $A/newton_train.py $A/band/spd_sublayer_from0 /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/second_order_audit_20260926/band/init16m_spd_b08:0 $COMMON --stage-band top --stage-granularity sublayer > $A/band/spd_sublayer_from0.log 2>&1
$TR $A/newton_train.py $A/band/spd_control_from0 /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/second_order_audit_20260926/band/init16m_spd_b08:0 $COMMON --stage-off > $A/band/spd_control_from0.log 2>&1
