#!/bin/bash
# Does the finer sweep still need the band? S∘PD beta 0.8, sublayer stages, correction on all input directions (no band),
# 16M @9 -> 92 (MUON_CASE 2026-09-29 20:5x CDT).
cd /share/data/dl-theory/amin/projects/explore_muon
A=logs/muon_spectra/second_order_audit_20260926
.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 $A/newton_train.py $A/band/spd_sublayer_full_b08 /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_prefilter16m_20260928/SPD_a0.5_b16M_lr0.028_mom0.8_s260925_ada:9 --normalize 0 --momentum 0.8 --clip 1.0 --staged-pd 0.5 --soap --stage-granularity sublayer --validation-every 10 --sharpness-every 10 --checkpoint-every 1000 > $A/band/spd_sublayer_full_b08.log 2>&1
