#!/bin/bash
# LR optimum, lower side: the matched EMA-root control (no coordination) at body LR x0.7, S∘PD beta 0.8 from initialization,
# 8 curvature sequences + EMA 0.9 of C; spd_control_c8ema_lr07_from0 (MUON_CASE 2026-09-30 11:2x CDT).
cd /share/data/dl-theory/amin/projects/explore_muon
A=logs/muon_spectra/second_order_audit_20260926
.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 $A/newton_train.py $A/band/spd_control_c8ema_lr07_from0 /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/second_order_audit_20260926/band/init16m_spd_b08:0 --normalize 0 --momentum 0.8 --clip 1.0 --staged-pd 0.5 --soap --stage-off --curvature-sequences 8 --stage-c-ema 0.9 --lr-scale 0.7 --validation-every 10 --sharpness-every 10 --checkpoint-every 1000 > $A/band/spd_control_c8ema_lr07_from0.log 2>&1
