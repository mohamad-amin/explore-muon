#!/bin/bash
# Batch law at 32M: S∘PD beta 0.8 from initialization, cheap version (8 curvature sequences + EMA 0.9 of C), sublayer band
# sweep; spd_cheap_32m_from0 (MUON_CASE 2026-09-30 11:35 CDT).
cd /share/data/dl-theory/amin/projects/explore_muon
A=logs/muon_spectra/second_order_audit_20260926
.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 $A/newton_train.py $A/band/spd_cheap_32m_from0 /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/second_order_audit_20260926/band/init32m_spd_b08:0 --normalize 0 --momentum 0.8 --clip 1.0 --staged-pd 0.5 --soap --stage-band top --stage-granularity sublayer --curvature-sequences 8 --stage-c-ema 0.9 --validation-every 5 --sharpness-every 5 --checkpoint-every 1000 > $A/band/spd_cheap_32m_from0.log 2>&1
