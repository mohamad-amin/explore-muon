#!/bin/bash
# Low-noise full layer-staged correction (256 curvature sequences per rank), 16M @9 -> 92 (MUON_CASE 2026-09-29 08:45 CDT).
cd /share/data/dl-theory/amin/projects/explore_muon
A=logs/muon_spectra/second_order_audit_20260926
.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 $A/newton_train.py $A/band/staged_big_256 /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_batch16m_20260927/PD_a0.5_b16M_lr0.028_mom0.9_s260925_l40s:9 --normalize 0 --momentum 0.9 --clip 1.0 --staged-pd 0.5 --curvature-sequences 256 --validation-every 10 --sharpness-every 10 --checkpoint-every 1000 --keep 46 > $A/band/staged_big_256.log 2>&1
