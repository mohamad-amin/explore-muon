#!/bin/bash
# 4M with the finer sweep: PD beta 0.9 + above-mean band, sublayer stages, 16M-matched dose (--stage-scale 0.5), @37 -> 368,
# against the 4M harness control (MUON_CASE 2026-09-29 22:4x CDT).
cd /share/data/dl-theory/amin/projects/explore_muon
A=logs/muon_spectra/second_order_audit_20260926
.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 $A/newton_train.py $A/band/band_sublayer_4m_half /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_mom4m4_20260927/PD_a0.5_b4M_lr0.02_mom0.9_s260925_ada:37 --normalize 0 --momentum 0.9 --clip 1.0 --staged-pd 0.5 --stage-band top --stage-granularity sublayer --stage-scale 0.5 --validation-every 20 --sharpness-every 40 --checkpoint-every 1000 > $A/band/band_sublayer_4m_half.log 2>&1
