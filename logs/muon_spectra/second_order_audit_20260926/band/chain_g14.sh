#!/bin/bash
# Half-strength full correction (16M @9 -> 92), then the above-mean band at 4M with its harness control (@37 -> 368)
# (MUON_CASE 2026-09-29 08:45 CDT).
cd /share/data/dl-theory/amin/projects/explore_muon
A=logs/muon_spectra/second_order_audit_20260926
TR=".venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4"
$TR $A/newton_train.py $A/band/staged_half /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_batch16m_20260927/PD_a0.5_b16M_lr0.028_mom0.9_s260925_l40s:9 --normalize 0 --momentum 0.9 --clip 1.0 --staged-pd 0.5 --stage-scale 0.5 --validation-every 10 --sharpness-every 10 --checkpoint-every 1000 > $A/band/staged_half.log 2>&1
$TR $A/newton_train.py $A/band/control_4m /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_mom4m4_20260927/PD_a0.5_b4M_lr0.02_mom0.9_s260925_ada:37 --normalize 0 --momentum 0.9 --clip 1.0 --staged-pd 0.5 --stage-off --validation-every 20 --sharpness-every 40 --checkpoint-every 1000 > $A/band/control_4m.log 2>&1
$TR $A/newton_train.py $A/band/band_top_4m /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_mom4m4_20260927/PD_a0.5_b4M_lr0.02_mom0.9_s260925_ada:37 --normalize 0 --momentum 0.9 --clip 1.0 --staged-pd 0.5 --stage-band top --validation-every 20 --sharpness-every 40 --checkpoint-every 1000 > $A/band/band_top_4m.log 2>&1
