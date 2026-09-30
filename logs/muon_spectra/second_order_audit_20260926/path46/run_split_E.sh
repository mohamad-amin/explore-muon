#!/bin/bash
# The stiff/bulk split on the cheap inner loop (MUON_CASE 2026-09-30 14:0x CDT): arm E (four PD iterations per 16M batch on
# M + kappa G_curv (displacement)) with the top 16 GN eigenvectors projected out of iterations 2-4 and of their curvature term.
cd /share/data/dl-theory/amin/projects/explore_muon
A=logs/muon_spectra/second_order_audit_20260926
PD=/share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_mom16m_20260928/PD_a0.5_b16M_lr0.028_mom0.8_s260925_l40s
TR=".venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4"
[ -f $A/path46/diag2/Esplit/done.json ] || $TR $A/newton_train.py $A/path46/diag2/Esplit $PD:46 --normalize 0 --clip 1.0 --staged-pd 0.5 --checkpoint-every 1000 --momentum 0.8 --stage-off --stop-after 20 --validation-every 1 --inner-steps 4 --inner-lr-scale 0.714286 --inner-split 16 > $A/path46/diag2/Esplit.log 2>&1
