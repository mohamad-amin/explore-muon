#!/bin/bash
# Dose-response of the frozen-curvature interval (MUON_CASE 2026-09-30 14:4x CDT): the same 4M sub-batch steps as B and C,
# re-linearized every 2 steps (two linearization points per 16M batch), aux frozen, lambda at every batch start. B is
# K_lin = 1 (the true gradient every step), C is K_lin = 4.
cd /share/data/dl-theory/amin/projects/explore_muon
A=logs/muon_spectra/second_order_audit_20260926
TR=".venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4"
R="--normalize 0 --clip 1.0 --staged-pd 0.5 --checkpoint-every 1000 --freeze-aux --sharpness-from-start"
[ -f $A/path46/diag2/C2/done.json ] || $TR $A/newton_train.py $A/path46/diag2/C2 $A/path46/start_k4:184 $R --momentum 0.9 --stage-off --lr-scale 0.714286 --stop-after 80 --validation-every 4 --sharpness-every 4 --linearize-every 2 > $A/path46/diag2/C2.log 2>&1
