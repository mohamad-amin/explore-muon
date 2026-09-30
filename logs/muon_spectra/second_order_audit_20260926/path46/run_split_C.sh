#!/bin/bash
# The review's stiff/bulk split on the GN-model inner loop (MUON_CASE 2026-09-30 14:0x CDT): arm C (four steps per 16M batch on
# the batch's GN model, aux frozen) with the top 16 GN eigenvectors at each linearization point projected out of inner steps
# 2-4 and of their curvature term; lambda at every linearization point; 20 batches.
cd /share/data/dl-theory/amin/projects/explore_muon
A=logs/muon_spectra/second_order_audit_20260926
TR=".venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4"
R="--normalize 0 --clip 1.0 --staged-pd 0.5 --checkpoint-every 1000 --freeze-aux --sharpness-from-start"
[ -f $A/path46/diag2/Csplit/done.json ] || $TR $A/newton_train.py $A/path46/diag2/Csplit $A/path46/start_k4:184 $R --momentum 0.9 --stage-off --lr-scale 0.714286 --stop-after 80 --validation-every 4 --sharpness-every 4 --linearize-every 4 --inner-split 16 > $A/path46/diag2/Csplit.log 2>&1
