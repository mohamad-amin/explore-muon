#!/bin/bash
# Review's diagnostic (MUON_CASE 2026-09-30 13:50 CDT), arm C: four steps per 16M batch on the batch's GN model, aux frozen,
# lambda_max at every linearization point, the model's curvature along the displacement at every inner step, 20 batches.
cd /share/data/dl-theory/amin/projects/explore_muon
A=logs/muon_spectra/second_order_audit_20260926
TR=".venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4"
R="--normalize 0 --clip 1.0 --staged-pd 0.5 --checkpoint-every 1000 --freeze-aux --sharpness-from-start"
[ -f $A/path46/diag2/C/done.json ] || $TR $A/newton_train.py $A/path46/diag2/C $A/path46/start_k4:184 $R --momentum 0.9 --stage-off --lr-scale 0.714286 --stop-after 80 --validation-every 4 --sharpness-every 4 --linearize-every 4 > $A/path46/diag2/C.log 2>&1
