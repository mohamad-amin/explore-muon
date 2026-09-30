#!/bin/bash
# De-linearization test (peer discussion, MUON_CASE 2026-09-30 14:53 CDT): four 4M steps per 16M batch from PD @46, aux frozen,
# re-linearized every batch, inner-step gradient mode diff; residual and second-difference logging; lambda at every batch start.
cd /share/data/dl-theory/amin/projects/explore_muon
A=logs/muon_spectra/second_order_audit_20260926
TR=".venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4"
R="--normalize 0 --clip 1.0 --staged-pd 0.5 --checkpoint-every 1000 --freeze-aux --sharpness-from-start"
[ -f $A/path46/diag2/L_diff/done.json ] || $TR $A/newton_train.py $A/path46/diag2/L_diff $A/path46/start_k4:184 $R --momentum 0.9 --stage-off --lr-scale 0.714286 --stop-after 80 --validation-every 4 --sharpness-every 4 --linearize-every 4 --linearize-mode diff > $A/path46/diag2/L_diff.log 2>&1
