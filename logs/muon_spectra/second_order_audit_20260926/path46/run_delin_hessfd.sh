#!/bin/bash
# Full-Hessian frozen quadratic model in the same loop as C (MUON_CASE 2026-09-30 15:41 CDT, from arXiv 2607.21716's negative-
# curvature finding): inner-step gradient g_q(theta_0) + [g_q(theta_0 + 0.1 D) - g_q(theta_0 - 0.1 D)] / 0.2, re-linearized
# every 16M batch, aux frozen, lambda at every batch start, 20 batches; residual logging as the other modes.
cd /share/data/dl-theory/amin/projects/explore_muon
A=logs/muon_spectra/second_order_audit_20260926
TR=".venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4"
R="--normalize 0 --clip 1.0 --staged-pd 0.5 --checkpoint-every 1000 --freeze-aux --sharpness-from-start"
[ -f $A/path46/diag2/L_hessfd/done.json ] || $TR $A/newton_train.py $A/path46/diag2/L_hessfd $A/path46/start_k4:184 $R --momentum 0.9 --stage-off --lr-scale 0.714286 --stop-after 80 --validation-every 4 --sharpness-every 4 --linearize-every 4 --linearize-mode hessfd > $A/path46/diag2/L_hessfd.log 2>&1
