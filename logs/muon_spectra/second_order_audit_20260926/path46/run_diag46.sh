#!/bin/bash
# Diagnostic for the GN-model inner loop's divergence (MUON_CASE 2026-09-30 13:24 CDT): the top GN eigenvalue on the curvature
# sequences at the 4th inner step of every 16M batch, for the frozen-model loop (C) and the true loop (B), 15 batches.
cd /share/data/dl-theory/amin/projects/explore_muon
A=logs/muon_spectra/second_order_audit_20260926
TR=".venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4"
R="--normalize 0 --clip 1.0 --staged-pd 0.5 --checkpoint-every 1000 --validation-every 4 --sharpness-every 4"
run() { local out=$1; shift; [ -f $A/path46/diag/$out/done.json ] || $TR $A/newton_train.py $A/path46/diag/$out "$@" > $A/path46/diag/$out.log 2>&1; }
run C $A/path46/start_k4:184 $R --momentum 0.9 --stage-off --lr-scale 0.714286 --stop-after 60 --linearize-every 4
run B $A/path46/start_k4:184 $R --momentum 0.9 --stage-off --lr-scale 0.714286 --stop-after 60
