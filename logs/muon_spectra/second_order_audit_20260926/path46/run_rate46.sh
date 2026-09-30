#!/bin/bash
# Rate test (MUON_CASE 2026-09-30 12:30 CDT, the reviewer's next step): 20 outer 16M batches from PD a1/2 beta 0.8 @46 with
# one PD step per batch (A), four true 4M steps per batch (B), four steps per batch on the batch's GN model re-linearized at
# every batch (C, --linearize-every 4), and one coordinated step per batch (D). Normal harness otherwise (aux AdamW, per-step
# PD root from 32 curvature sequences per rank, clip 1.0). E (inner4): the cheap inner loop, four PD iterations per 16M batch
# on the momentum plus kappa G (displacement so far) from the step's curvature sequences (--inner-steps 4, 12:4x CDT).
cd /share/data/dl-theory/amin/projects/explore_muon
A=logs/muon_spectra/second_order_audit_20260926
PD=/share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_mom16m_20260928/PD_a0.5_b16M_lr0.028_mom0.8_s260925_l40s
TR=".venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4"
R="--normalize 0 --clip 1.0 --staged-pd 0.5 --checkpoint-every 1000"
run() { local out=$1; shift; [ -f $A/path46/rate/$out/done.json ] || $TR $A/newton_train.py $A/path46/rate/$out "$@" > $A/path46/rate/$out.log 2>&1; }
run k4lin $A/path46/start_k4:184 $R --momentum 0.9 --stage-off --lr-scale 0.714286 --stop-after 80 --validation-every 4 --linearize-every 4
run k4 $A/path46/start_k4:184 $R --momentum 0.9 --stage-off --lr-scale 0.714286 --stop-after 80 --validation-every 4
run k1 $PD:46 $R --momentum 0.8 --stage-off --stop-after 20 --validation-every 1
run k1c $PD:46 $R --momentum 0.8 --stage-band top --stage-granularity sublayer --stop-after 20 --validation-every 1
run inner4 $PD:46 $R --momentum 0.8 --stage-off --stop-after 20 --validation-every 1 --inner-steps 4 --inner-lr-scale 0.714286
