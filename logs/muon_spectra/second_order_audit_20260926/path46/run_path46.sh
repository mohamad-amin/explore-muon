#!/bin/bash
# Step vs path at 16M (MUON_CASE 2026-09-30 11:40 CDT, revised after review 12:0x): from PD a1/2 beta 0.8 @46 on the run's
# next 16M batch, one 16M step and K sequential smaller-batch PD steps on the same data, on the true loss and on the batch's
# GN model at the start (--linearize-at-start). All arms: aux frozen, one PD root from 64 held-out sequences per rank at the
# start (--fixed-root 64), the inherited momentum (converted with the buffer's own beta 0.8).
cd /share/data/dl-theory/amin/projects/explore_muon
A=logs/muon_spectra/second_order_audit_20260926
PD=/share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_mom16m_20260928/PD_a0.5_b16M_lr0.028_mom0.8_s260925_l40s
TR=".venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4"
R="--normalize 0 --clip 1.0 --staged-pd 0.5 --fixed-root 64 --freeze-aux --validation-every 1000 --checkpoint-every 1000"
run() { local out=$1; shift; [ -f $A/path46/$out/done.json ] || $TR $A/newton_train.py $A/path46/$out "$@" > $A/path46/$out.log 2>&1; }
run k1 $PD:46 $R --momentum 0.8 --stage-off --stop-after 1
run k1lin $PD:46 $R --momentum 0.8 --stage-off --stop-after 1 --linearize-at-start
run k1c $PD:46 $R --momentum 0.8 --stage-band top --stage-granularity sublayer --stop-after 1
run k4 $A/path46/start_k4:184 $R --momentum 0.9 --stage-off --lr-scale 0.714286 --stop-after 4
run k4lin $A/path46/start_k4:184 $R --momentum 0.9 --stage-off --lr-scale 0.714286 --stop-after 4 --linearize-at-start
run k4d $A/path46/start_k4:184 $R --momentum 0.945742 --stage-off --lr-scale 0.25 --stop-after 4
run k4dlin $A/path46/start_k4:184 $R --momentum 0.945742 --stage-off --lr-scale 0.25 --stop-after 4 --linearize-at-start
run k16 $A/path46/start_k16:736 $R --momentum 0.95 --stage-off --lr-scale 0.357143 --stop-after 16
run k16lin $A/path46/start_k16:736 $R --momentum 0.95 --stage-off --lr-scale 0.357143 --stop-after 16 --linearize-at-start
[ -f $A/path46/probe.json ] || $TR $A/path_probe.py $A/path46/probe.json $PD:46 k1=$A/path46/k1 k1c=$A/path46/k1c k4=$A/path46/k4 k4lin=$A/path46/k4lin k4d=$A/path46/k4d k4dlin=$A/path46/k4dlin k16=$A/path46/k16 k16lin=$A/path46/k16lin real47=$PD/scientific/kept/step000047_weights.pt > $A/path46/probe.log 2>&1
