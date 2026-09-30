#!/bin/bash
# One step-vs-path probe set (MUON_CASE 2026-09-30 11:40 / 12:30 CDT): usage run_path_set.sh OUT_DIR ITEM16 ITEM4 ITEM16X [REAL_NEXT_WEIGHTS]
# ITEM16: the 16M start (ARM:STEP); ITEM4 / ITEM16X: its 4M / 1M relabelled starts. Same options as path46.
cd /share/data/dl-theory/amin/projects/explore_muon
A=logs/muon_spectra/second_order_audit_20260926
O=$1; S16=$2; S4=$3; S1=$4; REAL=$5
TR=".venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4"
R="--normalize 0 --clip 1.0 --staged-pd 0.5 --fixed-root 64 --freeze-aux --validation-every 1000 --checkpoint-every 1000"
run() { local out=$1; shift; [ -f $O/$out/done.json ] || $TR $A/newton_train.py $O/$out "$@" > $O/$out.log 2>&1; }
run k1 $S16 $R --momentum 0.8 --stage-off --stop-after 1
run k1c $S16 $R --momentum 0.8 --stage-band top --stage-granularity sublayer --stop-after 1
run k4 $S4 $R --momentum 0.9 --stage-off --lr-scale 0.714286 --stop-after 4
run k4lin $S4 $R --momentum 0.9 --stage-off --lr-scale 0.714286 --stop-after 4 --linearize-at-start
run k4d $S4 $R --momentum 0.945742 --stage-off --lr-scale 0.25 --stop-after 4
run k4dlin $S4 $R --momentum 0.945742 --stage-off --lr-scale 0.25 --stop-after 4 --linearize-at-start
run k16 $S1 $R --momentum 0.95 --stage-off --lr-scale 0.357143 --stop-after 16
run k16lin $S1 $R --momentum 0.95 --stage-off --lr-scale 0.357143 --stop-after 16 --linearize-at-start
EXTRA=""; [ -n "$REAL" ] && EXTRA="realnext=$REAL"
[ -f $O/probe.json ] || $TR $A/path_probe.py $O/probe.json $S16 k1=$O/k1 k1c=$O/k1c k4=$O/k4 k4lin=$O/k4lin k4d=$O/k4d k4dlin=$O/k4dlin k16=$O/k16 k16lin=$O/k16lin $EXTRA > $O/probe.log 2>&1
