#!/bin/bash
# Review's diagnostic (MUON_CASE 2026-09-30 13:50 CDT), arms A (one 16M PD step per batch) and B (four true 4M steps per batch),
# aux frozen, lambda_max at every batch start, 20 batches; then E's dose-0 control (four identical inner steps).
cd /share/data/dl-theory/amin/projects/explore_muon
A=logs/muon_spectra/second_order_audit_20260926
PD=/share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_mom16m_20260928/PD_a0.5_b16M_lr0.028_mom0.8_s260925_l40s
TR=".venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4"
R="--normalize 0 --clip 1.0 --staged-pd 0.5 --checkpoint-every 1000 --freeze-aux --sharpness-from-start"
run() { local out=$1; shift; [ -f $A/path46/diag2/$out/done.json ] || $TR $A/newton_train.py $A/path46/diag2/$out "$@" > $A/path46/diag2/$out.log 2>&1; }
run B $A/path46/start_k4:184 $R --momentum 0.9 --stage-off --lr-scale 0.714286 --stop-after 80 --validation-every 4 --sharpness-every 4
run A $PD:46 $R --momentum 0.8 --stage-off --stop-after 20 --validation-every 1 --sharpness-every 1
run E0 $PD:46 --normalize 0 --clip 1.0 --staged-pd 0.5 --checkpoint-every 1000 --momentum 0.8 --stage-off --stop-after 20 --validation-every 1 --inner-steps 4 --inner-lr-scale 0.714286 --inner-dose 0
