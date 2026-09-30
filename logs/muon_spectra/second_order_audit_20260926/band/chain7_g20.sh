#!/bin/bash
# 1M rung of the batch ladder: PD alpha 1/2 1M (beta 0.95, LR 0.01) harness control from @200 to 1470 (MUON_CASE 16:0x CDT).
cd /share/data/dl-theory/amin/projects/explore_muon
A=logs/muon_spectra/second_order_audit_20260926
.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 $A/newton_train.py $A/band/control_1m /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_pdtop1m_20260928/PD_a0.5_lr0.01_s260925_ada:200 --normalize 0 --momentum 0.95 --clip 1.0 --staged-pd 0.5 --stage-off --validation-every 50 --sharpness-every 100 --checkpoint-every 1000 > $A/band/control_1m.log 2>&1
