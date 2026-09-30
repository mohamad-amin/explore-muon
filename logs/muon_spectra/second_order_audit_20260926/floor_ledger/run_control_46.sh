#!/bin/bash
# Floor ledger (MUON_CASE 2026-09-29 07:19 CDT): layer-staged PD control regenerated from 16M PD @9 to step 46 (kept), then a
# 16-step linear anneal in the harness (steps 47-62), validation every 4 steps and at 46, sharpness every 8 steps.
cd /share/data/dl-theory/amin/projects/explore_muon
.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 logs/muon_spectra/second_order_audit_20260926/newton_train.py logs/muon_spectra/second_order_audit_20260926/floor_ledger/control_46 /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_batch16m_20260927/PD_a0.5_b16M_lr0.028_mom0.9_s260925_l40s:9 --normalize 0 --momentum 0.9 --clip 1.0 --staged-pd 0.5 --stage-off --validation-every 4 --sharpness-every 8 --stop-after 53 --checkpoint-every 1000 --keep 46 --anneal-from 46 --anneal-steps 16 > logs/muon_spectra/second_order_audit_20260926/floor_ledger/control_46.log 2>&1
