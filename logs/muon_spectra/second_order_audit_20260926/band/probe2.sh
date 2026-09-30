#!/bin/bash
cd /share/data/dl-theory/amin/projects/explore_muon
A=logs/muon_spectra/second_order_audit_20260926
.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 $A/band_probe.py $A/band/probe2_control_46.json /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_batch16m_20260927/PD_a0.5_b16M_lr0.028_mom0.9_s260925_l40s:9 --harness $A/floor_ledger/control_46/kept/step000046.pt > $A/band/probe2_control_46.log 2>&1
