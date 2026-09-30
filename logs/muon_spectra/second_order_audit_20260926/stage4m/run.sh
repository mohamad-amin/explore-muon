#!/bin/bash
# Where does staging-on-momentum flip? 4M states (2026-09-29 02:52 CDT), two GPUs of priv-g14.
cd /share/data/dl-theory/amin/projects/explore_muon
PY=.venv/bin/python
CUDA_VISIBLE_DEVICES=0 $PY -u /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/second_order_audit_20260926/one_step_blockgn.py /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/second_order_audit_20260926/stage4m/PD_mom4M.json /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_mom4m4_20260927/PD_a0.5_b4M_lr0.02_mom0.9_s260925_ada:183 --input momentum4M --momentum-c 0.9 --partitions full,gsmap > /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/second_order_audit_20260926/stage4m/PD_mom4M.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY -u /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/second_order_audit_20260926/one_step_blockgn.py /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/second_order_audit_20260926/stage4m/M_mom4M.json /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_batch4m_20260927/M_b4M_lr0.014_s260925_l40s:183 --input momentum4M --momentum-c 0.95 --partitions full,gsmap > /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/second_order_audit_20260926/stage4m/M_mom4M.log 2>&1 &
wait
