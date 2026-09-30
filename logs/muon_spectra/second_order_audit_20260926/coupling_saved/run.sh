#!/bin/bash
cd /share/data/dl-theory/amin/projects/explore_muon
PY=.venv/bin/python
for spec in "M1M500_g4M /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_traj_20260926/M_lr0.007_s260925_l40s:500 logs/muon_spectra/second_order_audit_20260926/gn/M_lr0.007_s260925_l40s_step000500_directions.pt g4M" "M1M500_mom /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_traj_20260926/M_lr0.007_s260925_l40s:500 logs/muon_spectra/second_order_audit_20260926/gn/M_lr0.007_s260925_l40s_step000500_directions.pt momentum" "PD1M500_g4M /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_traj_20260926/PD_a0.25_lr0.01_s260925_ada:500 logs/muon_spectra/second_order_audit_20260926/gn/PD_a0.25_lr0.01_s260925_ada_step000500_directions.pt g4M" "M4M183_g4M /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_batch4m_20260927/M_b4M_lr0.014_s260925_l40s:183 logs/muon_spectra/second_order_audit_20260926/gn/M_b4M_lr0.014_s260925_l40s_step000183_directions.pt g4M"; do
  set -- $spec
  $PY -u logs/muon_spectra/second_order_audit_20260926/direction_coupling_saved.py logs/muon_spectra/second_order_audit_20260926/coupling_saved/$1.json $2 --directions $3 --input $4 > logs/muon_spectra/second_order_audit_20260926/coupling_saved/$1.log 2>&1
  echo "$1 exit $?"
done
