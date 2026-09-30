#!/bin/bash
# priv-g14: a second seed of the 4M comparison at tuned momentum (0.9), all on L40S.
cd /share/data/dl-theory/amin/projects/explore_muon
exec .venv/bin/python -u logs/muon_spectra/soaudit_seed4m_20260927/node_queue.py seed4m_privg14 \
  /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_mom4m3_20260927/M_b4M_lr0.02_mom0.9_s260925_l40s \
  M_b4M_lr0.014_mom0.9_s260926_l40s TS_a0.25_b0.25gn_b4M_lr0.01_mom0.9_s260926_l40s PD_b4M_lr0.02_mom0.9_s260926_l40s SPD_b4M_lr0.01_mom0.9_s260926_l40s
