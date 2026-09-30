#!/bin/bash
# g20 queue (after the beta 0.81 PD run): PD and TS with momentum 0.9 at batch 4M.
cd /share/data/dl-theory/amin/projects/explore_muon
exec .venv/bin/python -u logs/muon_spectra/soaudit_mom4m3_20260927/node_queue.py mom4m3_g20 \
  /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_mom4m2_20260927/PD_b4M_lr0.02_mom0.81_s260925_ada \
  PD_b4M_lr0.02_mom0.9_s260925_ada TS_a0.25_b0.25gn_b4M_lr0.01_mom0.9_s260925_ada
