#!/bin/bash
# g20 queue (after the clip 0.1 tests): stronger preconditioning with fresher momentum (beta 0.9) at batch 4M.
cd /share/data/dl-theory/amin/projects/explore_muon
exec .venv/bin/python -u logs/muon_spectra/soaudit_mom4m4_20260927/node_queue.py mom4m4_g20 \
  /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_clip4m_20260927/M_b4M_lr0.014_clip0.1_s260925_ada \
  PD_a0.5_b4M_lr0.02_mom0.9_s260925_ada TS_a0.25_b0.5gn_b4M_lr0.01_mom0.9_s260925_ada
