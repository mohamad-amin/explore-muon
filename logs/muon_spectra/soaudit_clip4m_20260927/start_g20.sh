#!/bin/bash
# g20 queue (after the beta 0.81 combinations): gradients normalized before momentum (clip 0.1) at batch 4M.
cd /share/data/dl-theory/amin/projects/explore_muon
exec .venv/bin/python -u logs/muon_spectra/soaudit_clip4m_20260927/node_queue.py clip4m_g20 \
  /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_mom4m3_20260927/TS_a0.25_b0.25gn_b4M_lr0.01_mom0.9_s260925_ada \
  PD_b4M_lr0.02_clip0.1_s260925_ada M_b4M_lr0.014_clip0.1_s260925_ada
