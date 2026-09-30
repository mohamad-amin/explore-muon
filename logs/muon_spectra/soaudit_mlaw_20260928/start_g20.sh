#!/bin/bash
cd /share/data/dl-theory/amin/projects/explore_muon
exec .venv/bin/python -u logs/muon_spectra/soaudit_mlaw_20260928/node_queue.py mlaw_g20 \
  /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_pdtop1m_20260928/PD_a0.5_lr0.01_s260925_ada \
  SPD_a0.5_b16M_T2x_lr0.028_momwarm0.45to0.8in55_s260925_ada SPD_a0.5_b16M_T2x_lr0.028_mom0.7_s260925_ada PD_a0.5_b4M_lr0.02_mom0.85_s260925_ada
