#!/bin/bash
cd /share/data/dl-theory/amin/projects/explore_muon
exec .venv/bin/python -u logs/muon_spectra/soaudit_momwarm16m_20260928/node_queue.py momwarm16m_g20 \
  /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_mom4mb08_20260928/PD_a0.5_b4M_lr0.02_mom0.8_s260925_ada \
  SPD_a0.5_b16M_T2x_lr0.028_momwarm0.8to0.9_s260925_ada SPD_a0.5_b16M_lr0.028_momwarm0.8to0.9_s260925_ada
