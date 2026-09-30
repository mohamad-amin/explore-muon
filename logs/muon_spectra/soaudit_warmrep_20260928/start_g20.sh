#!/bin/bash
cd /share/data/dl-theory/amin/projects/explore_muon
exec .venv/bin/python -u logs/muon_spectra/soaudit_warmrep_20260928/node_queue.py warmrep_g20 \
  /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_momwarm16m_20260928/SPD_a0.5_b16M_lr0.028_momwarm0.8to0.9_s260925_ada \
  SPD_a0.5_b16M_T2x_lr0.028_momwarm0.8to0.9_s260926_ada SPD_a0.5_b16M_T2x_lr0.028_mom0.9_s260926_ada
