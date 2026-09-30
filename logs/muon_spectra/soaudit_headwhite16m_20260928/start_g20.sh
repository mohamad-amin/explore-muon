#!/bin/bash
cd /share/data/dl-theory/amin/projects/explore_muon
exec .venv/bin/python -u logs/muon_spectra/soaudit_headwhite16m_20260928/node_queue.py headwhite_g20 \
  /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_prefilter16m_20260928/SPD_a0.5_b16M_lr0.028_mom0.8_s260925_ada \
  SPDhw0.5_a0.5_b16M_lr0.028_mom0.9_s260925_ada SPDhw0.25_a0.5_b16M_lr0.028_mom0.9_s260925_ada
