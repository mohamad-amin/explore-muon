#!/bin/bash
cd /share/data/dl-theory/amin/projects/explore_muon
exec .venv/bin/python -u logs/muon_spectra/soaudit_prefilter16m_20260928/node_queue.py prefilter_g20 \
  /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_batch16m_20260927/SPD_a0.5_b16M_lr0.028_mom0.9_s260925_ada \
  SPD2tap_a0.5_b16M_lr0.028_mom0.8_s260925_ada SPD2tap_a0.5_b16M_lr0.028_mom0.9_s260925_ada SPD_a0.5_b16M_lr0.028_mom0.8_s260925_ada
