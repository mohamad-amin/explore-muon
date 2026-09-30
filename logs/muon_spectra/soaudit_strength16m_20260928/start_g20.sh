#!/bin/bash
cd /share/data/dl-theory/amin/projects/explore_muon
exec .venv/bin/python -u logs/muon_spectra/soaudit_strength16m_20260928/node_queue.py strength16m_g20 \
  /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_mom16mseed_20260928/SPD_a0.5_b16M_lr0.028_mom0.8_s260926_ada \
  SPD_a0.75_b16M_lr0.028_mom0.8_s260925_ada SPD_a0.5_b16M_lr0.04_mom0.8_s260925_ada
