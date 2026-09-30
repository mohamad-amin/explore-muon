#!/bin/bash
# g20 (RTX 6000 Ada): TS alpha 1/2 beta_out 1/2 with momentum 0.9 at 4M, LR 0.01 then 0.02, after the TS beta_out 1/2 arm.
cd /share/data/dl-theory/amin/projects/explore_muon
exec .venv/bin/python -u logs/muon_spectra/soaudit_strength4m_20260927/node_queue.py strength4m_g20 \
  /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_mom4m4_20260927/TS_a0.25_b0.5gn_b4M_lr0.01_mom0.9_s260925_ada \
  TS_a0.5_b0.5gn_b4M_lr0.01_mom0.9_s260925_ada TS_a0.5_b0.5gn_b4M_lr0.02_mom0.9_s260925_ada
