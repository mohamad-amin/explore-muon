#!/bin/bash
# g20 queue (after the two-sided 4M bracket): PD and TS with momentum 0.81 at batch 4M.
cd /share/data/dl-theory/amin/projects/explore_muon
exec .venv/bin/python -u logs/muon_spectra/soaudit_mom4m2_20260927/node_queue.py mom4m2_g20 \
  /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_ts4m_20260927/TS_a0.25_b0.25gn_b4M_lr0.01_s260925_ada \
  TS_a0.25_b0.25gn_b4M_lr0.02_mom0.81_s260925_ada PD_b4M_lr0.02_mom0.81_s260925_ada
