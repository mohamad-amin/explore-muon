#!/bin/bash
# g20 queue: two-sided data norm at batch 4M, paired with PD alpha 1/4's Ada bracket in ../soaudit_batch4m_20260927.
cd /share/data/dl-theory/amin/projects/explore_muon
exec .venv/bin/python -u logs/muon_spectra/soaudit_ts4m_20260927/node_queue.py ts4m_g20 \
  /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_batch4m_20260927/M_b4M_lr0.028_s260925_l40s \
  TS_a0.25_b0.25gn_b4M_lr0.02_s260925_ada TS_a0.25_b0.25gn_b4M_lr0.01_s260925_ada
