#!/bin/bash
# g20 (RTX 6000 Ada), inside an srun step with 4 GPUs: the 16M queue (after the audit's probes and anneals finish).
cd /share/data/dl-theory/amin/projects/explore_muon
exec .venv/bin/python -u logs/muon_spectra/soaudit_batch16m_20260927/node_queue.py batch16m_g20 \
  /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_strength4m_20260927/TS_a0.5_b0.5gn_b4M_lr0.02_mom0.9_s260925_ada \
  M_b16M_lr0.028_mom0.9_s260925_ada SPD_a0.5_b16M_lr0.04_mom0.9_s260925_ada TS_a0.5_b0.5gn_b16M_lr0.04_mom0.9_s260925_ada \
  M_b16M_lr0.04_mom0.9_s260925_ada SPD_a0.5_b16M_lr0.028_mom0.9_s260925_ada
