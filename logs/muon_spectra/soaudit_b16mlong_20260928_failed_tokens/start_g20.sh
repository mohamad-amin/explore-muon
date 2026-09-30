#!/bin/bash
cd /share/data/dl-theory/amin/projects/explore_muon
exec .venv/bin/python -u logs/muon_spectra/soaudit_b16mlong_20260928/node_queue.py b16mlong_g20 \
  /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_strength4m_20260927/SPD_a0.5_b4M_lr0.02_mom0.9_s260925_l40s \
  SPD_a0.5_b16M_T4x_lr0.028_mom0.9_s260925_ada
