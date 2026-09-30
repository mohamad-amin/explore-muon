#!/bin/bash
cd /share/data/dl-theory/amin/projects/explore_muon
exec .venv/bin/python -u logs/muon_spectra/soaudit_b16mlong_20260928/node_queue.py b16mlong_privg14 \
  /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_strength4m_20260927/SPD_a0.5_b4M_lr0.02_mom0.9_s260925_l40s \
  M_b16M_T2x_lr0.02_mom0.9_s260925_l40s
