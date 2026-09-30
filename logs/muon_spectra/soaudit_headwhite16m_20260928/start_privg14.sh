#!/bin/bash
cd /share/data/dl-theory/amin/projects/explore_muon
exec .venv/bin/python -u logs/muon_spectra/soaudit_headwhite16m_20260928/node_queue.py headwhite_privg14 \
  /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_prefilter16m_20260928/PD2tap_a0.5_b16M_lr0.028_mom0.9_s260925_l40s \
  PDhw0.5_a0.5_b16M_lr0.028_mom0.9_s260925_l40s PDhw0.25_a0.5_b16M_lr0.028_mom0.9_s260925_l40s
