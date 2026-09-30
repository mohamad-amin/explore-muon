#!/bin/bash
cd /share/data/dl-theory/amin/projects/explore_muon
exec .venv/bin/python -u logs/muon_spectra/soaudit_mom16m_20260928/node_queue.py mom16m_privg14 \
  /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_headwhite16m_20260928/PDhw0.25_a0.5_b16M_lr0.028_mom0.9_s260925_l40s \
  PD_a0.5_b16M_lr0.028_mom0.8_s260925_l40s M_b16M_lr0.02_mom0.8_s260925_l40s
