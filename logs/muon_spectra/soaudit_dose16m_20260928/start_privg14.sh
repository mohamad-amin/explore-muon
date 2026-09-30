#!/bin/bash
cd /share/data/dl-theory/amin/projects/explore_muon
exec .venv/bin/python -u logs/muon_spectra/soaudit_dose16m_20260928/node_queue.py dose16m_privg14 \
  /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_mom4mb08_20260928/SPD_a0.5_b4M_lr0.02_mom0.8_s260925_l40s \
  PD_a0.25_b16M_lr0.028_mom0.9_s260925_l40s PD_a0.25_b16M_lr0.028_mom0.8_s260925_l40s
