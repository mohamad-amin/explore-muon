#!/bin/bash
cd /share/data/dl-theory/amin/projects/explore_muon
exec .venv/bin/python -u logs/muon_spectra/soaudit_warmrep_20260928/node_queue.py warmrep_privg14 \
  /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_dose16m_20260928/PD_a0.25_b16M_lr0.028_mom0.8_s260925_l40s \
  PD_a0.5_b16M_T2x_lr0.028_momwarm0.8to0.9_s260925_l40s PD_a0.5_b16M_T2x_lr0.028_mom0.9_s260925_l40s
