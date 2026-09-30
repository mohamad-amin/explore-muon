#!/bin/bash
# priv-g14 queue of the 4M alpha test (the audit's GN measurements there have finished).
cd /share/data/dl-theory/amin/projects/explore_muon
exec .venv/bin/python -u logs/muon_spectra/soaudit_alpha4m_20260927/node_queue.py a4m_privg14 \
  /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_batch4m_20260927/M_b4M_lr0.028_s260925_l40s \
  SPD_a0.5_b4M_lr0.01_s260925_l40s SPD_a0.25_b4M_lr0.01_s260925_l40s SPD_a0.5_b4M_lr0.02_s260925_l40s
