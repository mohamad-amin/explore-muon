#!/bin/bash
cd /share/data/dl-theory/amin/projects/explore_muon
exec .venv/bin/python -u logs/muon_spectra/soaudit_pdtop_20260928/node_queue.py pdtop_privg14 \
  /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_batch16m_20260927/PD_a0.5_b16M_lr0.028_mom0.9_s260925_l40s \
  PDtop_a0.5_b16M_lr0.028_mom0.8_s260925_l40s PDtop_a0.5_b16M_lr0.028_mom0.9_s260925_l40s
