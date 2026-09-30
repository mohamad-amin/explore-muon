#!/bin/bash
cd /share/data/dl-theory/amin/projects/explore_muon
exec .venv/bin/python -u logs/muon_spectra/soaudit_mlr16m_20260928/node_queue.py mlr16m_privg14 \
  /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_clip16m_20260928/Mclip0.1_b16M_lr0.02_mom0.8_s260925_l40s \
  M_b16M_lr0.014_mom0.8_s260925_l40s M_b16M_lr0.028_mom0.8_s260925_l40s
