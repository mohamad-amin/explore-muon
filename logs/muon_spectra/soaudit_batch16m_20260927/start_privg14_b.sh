#!/bin/bash
# priv-g14 (L40S), second queue: Muon @0.014 (the first queue's best Muon LR, 0.02, was the bracket's bottom), after the first queue.
cd /share/data/dl-theory/amin/projects/explore_muon
exec .venv/bin/python -u logs/muon_spectra/soaudit_batch16m_20260927/node_queue.py batch16m_privg14_b \
  /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_batch16m_20260927/PD_a0.5_b16M_lr0.028_mom0.9_s260925_l40s \
  M_b16M_lr0.014_mom0.9_s260925_l40s
