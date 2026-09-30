#!/bin/bash
cd /share/data/dl-theory/amin/projects/explore_muon
exec .venv/bin/python -u logs/muon_spectra/soaudit_snrgate_large_20260928/node_queue.py snrgate_privg14 \
  /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_mlaw_20260928/SPD_a0.5_b4M_lr0.02_mom0.85_s260925_l40s \
  PDgate_a0.5_b16M_lr0.028_mom0.9_s260925_l40s PDgate_a0.5_b16M_lr0.028_mom0.8_s260925_l40s
