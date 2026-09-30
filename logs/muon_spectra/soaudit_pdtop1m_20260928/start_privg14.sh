#!/bin/bash
cd /share/data/dl-theory/amin/projects/explore_muon
exec .venv/bin/python -u logs/muon_spectra/soaudit_pdtop1m_20260928/node_queue.py pdtop1m_privg14 \
  /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_warm4m_20260928/SPD_a0.5_b4M_lr0.02_momwarm0.8to0.9in120_s260925_l40s \
  PD_a0.5_lr0.01_s260925_l40s PDtop_a0.5_lr0.01_s260925_l40s
