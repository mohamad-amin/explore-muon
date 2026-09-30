#!/bin/bash
# priv-g14 queue of the 4M momentum / clipping / Nesterov test (second-order audit, 2026-09-27).
cd /share/data/dl-theory/amin/projects/explore_muon
exec .venv/bin/python -u logs/muon_spectra/soaudit_mom4m_20260927/node_queue.py mom4m_privg14 \
  /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_batch4m_20260927/M_b4M_lr0.028_s260925_l40s \
  M_b4M_lr0.014_mom0.81_s260925_l40s SPD_b4M_lr0.01_noclip_s260925_l40s M_b4M_lr0.014_mom0.9_s260925_l40s \
  M_b4M_lr0.014_nesterov_s260925_l40s M_b4M_lr0.014_noclip_s260925_l40s
