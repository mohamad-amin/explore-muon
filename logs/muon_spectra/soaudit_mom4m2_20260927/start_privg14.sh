#!/bin/bash
# priv-g14 queue (after the momentum sweep): Muon beta 0.81 at LR 0.02 and S∘PD with beta 0.81, batch 4M.
cd /share/data/dl-theory/amin/projects/explore_muon
exec .venv/bin/python -u logs/muon_spectra/soaudit_mom4m2_20260927/node_queue.py mom4m2_privg14 \
  /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_mom4m_20260927/M_b4M_lr0.014_noclip_s260925_l40s \
  M_b4M_lr0.02_mom0.81_s260925_l40s SPD_b4M_lr0.01_mom0.81_s260925_l40s
