#!/bin/bash
# priv-g14 queue (after the momentum sweep): S∘PD and Muon (LR 0.02) with momentum 0.9 at batch 4M.
cd /share/data/dl-theory/amin/projects/explore_muon
exec .venv/bin/python -u logs/muon_spectra/soaudit_mom4m3_20260927/node_queue.py mom4m3_privg14 \
  /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_mom4m_20260927/M_b4M_lr0.014_noclip_s260925_l40s \
  SPD_b4M_lr0.01_mom0.9_s260925_l40s M_b4M_lr0.02_mom0.9_s260925_l40s
