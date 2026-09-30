#!/bin/bash
# priv-g14 (L40S): S∘PD alpha 1/2 with momentum 0.9 at 4M, LR 0.02 then 0.01 (strength-principle test).
cd /share/data/dl-theory/amin/projects/explore_muon
exec .venv/bin/python -u logs/muon_spectra/soaudit_strength4m_20260927/node_queue.py strength4m_privg14 \
  /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_seed4m_20260927/SPD_b4M_lr0.01_mom0.9_s260926_l40s \
  SPD_a0.5_b4M_lr0.02_mom0.9_s260925_l40s SPD_a0.5_b4M_lr0.01_mom0.9_s260925_l40s
