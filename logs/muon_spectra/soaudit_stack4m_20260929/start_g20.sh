#!/bin/bash
cd /share/data/dl-theory/amin/projects/explore_muon
exec .venv/bin/python -u logs/muon_spectra/soaudit_stack4m_20260929/node_queue.py stack_g20 \
  /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_snrgate_large_20260928/PDgate_a0.5_b4M_lr0.02_mom0.9_s260925_ada \
  PDgate_a0.5_b4M_lr0.02_momwarm0.8to0.9in120_s260925_ada
