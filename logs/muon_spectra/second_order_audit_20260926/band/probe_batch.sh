#!/bin/bash
# Why the coordination gain grows with batch: the correction's size/alignment (correction_probe.py) and the step's
# coherence and one-step effect (band_probe.py) at real kept PD alpha 1/2 states at 1M, 4M and 16M (MUON_CASE 19:5x CDT).
cd /share/data/dl-theory/amin/projects/explore_muon
A=logs/muon_spectra/second_order_audit_20260926
TR=".venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4"
for spec in "1m_900|/share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_pdtop1m_20260928/PD_a0.5_lr0.01_s260925_ada:900|0.95" "4m_183|/share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_mom4m4_20260927/PD_a0.5_b4M_lr0.02_mom0.9_s260925_ada:183|0.9" "16m_46|/share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_batch16m_20260927/PD_a0.5_b16M_lr0.028_mom0.9_s260925_l40s:46|0.9"; do
  IFS='|' read tag item beta <<< "$spec"
  $TR $A/correction_probe.py $A/band/batchprobe_correction_$tag.json $item --beta $beta > $A/band/batchprobe_correction_$tag.log 2>&1
  $TR $A/band_probe.py $A/band/batchprobe_band_$tag.json $item --beta $beta > $A/band/batchprobe_band_$tag.log 2>&1
done
