#!/bin/bash
# Global-logit-shift split of the coupling (2026-09-29 01:57 CDT), one GPU.
cd /share/data/dl-theory/amin/projects/explore_muon
L=/share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra
A=$L/second_order_audit_20260926
B16=$L/soaudit_batch16m_20260927
for item in M16M_46=$B16/M_b16M_lr0.02_mom0.9_s260925_l40s:46 PD16M_46=$B16/PD_a0.5_b16M_lr0.028_mom0.9_s260925_l40s:46 SPD16M_46=$B16/SPD_a0.5_b16M_lr0.028_mom0.9_s260925_ada:46 M1M_100=$L/soaudit_traj_20260926/M_lr0.007_s260925_l40s:100 M1M_900=$L/soaudit_traj_20260926/M_lr0.007_s260925_l40s:900 PD1M_900=$L/soaudit_traj_20260926/PD_a0.25_lr0.01_s260925_ada:900 SPD1M_900=$L/soaudit_traj_20260926/SPD_a0.25_lr0.01_s260925_ada:900; do
  tag=${item%%=*}; target=${item#*=}
  .venv/bin/python -u $A/step_coupling_mean_probe.py $A/step_coupling_mean/$tag.json $target --sequences 24 > $A/step_coupling_mean/$tag.log 2>&1
  echo "$tag exit $?"
done
