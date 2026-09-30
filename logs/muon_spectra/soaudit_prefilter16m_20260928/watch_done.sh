#!/bin/bash
# exits (printing each arm's state) when every arm is complete or any arm's controller reports a failure
C=/share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_prefilter16m_20260928
while true; do
  done_n=0; fail=""
  for a in SPD2tap_a0.5_b16M_lr0.028_mom0.8_s260925_ada SPD2tap_a0.5_b16M_lr0.028_mom0.9_s260925_ada SPD_a0.5_b16M_lr0.028_mom0.8_s260925_ada M2tap_b16M_lr0.02_mom0.9_s260925_l40s PD2tap_a0.5_b16M_lr0.028_mom0.9_s260925_l40s; do
    grep -q '"status": "complete"' $C/$a/scientific/status.json 2>/dev/null && done_n=$((done_n+1))
    grep -q -i -E "traceback|failed|error" $C/$a/controller.log 2>/dev/null && fail="$fail $a"
  done
  if [ $done_n -ge 5 ] || [ -n "$fail" ]; then echo "complete=$done_n failed:$fail"; exit 0; fi
  sleep 60
done
