#!/bin/bash
# Re-run the speedup analysis each time one of the pending runs finishes (2x 16M pair, SOAP-side 16M pair).
L=/share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra; D=$L/second_order_audit_20260926
ARMS="soaudit_b16mlong_20260928/M_b16M_T2x_lr0.02_mom0.9_s260925_l40s soaudit_b16mlong_20260928/SPD_a0.5_b16M_T2x_lr0.028_mom0.9_s260925_ada soaudit_soapside16m_20260928/SleftPD_a0.5_b16M_lr0.028_mom0.9_s260925_ada soaudit_soapside16m_20260928/SrightPD_a0.5_b16M_lr0.028_mom0.9_s260925_l40s"
declare -A done_
for i in $(seq 1 600); do
  changed=0; all=1
  for a in $ARMS; do
    p=$(python3 -c "import json; print(json.load(open('$L/$a/pipeline.json'))['phase'])" 2>/dev/null)
    if [ "$p" = "complete" ] || [ "$p" = "failed" ]; then [ -z "${done_[$a]}" ] && { done_[$a]=1; changed=1; echo "$(date +%H:%M) $a $p"; }; else all=0; fi
  done
  if [ $changed = 1 ]; then cd $D && /share/data/dl-theory/amin/projects/explore_muon/.venv/bin/python speedup_by_batch.py speedup_by_batch.json figures_floor/speedup_by_batch.png 2>&1 | grep -v "Warning\|legend" ; fi
  [ $all = 1 ] && break
  sleep 60
done
echo "$(date +%H:%M) auto_speedup finished"
