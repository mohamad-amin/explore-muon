#!/bin/bash
# usage: run_node.sh FIRST_QUEUE  (runs queues FIRST..FIRST+3 on GPUs 0..3 of this node)
set -u
cd /share/data/dl-theory/amin/projects/explore_muon
A=logs/muon_spectra/second_order_audit_20260926
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
for g in 0 1 2 3; do
  q=$(( $1 + g ))
  (
    while read -r item; do
      [ -z "$item" ] && continue
      tag=$(echo "$item" | sed 's#.*/muon_spectra/##; s#/#__#g; s#:#__#')
      [ -f "$A/step_profile/$tag.json" ] && continue
      CUDA_VISIBLE_DEVICES=$g .venv/bin/python $A/step_profile_probe.py "$A/step_profile/$tag.json" "$item" > "$A/step_profile/$tag.log" 2>&1 || echo "FAILED $item" >> "$A/step_profile/failures.txt"
    done < "$A/step_profile/queue_$q.txt"
    echo "queue $q done" >> "$A/step_profile/progress.txt"
  ) &
done
wait
