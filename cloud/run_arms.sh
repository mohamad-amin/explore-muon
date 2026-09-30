#!/bin/bash
# Run arms on one fresh Spot VM, collect the results into logs/, then delete the VM.
#   cloud/run_arms.sh VM COHORT ARM [ARM ...]
# COHORT is relative to the project root (logs/muon_spectra/<cohort>, with frozen/ and frozen_manifest.json).
# Arms run one after another on 4 GPUs. MACHINE (default a3-highgpu-4g) and ZONES (tried in order) override.
# If the Spot VM is preempted, the script reports it and leaves the stopped VM for inspection.
set -uo pipefail
cd "$(dirname "$0")/.."
G=${GCLOUD:-/share/data/dl-theory/amin/tools/google-cloud-sdk/bin/gcloud}
R=/share/data/dl-theory/amin/projects/explore_muon
vm=$1; cohort=$2; shift 2; arms=("$@")
machine=${MACHINE:-a3-highgpu-4g}
zones=${ZONES:-"us-east4-b us-east4-c us-east4-a us-central1-a us-central1-b us-central1-c us-west1-a us-west1-b"}
ref=logs/muon_spectra/depth8_w512_nobias_rms_qk_20260925
log() { echo "[$(date -u +%H:%M:%S) $vm] $*"; }
status() { $G compute instances list --filter="name=$vm" --format="value(status)"; }

for zone in $zones; do
  cloud/gcp.sh up "$vm" "$machine" "$zone" >/dev/null 2>&1
  [ -n "$(status)" ] && { log "created $machine in $zone"; break; }
done
[ -n "$(status)" ] || { log "no Spot capacity in: $zones"; exit 1; }
for i in $(seq 1 30); do cloud/gcp.sh ssh "$vm" true 2>/dev/null && break; sleep 10; done

paths=(cloud/lane.py "$cohort/frozen" "$cohort/frozen_manifest.json" "$ref/muon/scientific/metadata.json" "$ref/config_muon.json")
for arm in "${arms[@]}"; do paths+=("$cohort/$arm"); done
cloud/gcp.sh push "$vm" "${paths[@]}" || { log "push failed; deleting"; cloud/gcp.sh down "$vm"; exit 1; }
cloud/gcp.sh ssh "$vm" "tmux new -d -s lane 'cd $R && .venv/bin/python cloud/lane.py $cohort $vm 0,1,2,3 ${arms[*]} > $cohort/lane_$vm.log 2>&1'"
log "started: ${arms[*]}"

while sleep 60; do
  vm_status=$(status)
  if [ "$vm_status" != RUNNING ]; then log "VM is $vm_status (preempted?); left for inspection"; exit 2; fi
  state=$(cloud/gcp.sh ssh "$vm" "python3 -c \"import json; d=json.load(open('$R/$cohort/lane_$vm.json')); print('done' if 'ended_unix' in d else 'running')\"" 2>/dev/null)
  [ "$state" = done ] && break
done
for arm in "${arms[@]}"; do cloud/gcp.sh pull "$vm" "$cohort/$arm"; done
cloud/gcp.sh pull "$vm" "$cohort/lane_$vm.json" "$cohort/lane_$vm.log"
log "collected: $(python3 -c "import json; print(' '.join(f\"{r['arm']}={r['exit']}\" for r in json.load(open('$cohort/lane_$vm.json'))['runs']))")"
cloud/gcp.sh down "$vm" >/dev/null 2>&1 && log "deleted"
