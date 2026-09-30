#!/bin/bash
# Spot GPU VMs from the explore-muon image (project ucsd-hdsi-tianhaowang, us-central1).
# VMs mirror the TTIC paths, so project paths are identical on both sides. Run from anywhere.
#
#   cloud/gcp.sh up NAME [MACHINE] [ZONE]   create a Spot VM (default a3-highgpu-4g = 4xH100, us-central1-a)
#   cloud/gcp.sh ssh NAME [COMMAND...]      shell, or run a command
#   cloud/gcp.sh push NAME PATH...          copy project paths (relative to the project root) to the VM
#   cloud/gcp.sh pull NAME PATH...          copy them back (adds/updates files, never deletes)
#   cloud/gcp.sh down NAME                  delete the VM and its disk
#   cloud/gcp.sh ls                         list VMs in the project
#
# Spot VMs stop (not delete) on preemption or after MAX_HOURS (default 12), so their disk survives.
set -euo pipefail
G=${GCLOUD:-/share/data/dl-theory/amin/tools/google-cloud-sdk/bin/gcloud}
REPO=/share/data/dl-theory/amin/projects/explore_muon
KEY=$HOME/.ssh/google_compute_engine
SSH_OPTS=(-i "$KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR)

zone_of() { $G compute instances list --filter="name=$1" --format="value(zone.basename())"; }
ip_of() { $G compute instances describe "$1" --zone="$(zone_of "$1")" --format="value(networkInterfaces[0].accessConfigs[0].natIP)"; }

cmd=${1:-}; shift || true
case "$cmd" in
  up)
    name=$1; machine=${2:-a3-highgpu-4g}; zone=${3:-us-central1-a}
    case "$machine" in a3-ultragpu*|a4-*) disk=hyperdisk-balanced ;; *) disk=pd-balanced ;; esac
    # A3/A4 (and A2 ultra) include local SSDs; nothing is kept on them, so discard them on stop.
    lssd=(); case "$machine" in *-nolssd) ;; a3-*|a4-*|a2-ultragpu*) lssd=(--discard-local-ssds-at-termination-timestamp=true) ;; esac
    $G compute instances create "$name" --zone="$zone" --machine-type="$machine" "${lssd[@]}" \
      --image-family=explore-muon --boot-disk-size=200GB --boot-disk-type="$disk" \
      --network-interface=nic-type=GVNIC --maintenance-policy=TERMINATE \
      --provisioning-model=SPOT --instance-termination-action=STOP --max-run-duration="${MAX_HOURS:-12}h" \
      --labels=owner=mohamadamin,study=explore-muon \
      --metadata=ssh-keys="$USER:$(cat "$KEY.pub")"
    ;;
  ssh)
    name=$1; shift
    ssh "${SSH_OPTS[@]}" "$USER@$(ip_of "$name")" "$@"
    ;;
  push)
    name=$1; shift; ip=$(ip_of "$name")
    for path in "$@"; do
      ssh "${SSH_OPTS[@]}" "$USER@$ip" mkdir -p "$REPO/$(dirname "$path")"
      rsync -a -e "ssh ${SSH_OPTS[*]}" "$REPO/$path" "$USER@$ip:$REPO/$(dirname "$path")/"
    done
    ;;
  pull)
    name=$1; shift; ip=$(ip_of "$name")
    for path in "$@"; do
      mkdir -p "$REPO/$(dirname "$path")"
      rsync -a -e "ssh ${SSH_OPTS[*]}" "$USER@$ip:$REPO/$path" "$REPO/$(dirname "$path")/"
    done
    ;;
  down)
    $G compute instances delete "$1" --zone="$(zone_of "$1")" --quiet
    ;;
  ls)
    $G compute instances list --format="table(name,zone.basename(),machineType.basename(),status,scheduling.provisioningModel,labels.owner)"
    ;;
  *)
    sed -n '2,13p' "$0"; exit 1
    ;;
esac
