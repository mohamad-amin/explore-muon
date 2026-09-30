#!/bin/bash
# Runs on a build VM created from explore-muon-v1. Mirrors the TTIC absolute paths so the
# existing controllers (which compare data manifests by resolved path and mtime) run unchanged.
# Shard bytes were verified against TTIC sha256 hashes when v1 was built.
set -euo pipefail
LL=/share/data/dl-theory/amin/projects/last_layer
EM=/share/data/dl-theory/amin/projects/explore_muon
sudo mkdir -p $LL/data $EM/data
sudo mv /opt/muon/data/fineweb10B $LL/data/fineweb10B
sudo ln -s $LL/data/fineweb10B /opt/muon/data/fineweb10B
sudo ln -s /opt/muon/.venv $LL/.venv
sudo ln -s $LL/data/fineweb10B $EM/data/fineweb10B
sudo ln -s ../last_layer/.venv $EM/.venv
sudo python3 - <<'EOF'
import os
d = "/share/data/dl-theory/amin/projects/last_layer/data/fineweb10B"
for line in open("/tmp/fineweb10B.mtimes"):
    name, mtime, size = line.split()
    path = os.path.join(d, name)
    assert os.path.getsize(path) == int(size), name
    os.utime(path, ns=(int(mtime), int(mtime)))
EOF
sudo chmod 444 $LL/data/fineweb10B/*.bin
sudo chmod 1777 $EM
cd $EM
/opt/muon/.venv/bin/python - <<'EOF'
import glob, json
from pathlib import Path
ref = json.load(open("/tmp/reference_manifests.json"))
def manifest(pattern):
    rows = []
    for path in [Path(p).resolve() for p in sorted(glob.glob(pattern))]:
        st = path.stat()
        rows.append({"path": str(path), "tokens": (st.st_size - 1024) // 2, "size": st.st_size, "mtime_ns": st.st_mtime_ns})
    return rows
train, val = manifest("data/fineweb10B/fineweb_train_*.bin"), manifest("data/fineweb10B/fineweb_val_*.bin")
assert train == ref["train_manifest"], "train manifest differs from TTIC reference"
assert val == ref["validation_manifest"], "validation manifest differs from TTIC reference"
print(f"MANIFESTS_MATCH: {len(train)} train, {len(val)} validation entries identical to the TTIC reference")
EOF
cp /tmp/fineweb10B.mtimes /opt/muon/ 2>/dev/null || sudo cp /tmp/fineweb10B.mtimes /opt/muon/
sudo rm -f /tmp/fineweb10B.mtimes /tmp/reference_manifests.json /tmp/mirror_ttic_paths.sh
sync
echo MIRROR_DONE
