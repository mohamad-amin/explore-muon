#!/bin/bash
# Runs on the image-build VM. Installs the pinned environment and FineWeb shards under /opt/muon.
# Inputs copied to /tmp beforehand: requirements.lock (pip freeze of the TTIC .venv), fineweb10B.sha256.
set -euo pipefail
ROOT=/opt/muon
sudo mkdir -p $ROOT && sudo chown "$(id -u):$(id -g)" $ROOT

echo "== system packages =="
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq build-essential tmux rsync >/dev/null

echo "== uv and Python 3.11 =="
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=$ROOT/bin UV_NO_MODIFY_PATH=1 sh
export PATH=$ROOT/bin:$PATH UV_PYTHON_INSTALL_DIR=$ROOT/python UV_CACHE_DIR=/tmp/uv-cache
uv python install 3.11.2 || uv python install 3.11
uv venv $ROOT/.venv --python 3.11
uv pip install --python $ROOT/.venv/bin/python -r /tmp/requirements.lock

echo "== FineWeb shards (kjj0/fineweb10B-gpt2) =="
mkdir -p $ROOT/data/fineweb10B
$ROOT/.venv/bin/python - <<'EOF'
from huggingface_hub import hf_hub_download
names = ["fineweb_val_000000.bin"] + [f"fineweb_train_{i:06d}.bin" for i in range(1, 33)]
for name in names:
    hf_hub_download(repo_id="kjj0/fineweb10B-gpt2", filename=name, repo_type="dataset",
                    local_dir="/opt/muon/data/fineweb10B")
EOF
(cd $ROOT/data/fineweb10B && sha256sum -c --quiet /tmp/fineweb10B.sha256 && echo "all 33 shards match TTIC hashes")
cp /tmp/requirements.lock /tmp/fineweb10B.sha256 $ROOT/

echo "== environment record =="
$ROOT/.venv/bin/python - <<'EOF' | tee $ROOT/ENVIRONMENT.txt
import platform, sys, torch, numpy, matplotlib
print("python", sys.version.split()[0])
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("numpy", numpy.__version__, "matplotlib", matplotlib.__version__)
print("os", platform.platform())
EOF
rm -rf /tmp/uv-cache
sudo apt-get clean
echo "BUILD_DONE"
