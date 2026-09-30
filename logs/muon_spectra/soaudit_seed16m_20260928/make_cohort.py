"""Freeze the live source into a new cohort and create arm directories (config.json, tiny data, job.sbatch).

usage: make_cohort.py COHORT_DIR ARMS_JSON
ARMS_JSON maps arm name -> {"change": {...config overrides...}, "sbatch": true/false}
Refuses to touch an existing cohort directory.
"""
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path("/share/data/dl-theory/amin/projects/explore_muon")
sys.path.insert(0, str(REPO))
from research.adamw_spectra.train import load_config

cohort = REPO / sys.argv[1]
arms = json.loads(Path(sys.argv[2]).read_text())
V = REPO / "logs/muon_spectra/depth8_w512_nobias_rms_qk_20260925"
if cohort.exists():
    raise SystemExit(f"{cohort} exists")
(cohort / "frozen").mkdir(parents=True)
subprocess.run(["rsync", "-a", "--exclude", "__pycache__", str(REPO / "research/adamw_spectra"), str(cohort / "frozen/")], check=True)
manifest = {str(p.relative_to(cohort / "frozen")): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((cohort / "frozen").rglob("*")) if p.is_file()}
(cohort / "frozen_manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
base = load_config(overrides=json.loads((V / "config_muon.json").read_text()))
SBATCH = """#!/bin/bash
#SBATCH --job-name=muon-{wave}-{arm}
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_rtx_a6000:4
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time={time}
#SBATCH --output={root}/slurm-%j.log
#SBATCH --error={root}/slurm-%j.log
#SBATCH --chdir={repo}
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 PYTHONUNBUFFERED=1
exec {repo}/.venv/bin/python -u {cohort}/frozen/adamw_spectra/run_arm.py --root {root} --world-size 4
"""
wave = cohort.name.split("_")[1]
for arm, spec in arms.items():
    d = cohort / arm
    d.mkdir()
    cfg = load_config(overrides={**base, **spec["change"]})
    (d / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    for f in ("tiny_train.bin", "tiny_val.bin"):
        (d / f).write_bytes((V / f).read_bytes())
    if spec.get("sbatch"):
        (d / "job.sbatch").write_text(SBATCH.format(wave=wave, arm=arm, root=d, repo=REPO, cohort=cohort, time=spec.get("time", "03:00:00")))
    print(arm, {k: cfg[k] for k in cfg if k != "model" and cfg[k] != base[k]},
          {k: v for k, v in cfg["model"].items() if base["model"].get(k) != v})
