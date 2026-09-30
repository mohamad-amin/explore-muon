"""Freeze and submit bounded standalone surrogate cohorts to sub-48GB GPUs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


PROJECT = Path(__file__).resolve().parents[2]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def freeze(destination):
    files = list((PROJECT / "research/tiny_spectra").glob("*.py"))
    files += [PROJECT / "research/adamw_spectra" / name
              for name in ("__init__.py", "model.py", "muon.py", "data_norm_muon.py", "train.py")]
    manifest = {}
    for source in files:
        relative = source.relative_to(PROJECT)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        manifest[str(relative)] = hashlib.sha256(target.read_bytes()).hexdigest()
    write_json(destination.parent / "source_manifest.json", manifest)


def base_config():
    source_path = PROJECT / "data/tiny_shakespeare/input.txt"
    return dict(data_path=str(source_path), data_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
                seed=20260928, method="muon", lr=.008, aux_lr=.002, momentum=.9,
                alpha=.25, out_beta=.25, decay=.01, soap_beta2=.9,
                root_refresh=5, cov_ema=.9, cov_stride=4, out_sequences=16, out_ema=.8,
                n_layer=4, n_embd=128, n_head=4, seq_len=128,
                batch_tokens=8192, total_tokens=4194304, microbatch_sequences=32,
                warmup_fraction=.034, cooldown_fraction=.1, grad_clip=1.0,
                validation_tokens=32768, eval_every=32, precision="bf16", device="cuda", cpu_threads=2)


def create(path, kind, config_list=None):
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=False)
    (path / "configs").mkdir()
    (path / "runs").mkdir()
    (path / "console").mkdir()
    base = base_config() if config_list is None else None
    configs = []
    if config_list is not None:
        configs = config_list
    else:
        for method in ("adamw", "muon", "pd", "ts", "spd"):
            lrs = [.0005, .001, .002] if method == "adamw" else [.004, .008, .016]
            if kind == "smoke":
                lrs = [lrs[-1]]
            for lr in lrs:
                cfg = dict(base, method=method, lr=lr)
                if kind == "smoke":
                    cfg.update(total_tokens=12*base["batch_tokens"], eval_every=12, validation_tokens=4096)
                configs.append(cfg)
    tasks = []
    for cfg in configs:
        run_id = cfg.get("run_id", f'{cfg["method"]}_b{cfg["batch_tokens"]}_lr{cfg["lr"]:g}_m{cfg["momentum"]:g}_s{cfg["seed"]}')
        cfg["run_id"] = run_id
        target = path / "configs" / (run_id + ".json")
        if target.exists():
            raise ValueError("Duplicate run_id: " + run_id)
        write_json(target, cfg)
        tasks.append(dict(config=str(target), out=str(path / "runs" / run_id)))
    # Qualification methods run sequentially on one physical GPU.
    task_groups = [tasks] if kind == "smoke" else [[task] for task in tasks]
    write_json(path / "tasks.json", task_groups)
    write_json(path / "cohort.json", dict(kind=kind, runs=len(tasks), task_groups=len(task_groups),
                                         stage="qualification" if kind=="smoke" else "discovery",
                                         gpu_gres="gpu:nvidia_rtx_a4000:1", max_parallel=1 if kind=="smoke" else 4))
    freeze(path / "frozen")
    script = """#!/bin/bash
set -euo pipefail
export CC=/usr/bin/gcc CXX=/usr/bin/g++ OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONUNBUFFERED=1
exec __WRAPPER__ __DRIVER__ task __COHORT__ --index "$SLURM_ARRAY_TASK_ID"
"""
    import shlex
    for key, value in (("__WRAPPER__", PROJECT / "run"),
                       ("__DRIVER__", path / "frozen/research/tiny_spectra/cohort.py"),
                       ("__COHORT__", path)):
        script = script.replace(key, shlex.quote(str(value)))
    (path / "job.sh").write_text(script)
    (path / "README.md").write_text(
        f"# Tiny surrogate: {kind}\n\n"
        "See this study's PROTOCOL.md for the decision and comparison criteria.\n"
        f"{len(tasks)} fresh runs; sources frozen here. A4000 only (16GB), plus runtime VRAM gate.\n"
        "Status: created, not yet submitted. tasks.json lists all arms; configs and failures are retained.\n"
        "This is a standalone character-model harness, not the original FineWeb pipeline.\n")
    return path


def submit(path):
    path = Path(path).resolve()
    if (path / "submission.json").exists():
        raise FileExistsError("Cohort already submitted; inspect existing handles instead of restarting")
    cohort = json.loads((path / "cohort.json").read_text())
    count = cohort["task_groups"]
    gres = cohort["gpu_gres"]
    minutes = cohort.get("walltime_minutes", 10 if cohort["kind"] == "smoke" else 30)
    if type(minutes) is not int or not 1 <= minutes <= 30:
        raise ValueError("Explicit cohort walltime must be an integer between 1 and 30 minutes")
    if gres not in ("gpu:nvidia_rtx_a4000:1", "gpu:nvidia_geforce_rtx_2080_ti:1",
                    "gpu:nvidia_rtx_5000_ada_generation:1"):
        raise ValueError("Only explicitly identified GPU types below 48 GB are permitted")
    command = ["sbatch", "--parsable", "--partition=gpu", "--nodes=1", "--ntasks=1",
               "--gres="+gres, "--cpus-per-task=2", "--mem=8G",
               f"--time=00:{minutes:02d}:00",
               f'--array=0-{count-1}%{cohort["max_parallel"]}', "--job-name=tiny-"+cohort["kind"],
               "--chdir="+str(path), "--output="+str(path/"console/%A_%a.log"), str(path/"job.sh")]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    write_json(path / "submission.json", dict(job_id=result.stdout.strip(), command=command))
    readme = path / "README.md"
    readme.write_text(readme.read_text().replace("Status: created, not yet submitted.",
                     f"Submitted as Slurm array {result.stdout.strip()}; inspect its live state."))
    print(result.stdout.strip(), flush=True)


def task(path, index):
    path = Path(path).resolve()
    manifest = json.loads((path / "source_manifest.json").read_text())
    for relative, expected in manifest.items():
        actual = hashlib.sha256((path / "frozen" / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError("Frozen source mismatch: " + relative)
    group = json.loads((path / "tasks.json").read_text())[index]
    env = dict(os.environ, PYTHONPATH=str(path / "frozen"))
    for arm in group:
        result = subprocess.run([sys.executable, "-m", "research.tiny_spectra.train", "--config", arm["config"],
                                 "--out", arm["out"]], cwd=path/"frozen", env=env)
        if result.returncode:
            raise SystemExit(result.returncode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=["create", "submit", "task"])
    parser.add_argument("path")
    parser.add_argument("--kind", choices=["smoke", "screen", "custom"], default="smoke")
    parser.add_argument("--configs", help="Explicit JSON config list for a predeclared follow-up")
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    if args.operation == "create":
        create(args.path, args.kind, json.loads(Path(args.configs).read_text()) if args.configs else None)
    elif args.operation == "submit":
        submit(args.path)
    else:
        task(args.path, args.index)


if __name__ == "__main__":
    main()
