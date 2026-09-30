"""Bounded eight-GPU qualification and one fixed-hyperparameter depth run."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback

parser = argparse.ArgumentParser()
parser.add_argument("--root", required=True, type=Path)
args = parser.parse_args()
ROOT = args.root.resolve()
REPO = ROOT.parents[2]
sys.path.insert(0, str(ROOT / "frozen"))
from adamw_spectra.train import atomic_json

START = time.time()
END = START + 7 * 3600 + 45 * 60
CHILD = None
STATE = {"phase": "starting", "job_id": os.environ.get("SLURM_JOB_ID"),
         "host": socket.gethostname(), "pid": os.getpid(), "started_unix": START,
         "world_size": 8, "workers": []}
CONFIG = json.loads((ROOT / "config.json").read_text())
PYTHON = str(REPO / ".venv/bin/python")


def persist():
    atomic_json(ROOT / "pipeline.json", STATE)


def stage(name, config, stop_after=None, audit=False, cap=None):
    global CHILD
    STATE["phase"] = name
    config_path = ROOT / f"{name}_config.json"
    atomic_json(config_path, config)
    command = [PYTHON, "-u", "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
               "--nproc_per_node=8", "--max_restarts=0", "-m", "adamw_spectra.distributed",
               "--config", str(config_path), "--out", str(ROOT / name)]
    if stop_after:
        command += ["--stop-after", str(stop_after)]
    if audit:
        command += ["--audit-replicas"]
    environment = os.environ.copy()
    environment.update(PYTHONPATH=str(ROOT / "frozen"), OMP_NUM_THREADS="2",
                       OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="2",
                       TORCHINDUCTOR_COMPILE_THREADS="1", PYTHONUNBUFFERED="1")
    for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
        environment.pop(key, None)
    started = time.time()
    with (ROOT / f"{name}.log").open("xb", buffering=0) as log:
        CHILD = subprocess.Popen(command, cwd=REPO, env=environment, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    record = {"name": name, "pid": CHILD.pid, "started_unix": started,
              "command": command, "state": "running"}
    STATE["workers"].append(record)
    persist()
    timeout = max(1, min(END - time.time(), cap or float("inf")))
    try:
        code = CHILD.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(CHILD.pid, signal.SIGTERM)
        try:
            CHILD.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(CHILD.pid, signal.SIGKILL)
            CHILD.wait(timeout=10)
        record["timed_out"] = True
        code = CHILD.returncode
    record.update(state="succeeded" if code == 0 else "failed", exit_code=code,
                  worker_seconds=time.time() - started, ended_unix=time.time())
    persist()
    CHILD = None
    if code != 0:
        raise RuntimeError(f"{name} worker group failed with exit code {code}")
    return json.loads((ROOT / name / "status.json").read_text())


def qualify_full():
    status = stage("qualification", CONFIG, stop_after=5, audit=True, cap=1800)
    out = ROOT / "qualification"
    if status["world_size"] != 8 or status["step"] != 5 or status["tokens"] != 5 * CONFIG["batch_tokens"]:
        raise RuntimeError("Full-size qualification did not meet the intended global horizon")
    for step in (1, 5):
        if not json.loads((out / f"replica_audit_step{step:06d}.json").read_text())["all_equal"]:
            raise RuntimeError("Qualification replica agreement failed")
    rows = [json.loads(p.read_text()) for p in sorted((out / "steps").glob("step*.json"))]
    normal = [r for r in rows if 2 <= r["step"] <= 4]
    step_seconds = sum(r["training_seconds"] for r in normal) / len(normal)
    spectral_seconds = next(r["measurement_seconds"] for r in rows if "matrices" in r)
    val_seconds = rows[-1]["validation_seconds"]
    residual = max(0, status["invocation_seconds"] - status["setup_seconds"] -
                   sum(r.get(key, 0) for r in rows for key in
                       ("training_seconds", "measurement_seconds", "validation_seconds")))
    forecast = 1469 * step_seconds + 60 * spectral_seconds + 31 * val_seconds + 15 * residual + 300
    report = {"passed": True, "status": status, "steady_step_seconds": step_seconds,
              "spectral_collection_seconds": spectral_seconds, "validation_seconds": val_seconds,
              "rough_main_seconds_including_io_and_300s_allowance": forecast,
              "remaining_controller_seconds": END - time.time(),
              "limit": "Short qualification forecast; measured full-run time remains the outcome"}
    atomic_json(ROOT / "QUALIFICATION_PASSED.json", report)
    if forecast > END - time.time():
        raise RuntimeError("Qualification forecasts insufficient time in this eight-hour allocation")


def main():
    import torch
    if torch.cuda.device_count() != 8:
        raise RuntimeError(f"Expected eight allocated GPUs; see {torch.cuda.device_count()}")
    for name, digest in json.loads((ROOT / "frozen_manifest.json").read_text()).items():
        if hashlib.sha256((ROOT / "frozen" / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Frozen source mismatch: {name}")
    if (CONFIG["model"]["n_embd"], CONFIG["model"]["n_head"], CONFIG["model"]["seq_len"],
        CONFIG["batch_tokens"], CONFIG["total_tokens"], CONFIG["learning_rate"]) != (
            512, 8, 512, 1048576, 1539870720, .0012):
        raise RuntimeError("Configuration is not the authorized depth-only comparison")
    STATE["depth"] = CONFIG["model"]["n_layer"]
    STATE["gpu_inventory"] = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total", "--format=csv"], text=True)
    persist()
    tiny = {**CONFIG, "model": {"vocab_size": 128, "n_layer": 2, "n_embd": 32, "n_head": 2, "seq_len": 16},
            "train_pattern": str(ROOT / "tiny_train.bin"), "validation_pattern": str(ROOT / "tiny_val.bin"),
            "batch_tokens": 128, "microbatch_sequences": 1, "total_tokens": 320,
            "validation_tokens": 80, "validation_every": 2, "warmup_steps": 1,
            "checkpoint_every": 2, "spectra_every": 1}
    stage("tiny_cuda", tiny, audit=True, cap=1200)
    qualify_full()
    result = stage("scientific", CONFIG)
    if result["status"] != "complete" or result["tokens"] != 1539870720 or result["step"] != 1469:
        raise RuntimeError("Scientific run did not reach its fixed horizon")
    STATE["phase"] = "analysis"
    persist()
    from adamw_spectra.analyze import analyze
    summary = analyze(ROOT / "scientific")
    if not summary["budget_complete"] or summary["spectra_samples"] != 60:
        raise RuntimeError("Incomplete scientific spectra")
    STATE.update(phase="complete", ended_unix=time.time(),
                 allocated_gpu_seconds=8 * (time.time() - START),
                 active_worker_gpu_seconds=8 * sum(w["worker_seconds"] for w in STATE["workers"]))
    persist()


def interrupt(signum, frame):
    raise RuntimeError(f"Received signal {signum}")


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    try:
        main()
    except BaseException as exc:
        if CHILD is not None and CHILD.poll() is None:
            os.killpg(CHILD.pid, signal.SIGTERM)
            try:
                CHILD.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(CHILD.pid, signal.SIGKILL)
                CHILD.wait(timeout=10)
        STATE.update(phase="failed", error=f"{type(exc).__name__}: {exc}", ended_unix=time.time())
        persist()
        traceback.print_exc()
        raise
