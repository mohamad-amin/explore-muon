"""Bounded two-arm pipeline for the frontier-norm architecture variant (MUON_CASE.md).

Muon arm: tiny CUDA check, five-update full-size qualification, one fixed 1,469-update
run, spectral analysis and the spike diagnostic. AdamW arm: three predeclared LR pilots,
qualification paired to the Muon arm's initial weights, one run, analysis, diagnostic.
"""
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
parser.add_argument("--world-size", required=True, type=int, choices=(4,))
args = parser.parse_args()
WORLD = args.world_size
ROOT = args.root.resolve()
REPO = ROOT.parents[2]
sys.path.insert(0, str(ROOT / "frozen"))
from adamw_spectra.train import atomic_json

START = time.time()
END = START + 7 * 3600
CHILDREN = []
STATE = {"phase": "starting", "job_id": os.environ.get("SLURM_JOB_ID"), "step_id": os.environ.get("SLURM_STEP_ID"),
         "host": socket.gethostname(), "pid": os.getpid(), "started_unix": START, "world_size": WORLD,
         "workers": []}
MUON = json.loads((ROOT / "config_muon.json").read_text())
ADAMW = json.loads((ROOT / "config_adamw.json").read_text())
VARIANT = {"bias": False, "norm": "rmsnorm", "qk_norm": True}
PILOT_RATES = (0.0006, 0.0012, 0.0024)
PYTHON = str(REPO / ".venv/bin/python")


def persist():
    atomic_json(ROOT / "pipeline.json", STATE)


def environment(gpus=None):
    env = os.environ.copy()
    env.update(PYTHONPATH=str(ROOT / "frozen"), OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="2", TORCHINDUCTOR_COMPILE_THREADS="1", PYTHONUNBUFFERED="1")
    for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
        env.pop(key, None)
    if gpus is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpus
    return env


def launch(name, command, log_path, gpus=None):
    with log_path.open("xb", buffering=0) as log:
        child = subprocess.Popen(command, cwd=REPO, env=environment(gpus), stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    CHILDREN.append(child)
    record = {"name": name, "pid": child.pid, "started_unix": time.time(), "command": command,
              "gpus": gpus, "state": "running"}
    STATE["workers"].append(record)
    persist()
    return child, record


def kill(child):
    os.killpg(child.pid, signal.SIGTERM)
    try:
        child.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait(timeout=10)


def finish(child, record, cap=None):
    try:
        code = child.wait(timeout=max(1, min(END - time.time(), cap or float("inf"))))
    except subprocess.TimeoutExpired:
        kill(child)
        record["timed_out"] = True
        code = child.returncode
    CHILDREN.remove(child)
    record.update(state="succeeded" if code == 0 else "failed", exit_code=code, ended_unix=time.time(),
                  worker_seconds=time.time() - record["started_unix"])
    persist()
    return code


def stage(arm, name, config, stop_after=None, audit=False, cap=None):
    STATE["phase"] = f"{arm}/{name}"
    folder = ROOT / arm
    folder.mkdir(exist_ok=True)
    config_path = folder / f"{name}_config.json"
    atomic_json(config_path, config)
    command = [PYTHON, "-u", "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
               f"--nproc_per_node={WORLD}", "--max_restarts=0", "-m", "adamw_spectra.distributed",
               "--config", str(config_path), "--out", str(folder / name)]
    if stop_after:
        command += ["--stop-after", str(stop_after)]
    if audit:
        command += ["--audit-replicas"]
    child, record = launch(f"{arm}/{name}", command, folder / f"{name}.log")
    if finish(child, record, cap) != 0:
        raise RuntimeError(f"{arm}/{name} worker group failed")
    return json.loads((folder / name / "status.json").read_text())


def qualify(arm, config, paired_hash=None):
    status = stage(arm, "qualification", {**config, "warmup_steps": 1}, stop_after=5, audit=True, cap=1800)
    out = ROOT / arm / "qualification"
    if status["world_size"] != WORLD or status["step"] != 5 or status["tokens"] != 5 * config["batch_tokens"]:
        raise RuntimeError(f"{arm} qualification did not meet the intended global horizon")
    for step in (1, 5):
        if not json.loads((out / f"replica_audit_step{step:06d}.json").read_text())["all_equal"]:
            raise RuntimeError(f"{arm} qualification replica agreement failed")
    observed = json.loads((out / "metadata.json").read_text())
    if paired_hash is not None and observed["initial_model_sha256"] != paired_hash:
        raise RuntimeError(f"{arm} initial weights are not paired with the Muon arm")
    rows = [json.loads(p.read_text()) for p in sorted((out / "steps").glob("step*.json"))]
    normal = [r for r in rows if 2 <= r["step"] <= 4]
    step_seconds = sum(r["training_seconds"] for r in normal) / len(normal)
    spectral_seconds = next(r["measurement_seconds"] for r in rows if "matrices" in r)
    val_seconds = rows[-1]["validation_seconds"]
    forecast = 1469 * step_seconds + 60 * spectral_seconds + 31 * val_seconds + 600
    report = {"passed": True, "engineering_override": "warmup_steps=1 tests peak rates; science keeps 50",
              "status": status, "steady_step_seconds": step_seconds,
              "spectral_collection_seconds": spectral_seconds, "validation_seconds": val_seconds,
              "rough_main_seconds_including_600s_allowance": forecast,
              "remaining_controller_seconds": END - time.time(),
              "initial_model_sha256": observed["initial_model_sha256"]}
    atomic_json(ROOT / arm / "QUALIFICATION_PASSED.json", report)
    if forecast > END - time.time():
        raise RuntimeError(f"{arm} qualification forecasts insufficient controller time")
    return observed["initial_model_sha256"]


def science(arm, config, quantities):
    result = stage(arm, "scientific", config)
    if result["status"] != "complete" or result["tokens"] != 1539870720 or result["step"] != 1469:
        raise RuntimeError(f"{arm} scientific run did not reach its fixed horizon")
    STATE["phase"] = f"{arm}/analysis"
    persist()
    from adamw_spectra.analyze import analyze
    for quantity in quantities:
        summary = analyze(ROOT / arm / "scientific", **({} if quantity == "update" else {"quantity": quantity}))
        if not summary["budget_complete"] or summary["spectra_samples"] != 60:
            raise RuntimeError(f"{arm} incomplete {quantity} spectra")


def diagnostic(arm, gpu):
    command = [PYTHON, "-u", "-m", "adamw_spectra.spike_diagnostics", "--run", str(ROOT / arm / "scientific"),
               "--out", str(ROOT / arm / "diagnostic")]
    return launch(f"{arm}/diagnostic", command, ROOT / arm / "diagnostic.log", gpus=gpu)


def check_diagnostic(arm, child, record):
    if finish(child, record, cap=3600) != 0:
        raise RuntimeError(f"{arm} diagnostic failed")
    status = json.loads((ROOT / arm / "diagnostic" / "status.json").read_text())["status"]
    if status != "complete":
        raise RuntimeError(f"{arm} diagnostic ended as {status}")


def pilots():
    STATE["phase"] = "adamw/pilots"
    folder = ROOT / "adamw"
    folder.mkdir(exist_ok=True)
    running = []
    for gpu, rate in enumerate(PILOT_RATES):
        config = {**ADAMW, "seed": ADAMW["seed"] - 1, "learning_rate": rate, "spectra_every": 0}
        path = folder / f"pilot_lr_{rate:g}_config.json"
        atomic_json(path, config)
        command = [PYTHON, "-u", "-m", "adamw_spectra.train", "--config", str(path),
                   "--out", str(folder / f"pilot_lr_{rate:g}"), "--stop-after", "300"]
        running.append((rate, *launch(f"adamw/pilot_lr_{rate:g}", command,
                                      folder / f"pilot_lr_{rate:g}.log", gpus=str(gpu))))
    candidates = []
    for rate, child, record in running:
        candidate = {"learning_rate": rate, "valid": False, "pilot_seed": ADAMW["seed"] - 1, "worker": record}
        try:
            if finish(child, record, cap=7200) != 0:
                raise RuntimeError("pilot worker failed")
            run = folder / f"pilot_lr_{rate:g}"
            status = json.loads((run / "status.json").read_text())
            if status["step"] != 300 or status["status"] != "stopped_at_requested_step":
                raise RuntimeError("pilot did not run the intended prefix")
            scores = [json.loads((run / "steps" / f"step{s:06d}.json").read_text())["validation_nll"]
                      for s in (200, 250, 300)]
            if not all(math.isfinite(s) for s in scores):
                raise RuntimeError("non-finite selection loss")
            candidate.update(valid=True, score=sum(scores) / 3, evaluation_nll=scores)
        except (RuntimeError, OSError, KeyError, ValueError) as exc:
            candidate["error"] = f"{type(exc).__name__}: {exc}"
        candidates.append(candidate)
    valid = [c for c in candidates if c["valid"]]
    if not valid:
        raise RuntimeError("No valid AdamW pilot; the AdamW arm is not qualified")
    chosen = min(valid, key=lambda c: (c["score"], c["learning_rate"]))
    atomic_json(folder / "selection.json", {
        "evaluation_steps": [200, 250, 300], "criterion": "minimum mean validation NLL; exact tie chooses lower rate",
        "grid": list(PILOT_RATES), "candidates": candidates, "selected_learning_rate": chosen["learning_rate"],
        "boundary_winner": chosen["learning_rate"] in (min(PILOT_RATES), max(PILOT_RATES))})
    return chosen["learning_rate"]


def main():
    import torch
    if torch.cuda.device_count() != WORLD:
        raise RuntimeError(f"Expected {WORLD} GPUs; see {torch.cuda.device_count()}")
    for name, digest in json.loads((ROOT / "frozen_manifest.json").read_text()).items():
        if hashlib.sha256((ROOT / "frozen" / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Frozen source mismatch: {name}")
    for config, optimizer in ((MUON, "muon"), (ADAMW, "adamw")):
        model = config["model"]
        if ((model["n_layer"], model["n_embd"], model["n_head"], model["seq_len"]) != (8, 512, 8, 512)
                or {k: model.get(k) for k in VARIANT} != VARIANT or config["optimizer"] != optimizer
                or (config["batch_tokens"], config["total_tokens"], config["seed"]) != (1048576, 1539870720, 260924)):
            raise RuntimeError(f"{optimizer} configuration is not the authorized variant comparison")
    if (MUON["learning_rate"], MUON["aux_learning_rate"], MUON["muon_momentum"]) != (.01, .002, .95):
        raise RuntimeError("Muon recipe mismatch")
    STATE["gpu_inventory"] = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total", "--format=csv"], text=True)
    persist()
    tiny = {**MUON, "model": {"vocab_size": 128, "n_layer": 2, "n_embd": 32, "n_head": 2, "seq_len": 16, **VARIANT},
            "train_pattern": str(ROOT / "tiny_train.bin"), "validation_pattern": str(ROOT / "tiny_val.bin"),
            "batch_tokens": 128, "microbatch_sequences": 1, "total_tokens": 288, "validation_tokens": 80,
            "validation_every": 2, "warmup_steps": 1, "checkpoint_every": 2, "spectra_every": 1}
    stage("muon", "tiny_cuda", tiny, audit=True, cap=1200)
    muon_hash = qualify("muon", MUON)
    science("muon", MUON, ("update", "momentum"))
    muon_diag = diagnostic("muon", "3")
    rate = pilots()
    check_diagnostic("muon", *muon_diag)
    adamw = {**ADAMW, "learning_rate": rate}
    atomic_json(ROOT / "adamw" / "selected_config.json", adamw)
    qualify("adamw", adamw, paired_hash=muon_hash)
    science("adamw", adamw, ("update",))
    check_diagnostic("adamw", *diagnostic("adamw", "0"))
    STATE.update(phase="complete", ended_unix=time.time(), selected_adamw_learning_rate=rate,
                 allocated_gpu_seconds=WORLD * (time.time() - START))
    persist()


def interrupt(signum, frame):
    raise RuntimeError(f"Received signal {signum}")


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    try:
        main()
    except BaseException as exc:
        for child in list(CHILDREN):
            if child.poll() is None:
                kill(child)
        STATE.update(phase="failed", error=f"{type(exc).__name__}: {exc}", ended_unix=time.time())
        persist()
        traceback.print_exc()
        raise
