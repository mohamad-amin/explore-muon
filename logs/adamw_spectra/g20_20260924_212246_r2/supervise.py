"""Frozen, bounded g20 execution: qualify, paired LR pilots, one spectrum run."""
import datetime
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

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
CODE = ROOT / "frozen"
sys.path.insert(0, str(CODE))
from adamw_spectra.train import atomic_json
from adamw_spectra.sweep import RATES, select_candidate

PYTHON = str(REPO / ".venv/bin/python")
CONFIG = json.loads((ROOT / "config.json").read_text())
ACTIVE = []
STATE = {"phase": "starting", "allocation": "2618555", "host": socket.gethostname(),
         "supervisor_pid": os.getpid(), "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
         "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
         "workers": []}


def persist():
    atomic_json(ROOT / "pipeline.json", STATE)


def launch(name, gpu, config, limit, timeout):
    config_path = ROOT / f"{name}_config.json"
    atomic_json(config_path, config)
    out = ROOT / name
    cmd = [PYTHON, "-u", "-m", "adamw_spectra.train", "--config", str(config_path), "--out", str(out)]
    if limit:
        cmd += ["--stop-after", str(limit)]
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(CODE),
               OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="4",
               TORCHINDUCTOR_COMPILE_THREADS="2", PYTHONUNBUFFERED="1")
    with (ROOT / f"{name}.log").open("xb", buffering=0) as log:
        process = subprocess.Popen(cmd, cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    record = {"name": name, "gpu": gpu, "pid": process.pid, "command": cmd,
              "started_unix": time.time(), "timeout_seconds": timeout, "state": "running"}
    STATE["workers"].append(record)
    ACTIVE.append((process, record))
    persist()
    return process, record


def wait_workers(workers):
    pending = list(workers)
    while pending:
        for process, record in list(pending):
            if process.poll() is None and time.time() - record["started_unix"] > record["timeout_seconds"]:
                record["timed_out"] = True
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
            code = process.poll()
            if code is not None:
                record.update(exit_code=code, ended_unix=time.time(),
                              state="succeeded" if code == 0 else "failed")
                record["worker_seconds"] = record["ended_unix"] - record["started_unix"]
                pending.remove((process, record))
                persist()
        if pending:
            time.sleep(3)


def rows(name):
    return [json.loads(p.read_text()) for p in sorted((ROOT / name / "steps").glob("step*.json"))]


def check_qualification():
    out = ROOT / "qualification"
    metadata = json.loads((out / "metadata.json").read_text())
    status = json.loads((out / "status.json").read_text())
    records = rows("qualification")
    if status["step"] != 5 or status["status"] != "stopped_at_requested_step":
        raise RuntimeError("Full-size qualification did not complete five updates")
    if metadata["parameter_count"] != 76993536 or len(metadata["measured_parameters"]) != 24:
        raise RuntimeError("Qualification model/panel mismatch")
    panel = next(r["matrices"] for r in records if "matrices" in r)
    if any(not math.isfinite(v["normalized_energy_sum"]) or
           abs(v["normalized_energy_sum"] - 1) > 2e-4 for v in panel.values()):
        raise RuntimeError("Qualification spectra failed normalization")
    training = [r for r in records if r["step"] >= 2]
    mean_step = sum(r["training_seconds"] for r in training) / len(training)
    measurement = next(r["measurement_seconds"] for r in records if "matrices" in r)
    validation = records[-1]["validation_seconds"]
    report = {"passed": True, "host": STATE["host"], "metadata": metadata,
              "status": status, "mean_training_seconds_steps_2_to_5": mean_step,
              "measurement_seconds": measurement, "validation_seconds": validation,
              "rough_pilot_seconds_excluding_startup_io": 300 * mean_step + 7 * validation,
              "rough_main_seconds_excluding_startup_io": 1469 * mean_step + 60 * measurement + 31 * validation,
              "forecast_limit": "Short qualification; excludes compilation, checkpoint I/O and parallel CPU contention"}
    atomic_json(ROOT / "QUALIFICATION_PASSED.json", report)
    return metadata


def select_rate(qualification_metadata):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    candidates = []
    first_hash = None
    fig, ax = plt.subplots(figsize=(7, 4))
    for lr in RATES:
        name = f"pilot_lr_{lr:g}"
        record = next(r for r in STATE["workers"] if r["name"] == name)
        candidate = {"learning_rate": lr, "valid": False, "pilot_seed": CONFIG["seed"] - 1,
                     "worker": record}
        if record["exit_code"] != 0:
            candidate["error"] = "Worker execution failed; inspect its log (not a scientific negative)"
            candidates.append(candidate)
            continue
        out = ROOT / name
        status = json.loads((out / "status.json").read_text())
        metadata = json.loads((out / "metadata.json").read_text())
        expected_config = {**CONFIG, "seed": CONFIG["seed"] - 1, "learning_rate": lr, "spectra_every": 0}
        if metadata["config"] != expected_config:
            raise RuntimeError("Pilot config differs from predeclared condition")
        for key in ("source_sha256", "train_manifest", "validation_manifest", "parameter_count"):
            if metadata[key] != qualification_metadata[key]:
                raise RuntimeError(f"Pilot mismatch: {key}")
        if first_hash is None:
            first_hash = metadata["initial_model_sha256"]
        if metadata["initial_model_sha256"] != first_hash:
            raise RuntimeError("Pilot initial models are not paired")
        if status["step"] != 300 or status["tokens"] != 300 * CONFIG["batch_tokens"]:
            raise RuntimeError("Pilot horizon mismatch")
        history = rows(name)
        evaluation = {r["step"]: r["validation_nll"] for r in history if "validation_nll" in r}
        scores = [evaluation[s] for s in (200, 250, 300)]
        if not all(math.isfinite(s) for s in scores):
            raise RuntimeError("Non-finite selection score")
        candidate.update(valid=True, score=sum(scores) / 3, evaluation_nll=scores,
                         clipping_fraction=sum(r["gradient_clipped"] for r in history[1:]) / 300)
        candidates.append(candidate)
        ax.plot(list(evaluation), list(evaluation.values()), label=f"LR={lr:g}")
    atomic_json(ROOT / "candidates.json", candidates)
    selected = select_candidate(candidates)
    ax.set(xlabel="step", ylabel="held-out NLL", title="Paired 77M AdamW LR pilots")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT / "lr_selection.png", dpi=150)
    plt.close(fig)
    selected_config = {**CONFIG, "learning_rate": selected["learning_rate"]}
    report = {"evaluation_steps": [200, 250, 300], "validation_tokens": CONFIG["validation_tokens"],
              "criterion": "minimum mean validation NLL; exact tie chooses lower rate",
              "candidates": candidates, "selected_learning_rate": selected["learning_rate"],
              "boundary_winner": selected["learning_rate"] in (min(RATES), max(RATES)),
              "paired_initial_sha256": first_hash, "parallel_pilots": True}
    atomic_json(ROOT / "selection.json", report)
    atomic_json(ROOT / "selected_config.json", selected_config)
    return selected_config


def interrupted(signum, frame):
    raise RuntimeError(f"Supervisor received signal {signum}")


def main():
    if not socket.gethostname().split(".")[0] == "g20":
        raise RuntimeError("This frozen launch is restricted to g20")
    manifest = json.loads((ROOT / "frozen_manifest.json").read_text())
    for name, expected in manifest.items():
        if hashlib.sha256((CODE / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Frozen source mismatch: {name}")
    active_gpu_processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
    if active_gpu_processes:
        raise RuntimeError("g20 GPU processes appeared before launch; refusing to overlap unrelated workers")
    persist()
    STATE["phase"] = "qualification"
    qualification = launch("qualification", 0, CONFIG, 5, 900)
    wait_workers([qualification])
    if qualification[1]["exit_code"] != 0:
        raise RuntimeError("Full-size qualification failed; pilots not started")
    qualification_metadata = check_qualification()
    STATE["phase"] = "lr_pilots"
    pilots = [launch(f"pilot_lr_{lr:g}", gpu,
                     {**CONFIG, "seed": CONFIG["seed"] - 1, "learning_rate": lr, "spectra_every": 0},
                     300, 7200) for gpu, lr in enumerate(RATES)]
    wait_workers(pilots)
    selected = select_rate(qualification_metadata)
    STATE.update(phase="scientific_run", selected_learning_rate=selected["learning_rate"])
    scientific = launch("77m_seed260924", 0, selected, None, 21600)
    wait_workers([scientific])
    if scientific[1]["exit_code"] != 0:
        raise RuntimeError("Scientific worker failed; inspect checkpoint and logs")
    STATE["phase"] = "analysis"
    persist()
    from adamw_spectra.analyze import analyze
    summary = analyze(ROOT / "77m_seed260924")
    if not summary["budget_complete"] or summary["spectra_samples"] != 60:
        raise RuntimeError("Scientific output did not meet the planned horizon/measurement count")
    STATE.update(phase="complete", ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 active_worker_gpu_seconds=sum(r["worker_seconds"] for r in STATE["workers"]),
                 result_summary=str(ROOT / "77m_seed260924/summary.json"))
    persist()


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        main()
    except BaseException as exc:
        for process, record in ACTIVE:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
                record.update(state="terminated", exit_code=process.returncode)
        STATE.update(phase="failed", error=f"{type(exc).__name__}: {exc}",
                     ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        persist()
        traceback.print_exc()
        raise
