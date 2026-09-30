"""One arm of the Muon improvement search (MUON_CASE.md, 2026-09-25 onward).

Frontier-norm depth-8 Muon; an arm may change only whitelisted optimizer settings
(and the seed) relative to that run's config. Qualification mirrors run_muon.py;
seed-260924 arms must reproduce the reference initial weights.
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
ROOT = args.root.resolve()            # <repo>/logs/muon_spectra/<cohort>/<arm>
REPO = ROOT.parents[3]
FROZEN = ROOT.parent / "frozen"
REFERENCE = REPO / "logs/muon_spectra/depth8_w512_nobias_rms_qk_20260925/muon/scientific/metadata.json"
BASE_CONFIG = REPO / "logs/muon_spectra/depth8_w512_nobias_rms_qk_20260925/config_muon.json"
TREATMENTS = {"seed", "ns_polynomial", "ns_steps", "deflation", "deflation_mode", "deflation_head_weight",
              "deflation_power_iters", "deflation_window", "deflation_oversample",
              "deflation_subspace_iters", "deflation_threshold", "deflation_pad",
              "learning_rate", "muon_nesterov", "mean_whitening", "mean_whitening_beta",
              "soap_precondition", "soap_beta2", "soap_denom_power", "mean_whitening_power",
              "mean_bias_adam", "soap_basis", "soap_second_moment",
              "data_norm_alpha", "data_norm_damping", "data_norm_refresh", "soap_norm", "data_norm_post", "data_norm_rows", "data_norm_center",
              "data_norm_pre", "data_norm_mode", "soap_layers",
              "total_tokens",   # horizon arms (wave 13); the controller reads the horizon from the config
              "data_norm_alpha_by_kind",
              # published comparators (Newton-Muon and PMuon head-to-head)
              "newton_muon", "newton_muon_damping", "newton_muon_refresh", "newton_muon_ema",
              "newton_muon_down_blocks", "pmuon", "pmuon_gamma", "pmuon_beta",
              # Track 3 transfer tests: one Track 3 setting at a time (comparisons pair arms that share it)
              "weight_decay", "cooldown_fraction", "batch_tokens",
              "warmup_steps",   # batch-size study (second-order audit): warmup held fixed in tokens across batch sizes
              "data_norm_out_beta", "data_norm_out_source", "data_norm_out_refresh",   # two-sided data norm
              "data_norm_out_sequences", "data_norm_out_ema", "data_norm_out_placebo",
              "data_norm_decay", "data_norm_decay_power",   # PD-geometry weight decay (Track 3 follow-up)
              "muon_momentum", "grad_clip",   # second-order audit: momentum sweep and no-clip controls at batch 4M
              "muon_prefilter",   # second-order audit: two-tap momentum pre-filter at 16M (2026-09-28)
              "head_whitening_alpha", "head_whitening_center", "head_whitening_norm",
              "muon_momentum_start", "muon_momentum_warmup",   # second-order audit: momentum warmup (2026-09-28)
              "data_norm_top_only",
              "momentum_split_top",
              "log_gradient_noise",
              "data_norm_snr_gate", "data_norm_snr_ema"}   # second-order audit: SNR-gated tail amplification (2026-09-28)   # second-order audit: across-rank gradient noise r and drift q per step (2026-09-28)   # second-order audit: short momentum in the top input subspace only (2026-09-28)   # second-order audit: top-only input whitening, PD-top (2026-09-28)   # second-order audit: Adam in input-whitened coordinates for the head (2026-09-28)
MODEL_TREATMENTS = {"track_input_stats", "track_input_cov", "track_head_cov",
                    "n_embd", "n_head",   # width scale check (wave 19): comparisons pair arms of one width
                    "bias"}               # Track 3 transfer test: Linear biases
# Measurement-only settings: they change where spectra are computed, not training.
MEASUREMENT = {"svd_device",
               "keep_checkpoints",   # second-order audit: kept checkpoints and next-step weights
               "muon_track_displacement"}   # second-order audit: EMA of displacements in the optimizer state
sys.path.insert(0, str(FROZEN))
from adamw_spectra.train import atomic_json

START = time.time()
# Horizon from the arm's own token budget; the default 1,539,870,720 tokens is 1469 steps and 60 spectra.
HORIZON_TOKENS = json.loads((ROOT / "config.json").read_text())["total_tokens"]
HORIZON_STEPS = -(-HORIZON_TOKENS // json.loads((ROOT / "config.json").read_text())["batch_tokens"])
SPECTRA = 1 + HORIZON_STEPS // 25 + (1 if HORIZON_STEPS % 25 else 0)
END = START + (7 if HORIZON_STEPS <= 1469 else 7 * HORIZON_STEPS / 1469 + 1) * 3600
CHILD = None
STATE = {"phase": "starting", "job_id": os.environ.get("SLURM_JOB_ID"),
         "host": socket.gethostname(), "pid": os.getpid(), "started_unix": START,
         "world_size": WORLD, "step_id": os.environ.get("SLURM_STEP_ID"), "workers": []}
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
               f"--nproc_per_node={WORLD}", "--max_restarts=0", "-m", "adamw_spectra.distributed",
               "--config", str(config_path), "--out", str(ROOT / name)]
    if stop_after:
        command += ["--stop-after", str(stop_after)]
    if audit:
        command += ["--audit-replicas"]
    environment = os.environ.copy()
    environment.update(PYTHONPATH=str(FROZEN), OMP_NUM_THREADS="2",
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
    # Engineering override: Newton-Muon first refreshes at step 64, so the five audited steps refresh every 2.
    newton = {"newton_muon_refresh": 2} if CONFIG.get("newton_muon") else {}
    # Engineering override: a kept checkpoint inside the five qualification steps exercises that path at full size.
    kept = {"keep_checkpoints": [2]} if CONFIG["keep_checkpoints"] else {}
    status = stage("qualification", {**CONFIG, "warmup_steps": 1, **newton, **kept}, stop_after=5, audit=True, cap=1800)
    out = ROOT / "qualification"
    if kept:
        import torch
        saved = torch.load(out / "kept/step000002.pt", map_location="cpu", weights_only=False)
        after = torch.load(out / "kept/step000003_weights.pt", map_location="cpu", weights_only=False)
        if saved["step"] != 2 or after["step"] != 3 or set(saved["model"]) != set(after["model"]):
            raise RuntimeError("Kept checkpoint qualification: wrong steps or keys")
        moved = [k for k in saved["model"] if not torch.equal(saved["model"][k], after["model"][k])]
        if not moved or not saved["optimizer"]["state"]:
            raise RuntimeError("Kept checkpoint qualification: no parameter change or empty optimizer state")
        del saved, after
    if status["world_size"] != WORLD or status["step"] != 5 or status["tokens"] != 5 * CONFIG["batch_tokens"]:
        raise RuntimeError("Full-size qualification did not meet the intended global horizon")
    for step in (1, 5):
        if not json.loads((out / f"replica_audit_step{step:06d}.json").read_text())["all_equal"]:
            raise RuntimeError("Qualification replica agreement failed")
    observed = json.loads((out / "metadata.json").read_text())
    baseline = json.loads(REFERENCE.read_text())
    base_model = json.loads(BASE_CONFIG.read_text())["model"]
    same_size = all(CONFIG["model"].get(key) == base_model.get(key) for key in ("n_layer", "n_embd", "n_head", "bias"))
    # Other widths (or biases) have their own parameter count and initial weights; the data manifests must still match.
    fields = ("train_manifest", "validation_manifest") + (("parameter_count",) if same_size else ()) + (
        ("initial_model_sha256",) if CONFIG["seed"] == 260924 and same_size else ())
    for field in fields:
        if observed[field] != baseline[field]:
            raise RuntimeError(f"Frontier-norm Muon comparator mismatch: {field}")
    rows = [json.loads(p.read_text()) for p in sorted((out / "steps").glob("step*.json"))]
    normal = [r for r in rows if 2 <= r["step"] <= 4]
    step_seconds = sum(r["training_seconds"] for r in normal) / len(normal)
    spectral_seconds = next(r["measurement_seconds"] for r in rows if "matrices" in r)
    val_seconds = rows[-1]["validation_seconds"]
    residual = max(0, status["invocation_seconds"] - status["setup_seconds"] -
                   sum(r.get(key, 0) for r in rows for key in
                       ("training_seconds", "measurement_seconds", "validation_seconds")))
    forecast = (HORIZON_STEPS * step_seconds + SPECTRA * spectral_seconds + (HORIZON_STEPS // 50 + 2) * val_seconds
                + 15 * residual + 300)
    report = {"passed": True, "engineering_override": "warmup_steps=1 tests peak rates; scientific warmup remains 50",
              "initial_model_sha256": observed["initial_model_sha256"], "status": status, "steady_step_seconds": step_seconds,
              "spectral_collection_seconds": spectral_seconds, "validation_seconds": val_seconds,
              "rough_main_seconds_including_io_and_300s_allowance": forecast,
              "remaining_controller_seconds": END - time.time(),
              "limit": "Short qualification forecast; measured full-run time remains the outcome"}
    atomic_json(ROOT / "QUALIFICATION_PASSED.json", report)
    if forecast > END - time.time():
        raise RuntimeError("Qualification forecasts insufficient time in this eight-hour allocation")


def main():
    import torch
    if torch.cuda.device_count() != WORLD:
        raise RuntimeError(f"Expected {WORLD} allocated GPUs; see {torch.cuda.device_count()}")
    for name, digest in json.loads((ROOT.parent / "frozen_manifest.json").read_text()).items():
        if hashlib.sha256((FROZEN / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Frozen source mismatch: {name}")
    from adamw_spectra.train import load_config
    base = load_config(overrides=json.loads(BASE_CONFIG.read_text()))
    loaded = load_config(overrides=CONFIG)
    changed = {key for key in loaded if key != "model" and loaded[key] != base[key]}
    model_changed = {key for key in loaded["model"] if loaded["model"][key] != base["model"].get(key)}
    if model_changed - MODEL_TREATMENTS:
        raise RuntimeError(f"Arm changes the architecture: {sorted(model_changed - MODEL_TREATMENTS)}")
    changed |= {f"model.{key}" for key in model_changed}
    changed -= MEASUREMENT
    if changed - TREATMENTS - {f"model.{key}" for key in MODEL_TREATMENTS}:
        raise RuntimeError(f"Arm changes non-treatment settings: {sorted(changed - TREATMENTS)}")
    if set(CONFIG) != set(loaded):
        raise RuntimeError("Arm config must list every setting explicitly")
    STATE["depth"] = CONFIG["model"]["n_layer"]
    STATE["arm"] = ROOT.name
    STATE["treatment"] = {key: (loaded["model"][key[6:]] if key.startswith("model.") else loaded[key])
                          for key in sorted(changed)}
    STATE["gpu_inventory"] = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total", "--format=csv"], text=True)
    persist()
    # Keep the arm's architecture flags (bias, norm, qk_norm, track_input_stats) in the tiny model.
    tiny = {**CONFIG, "model": {**CONFIG["model"], "vocab_size": 128, "n_layer": 2, "n_embd": 32,
                                "n_head": 2, "seq_len": 16},
            "train_pattern": str(ROOT / "tiny_train.bin"), "validation_pattern": str(ROOT / "tiny_val.bin"),
            "batch_tokens": 128, "microbatch_sequences": 1, "total_tokens": 288,
            "validation_tokens": 80, "validation_every": 2, "warmup_steps": 1,
            "checkpoint_every": 2, "spectra_every": 1,
            # Engineering override: keep step 1 (and step 2's weights) so the tiny run exercises kept checkpoints.
            **({"keep_checkpoints": [1]} if CONFIG["keep_checkpoints"] else {}),
            # Engineering override: a 32-wide tiny model needs a wider window for the gate to fire.
            **({"deflation_window": 0.2} if CONFIG["deflation"] and CONFIG["deflation_mode"] == "paper" else {}),
            # Engineering override: refresh Newton-Muon every step so the audited tiny run exercises it.
            **({"newton_muon_refresh": 1} if CONFIG.get("newton_muon") else {}),
            # Engineering override: the 3-step tiny run's warmup step would fall inside a long cooldown.
            **({"cooldown_fraction": 0.1} if CONFIG["cooldown_fraction"] > 0.5 else {})}
    stage("tiny_cuda", tiny, audit=True, cap=1200)
    qualify_full()
    result = stage("scientific", CONFIG)
    if result["status"] != "complete" or result["tokens"] != HORIZON_TOKENS or result["step"] != HORIZON_STEPS:
        raise RuntimeError("Scientific run did not reach its fixed horizon")
    STATE["phase"] = "analysis"
    persist()
    from adamw_spectra.analyze import analyze
    summary = analyze(ROOT / "scientific")
    if not summary["budget_complete"] or summary["spectra_samples"] != SPECTRA:
        raise RuntimeError("Incomplete scientific spectra")
    momentum_summary = analyze(ROOT / "scientific", quantity="momentum")
    if not momentum_summary["budget_complete"] or momentum_summary["spectra_samples"] != SPECTRA:
        raise RuntimeError("Incomplete momentum spectra")
    STATE.update(phase="complete", ended_unix=time.time(),
                 allocated_gpu_seconds=WORLD * (time.time() - START),
                 active_worker_gpu_seconds=WORLD * sum(w["worker_seconds"] for w in STATE["workers"]))
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
