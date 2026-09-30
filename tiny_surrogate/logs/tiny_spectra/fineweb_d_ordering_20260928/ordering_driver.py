"""Freeze and execute the user-authorized native Candidate-D ordering screen.

The worker uses the successful smoke's immutable numerical sources. This file
only schedules bounded single-GPU processes and records their terminal states.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shlex
import shutil
import signal
import subprocess
import sys
import time
import traceback


METHODS = ("adamw", "muon", "pd", "ts", "spd")
SEEDS = (20261001, 20261002)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(8<<20),b""):
            value.update(chunk)
    return value.hexdigest()


def write(path,value):
    path = Path(path)
    if path.exists():
        raise FileExistsError("Preserve existing execution record: "+str(path))
    with path.open("x") as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write("\n")


def make_initial(out,reference,preparation):
    from .qualify_d import prepared_data,verify_sources
    out,reference,preparation = map(lambda p:Path(p).resolve(),(out,reference,preparation))
    study = Path(__file__).resolve().parents[2]
    if not out.is_relative_to(study):
        raise ValueError("All output must stay inside tiny_surrogate")
    verify_sources(reference)
    qualification = read(reference/"report/results.json")
    if qualification["numerical_qualification_passed"] is not True:
        raise ValueError("The native five-method numerical qualification must pass")
    candidate,manifest,completion = prepared_data(study/"data/fineweb_d_20260928",
                                                  preparation/"PLAN.json",preparation)
    original = {read(p)["method"]:read(p) for p in (reference/"configs").glob("*.json")}
    if set(original) != set(METHODS):
        raise ValueError("Exactly the five qualified methods are required")
    out.mkdir(parents=True,exist_ok=False)
    for directory in ("configs","runs","console","workers"):
        (out/directory).mkdir()
    # Preserve the exact successfully qualified eager trainer, kernels and data
    # loader; the later failed graph implementation is absent from this source.
    shutil.copytree(reference/"frozen",out/"frozen")
    shutil.copy2(reference/"source_manifest.json",out/"source_manifest.json")
    shutil.copy2(__file__,out/"ordering_driver.py")
    configs,groups = [],[]
    for seed in SEEDS:
        group = []
        for method in METHODS:
            for multiplier in candidate["lr_multipliers"]:
                cfg = dict(original[method],seed=seed,lr=candidate["screen_lr_centers"][method]*multiplier,
                           total_tokens=candidate["base_total_tokens"],eval_every=8,keep_model_every=0)
                cfg["run_id"] = f"{method}_D_lr{cfg['lr']:g}_m0.9_s{seed}"
                path = out/"configs"/(cfg["run_id"]+".json")
                write(path,cfg)
                configs.append(cfg)
                group.append(dict(config=str(path),out=str(out/"runs"/cfg["run_id"])))
        groups.extend([[arm] for arm in group])
    random.Random(20260928).shuffle(groups)
    plan = dict(stage="full_development_ordering",candidate="reference_directed_D",methods=list(METHODS),
        seeds=list(SEEDS),initial_arms=30,maximum_edge_arms=10,maximum_outward_rate_per_method=1,
        original_candidate_plan_sha256=sha(preparation/"PLAN.json"),original_candidate_plan=candidate,
        reference_numerical_cohort=str(reference),reference_qualification_sha256=sha(reference/"report/results.json"),
        source_manifest_sha256=sha(out/"source_manifest.json"),data_completion_sha256=sha(study/"data/fineweb_d_20260928/PREPARATION_COMPLETE.json"),
        training_manifest_sha256=completion["training_manifest_sha256"],full_development_targets=manifest["dev_target_count"],
        full_development_windows=manifest["dev_full_windows"],total_tokens=96242176,batch_tokens=262144,updates=368,
        last_batch_tokens=35328,evaluation_steps=[0,1]+list(range(8,369,8)),selection="one LR per method by mean full-development endpoint NLL over both seeds",
        adjacent_order=list(METHODS),each_seed_difference_must_be_negative=True,maximum_adjacent_mean_difference=-.005,
        all_methods_must_be_bracketed=True,exact_ties_touching_any_boundary_remain_open=True,
        boundary_rule="At most one outward factor-two LR per method on both seeds. Tied minima at both ends fail the bracket without a discretionary direction choice. A new outer minimum remains a failed bracket.",
        unstable_rule="Only explicit nonfinite training loss/gradient failures after a completed finite update are classified numerical instability. Their LR pair loses with infinite selection loss; both seeds still run. Operational/source/data failures halt both workers, with no automatic retry.",
        declared_expected_initial_gpu_hours=12.412704144528579,declared_expected_maximum_gpu_hours=16.55027219270477,
        maximum_gpu_hours=None,maximum_concurrent_gpus=None,array_throttle=None,gpu="gpu:nvidia_rtx_a4000:1",
        scheduler_job_walltime_seconds=3600,worker_process_seconds=3540,
        scheduler_kill_grace_seconds=30,process_kill_grace_seconds=15,per_arm_timeout_seconds=3480,
        automatic_requeue=False,automatic_retries=False,edges_require_complete_initial_readout=True,
        user_authorization=["That’s fast enough. Check is optimizer ordering.",
                            "Feel free to use more than 2 gpus. Don’t impose weird limits."],
        prior_eight_hour_forecast_failure_preserved=True,prior_graph_fidelity_failure_preserved=True,
        scientific_parameters_changed_from_candidate_plan=False,sealed_test_scoring=False,
        ordering_success_does_not_complete_full_surrogate_goal=True)
    write(out/"PLAN.json",plan)
    write(out/"tasks_initial.json",groups)
    write(out/"PREFLIGHT.json",dict(plan_sha256=sha(out/"PLAN.json"),source_manifest_sha256=sha(out/"source_manifest.json"),
        controller_sha256=sha(out/"ordering_driver.py"),tasks_initial_sha256=sha(out/"tasks_initial.json"),
        initial_config_sha256={p.name:sha(p) for p in sorted((out/"configs").glob("*.json"))}))
    make_script(out,study,"initial",3540)
    (out/"README.md").write_text("# Candidate D optimizer ordering\n\n30 initial native-code runs: five methods, three fixed LRs, two paired seeds. Each array task uses one 16 GB A4000; there is no array throttle or arbitrary aggregate compute cap. Slurm schedules available qualified GPUs. Full-development NLL chooses one LR per method jointly across seeds. Up to ten predeclared edge arms are conditional on a complete initial readout. Every test panel remains sealed.\n\nExpected cost: 12.41 GPU-hours initially, 16.55 with all edges. Each job has an ordinary one-hour walltime for an expected 17–30-minute run. No automatic retries. Frozen numerical sources are copied unchanged from the successful 21-update qualification. The earlier eight-hour planning-limit failure and graph fidelity failure remain preserved.\n")
    return out


def make_script(cohort,study,stage,seconds):
    exit_code = "import json,sys,time; from pathlib import Path; p=Path(sys.argv[1]); p.write_text(json.dumps(dict(exit_code=int(sys.argv[2]),finished_unix=time.time()),indent=2)+'\\n')"
    script = "\n".join(["#!/bin/bash","set -uo pipefail",
        "export CC=/usr/bin/gcc CXX=/usr/bin/g++ OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1",
        f"timeout --signal=TERM --kill-after=15s {seconds} {shlex.quote(str(study/'run'))} {shlex.quote(str(cohort/'ordering_driver.py'))} worker {shlex.quote(str(cohort))} --stage {stage} --index \"$SLURM_ARRAY_TASK_ID\"",
        "ORDERING_EXIT=$?",
        f"{shlex.quote(str(study/'run'))} -c {shlex.quote(exit_code)} {shlex.quote(str(cohort/'workers'))}/{stage}_exit_\"$SLURM_ARRAY_TASK_ID\".json \"$ORDERING_EXIT\"",
        'exit "$ORDERING_EXIT"',""])
    (cohort/("job_"+stage+".sh")).write_text(script)


def verify(cohort,stage):
    preflight = read(cohort/"PREFLIGHT.json")
    for name,key in (("PLAN.json","plan_sha256"),("source_manifest.json","source_manifest_sha256"),
                     ("ordering_driver.py","controller_sha256")):
        if sha(cohort/name) != preflight[key]:
            raise ValueError("Frozen ordering identity changed: "+name)
    for relative,expected in read(cohort/"source_manifest.json").items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or sha(cohort/"frozen"/path) != expected:
            raise ValueError("Qualified numerical source changed")
    if stage == "initial":
        receipt = preflight
    elif stage == "edges":
        receipt = read(cohort/"EDGE_PREFLIGHT.json")
        if receipt["plan_sha256"] != preflight["plan_sha256"]:
            raise ValueError("Edge plan identity changed")
    else:
        raise ValueError("Unknown ordering stage")
    if sha(cohort/("tasks_"+stage+".json")) != receipt["tasks_"+stage+"_sha256"]:
        raise ValueError("Frozen task queue changed")
    for name,expected in receipt[stage+"_config_sha256"].items():
        if sha(cohort/"configs"/name) != expected:
            raise ValueError("Frozen run configuration changed")
    return read(cohort/"PLAN.json")


def classify_exit(root,returncode):
    """An unsupported/partial execution never becomes an unstable-LR result."""
    if returncode == 0 and (root/"status.json").is_file() and read(root/"status.json").get("status") == "complete":
        return "complete"
    if not all((root/name).is_file() for name in ("failure.json","metadata.json","metrics.jsonl")):
        return "operational_failure"
    failure = read(root/"failure.json")
    error = failure.get("error","")
    eligible = (error.startswith("FloatingPointError(") and "Nonfinite training loss" in error
                or error.startswith("RuntimeError(") and "total norm" in error and "non-finite" in error)
    if not eligible:
        return "operational_failure"
    records = [json.loads(line) for line in (root/"metrics.jsonl").read_text().splitlines()]
    if any(r.get("step",0) >= 1 and isinstance(r.get("train_nll"),(int,float))
           and math.isfinite(r["train_nll"]) for r in records):
        return "numerical_instability"
    return "operational_failure"


def worker(cohort,stage,index):
    cohort = Path(cohort).resolve()
    plan = verify(cohort,stage)
    groups = read(cohort/("tasks_"+stage+".json"))
    if type(index) is not int or not 0 <= index < len(groups):
        raise ValueError("Invalid array task index")
    group = groups[index]
    if len(group) != 1:
        raise ValueError("Each array task runs exactly one declared recipe")
    prefix = stage+"_"+str(index)
    write(cohort/"workers"/(prefix+"_STARTED.json"),dict(started_unix=time.time(),job_id=os.getenv("SLURM_JOB_ID"),
         array_job_id=os.getenv("SLURM_ARRAY_JOB_ID"),array_index=index,runs=len(group)))
    process = None

    def stop_child():
        if process is not None and process.poll() is None:
            process.terminate()
            try: process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)

    def interrupted(signum,frame):
        stop_child()
        raise SystemExit(128+signum)

    signal.signal(signal.SIGTERM,interrupted)
    signal.signal(signal.SIGINT,interrupted)
    try:
        for arm in group:
            if (cohort/"HALT.json").exists():
                raise RuntimeError("Other worker recorded an operational failure")
            verify(cohort,stage)
            cfg = read(arm["config"])
            root = Path(arm["out"])
            if root.exists() or cfg["seed"] not in SEEDS:
                raise ValueError("Existing run output or mismatched paired seed; no automatic retry")
            started = time.time()
            env = dict(os.environ,PYTHONPATH=str(cohort/"frozen"))
            with (cohort/"console"/(cfg["run_id"]+".log")).open("x") as stream:
                process = subprocess.Popen([sys.executable,"-m","research.tiny_spectra.train","--config",arm["config"],
                    "--out",arm["out"]],cwd=cohort/"frozen",env=env,stdout=stream,stderr=subprocess.STDOUT)
                while process.poll() is None:
                    if time.time()-started > plan["per_arm_timeout_seconds"] or (cohort/"HALT.json").exists():
                        stop_child()
                        raise RuntimeError("Per-arm deadline exceeded or peer operational failure")
                    time.sleep(2)
            terminal = classify_exit(root,process.returncode)
            if not root.exists():
                root.mkdir(parents=True)
            write(root/"ARM_EXECUTION.json",dict(status=terminal,returncode=process.returncode,
                started_unix=started,ended_unix=time.time(),seed=cfg["seed"],method=cfg["method"],lr=cfg["lr"],
                config_sha256=sha(arm["config"]),source_manifest_sha256=sha(cohort/"source_manifest.json")))
            print(json.dumps(dict(run=cfg["run_id"],status=terminal)),flush=True)
            if terminal == "operational_failure":
                raise RuntimeError("Operational failure in "+cfg["run_id"])
        write(cohort/"workers"/(prefix+"_COMPLETE.json"),dict(completed_unix=time.time(),runs=len(group)))
    except BaseException as error:
        stop_child()
        record = dict(error=repr(error),traceback=traceback.format_exc(),stage=stage,index=index,failed_unix=time.time())
        write(cohort/"workers"/(prefix+"_FAILURE.json"),record)
        try: write(cohort/"HALT.json",record)
        except FileExistsError: pass
        raise


def submit(cohort,stage):
    cohort = Path(cohort).resolve()
    plan = verify(cohort,stage)
    if (cohort/"HALT.json").exists():
        raise ValueError("Preserved operational failure blocks further submission")
    if stage == "edges" and not (cohort/"EDGE_PREFLIGHT.json").is_file():
        raise ValueError("Edges require a completed initial readout and frozen preflight")
    minutes = plan["scheduler_job_walltime_seconds"]//60
    limit = f"{minutes//60:02d}:{minutes%60:02d}:00"
    count = len(read(cohort/("tasks_"+stage+".json")))
    if count < 1:
        raise ValueError("No empty array submission")
    command = ["sbatch","--parsable","--partition=gpu","--nodes=1","--ntasks=1","--gres="+plan["gpu"],
        "--cpus-per-task=2","--mem=8G","--time="+limit,"--no-requeue",f"--array=0-{count-1}",
        "--job-name=tiny-D-"+stage,"--chdir="+str(cohort),"--output="+str(cohort/"console"/(stage+"_%A_%a.log")),
        str(cohort/("job_"+stage+".sh"))]
    write(cohort/("SUBMISSION_INTENT_"+stage+".json"),dict(command=command,created_unix=time.time()))
    result = subprocess.run(command,capture_output=True,text=True,check=True)
    (cohort/("submission_"+stage+".stdout")).write_text(result.stdout)
    write(cohort/("submission_"+stage+".json"),dict(command=command,job_id=result.stdout.strip(),submitted_unix=time.time()))
    print(result.stdout.strip(),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation",choices=("create","worker","submit"))
    parser.add_argument("cohort",type=Path)
    parser.add_argument("--reference",type=Path)
    parser.add_argument("--preparation",type=Path)
    parser.add_argument("--stage",choices=("initial","edges"),default="initial")
    parser.add_argument("--index",type=int)
    args = parser.parse_args()
    if args.operation == "create": print(make_initial(args.cohort,args.reference,args.preparation))
    elif args.operation == "worker": worker(args.cohort,args.stage,args.index)
    else: submit(args.cohort,args.stage)
