"""Run arms of one cohort one after another on one group of 4 GPUs (adapted from the TTIC node_queue.py).

Usage on a VM, from the project root:
    .venv/bin/python cloud/lane.py COHORT_DIR NAME GPUS ARM [ARM ...]
Each arm runs the cohort's frozen run_arm.py; exit codes go to COHORT_DIR/lane_NAME.json.
A failed arm does not stop the lane.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
cohort, name, gpus, arms = Path(sys.argv[1]).resolve(), sys.argv[2], sys.argv[3], sys.argv[4:]
path = cohort / f"lane_{name}.json"
record = {"lane": name, "gpus": gpus, "arms": arms, "host": os.uname().nodename,
          "started_unix": time.time(), "runs": []}


def save():
    path.write_text(json.dumps(record, indent=2) + "\n")


save()
for arm in arms:
    started = time.time()
    with open(cohort / arm / "controller.log", "xb") as log:
        code = subprocess.call([str(REPO / ".venv/bin/python"), "-u", str(cohort / "frozen/adamw_spectra/run_arm.py"),
                                "--root", str(cohort / arm), "--world-size", "4"],
                               cwd=REPO, env={**os.environ, "CUDA_VISIBLE_DEVICES": gpus},
                               stdout=log, stderr=subprocess.STDOUT)
    record["runs"].append({"arm": arm, "exit": code, "started_unix": started, "ended_unix": time.time()})
    save()
record["ended_unix"] = time.time()
save()
