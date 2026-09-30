"""Run wave-7 arms one after another on one allocation (MUON_CASE.md wave 7; copied from wave 2).

Usage (inside an srun step with 4 GPUs): node_queue.py NAME WAIT_FOR_ARM_DIR ARM [ARM ...]
Waits until WAIT_FOR_ARM_DIR/pipeline.json is complete or failed, then runs each arm's
controller and records exit codes in queue_NAME.json; a failed arm does not stop the queue.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
name, wait_for, arms = sys.argv[1], sys.argv[2], sys.argv[3:]
path = ROOT / f"queue_{name}.json"
record = {"queue": name, "wait_for": wait_for, "arms": arms, "started_unix": time.time(), "runs": []}


def save():
    path.write_text(json.dumps(record, indent=2) + "\n")


save()
while True:
    try:
        phase = json.loads((Path(wait_for) / "pipeline.json").read_text())["phase"]
    except (OSError, ValueError, KeyError):
        phase = ""
    if phase in ("complete", "failed"):
        break
    time.sleep(30)
record["waited_until_unix"] = time.time()
save()
for arm in arms:
    started = time.time()
    with open(ROOT / arm / "controller.log", "xb") as log:
        code = subprocess.call([str(REPO / ".venv/bin/python"), "-u", str(ROOT / "frozen/adamw_spectra/run_arm.py"),
                                "--root", str(ROOT / arm), "--world-size", "4"],
                               cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
    record["runs"].append({"arm": arm, "exit": code, "started_unix": started, "ended_unix": time.time()})
    save()
record["ended_unix"] = time.time()
save()
