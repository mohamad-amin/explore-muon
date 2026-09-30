"""Earliest step at which a Track 3 configuration passes the benchmark's significance rule.

Track 3 rule 3: (3.28 - avg_loss) * num_runs**0.5 >= 0.004 over all non-cherry-picked runs; early stopping is allowed
if the stopping step is the same for every run (rule 5), so the late validations (every 25 steps from 90% of training)
of full-length runs can be read at any of those steps.

Runs are pooled only if the code logged at the top of their logfile is identical to SCRIPT (the logfile starts with
the script source, then a line of 100 '='). Only finished runs count; every finished run of the script is included.

Usage: python track3/earliest_step.py [SCRIPT]   (default train_gpt_pd_pdwd_a0.125.py, #36 schedule, 3250 steps)
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LINE = re.compile(r"^step:(\d+)/(\d+) val_loss:([\d.]+)", re.M)


def runs(script):
    source = (HERE / script).read_text()
    for path in sorted((HERE / "logs").glob("*.txt")):
        if path.name.startswith(("console", "queue")):
            continue
        text = path.read_text(errors="replace")
        code = text.split("\n" + "=" * 100 + "\n", 1)[0]
        if code.rstrip("\n") != source.rstrip("\n"):
            continue
        points = {int(m[1]): float(m[3]) for m in LINE.finditer(text)}
        total = int(next(LINE.finditer(text))[2])
        gpu = re.search(r"on NVIDIA ([^\n]*?) with world_size", text)
        yield path.name[:8], (gpu[1] if gpu else "?"), total, points


def main():
    script = sys.argv[1] if len(sys.argv) > 1 else "train_gpt_pd_pdwd_a0.125.py"
    found = list(runs(script))
    finished = [r for r in found if r[2] in r[3]]
    print(f"{script}: {len(found)} runs with identical code, {len(finished)} finished")
    for name, gpu, total, points in found:
        status = "finished" if total in points else f"running (step {max(points)})"
        late = ", ".join(f"{s}: {points[s]:.5f}" for s in sorted({3150, 3175, 3200, total}) if s in points)
        print(f"  {name} {gpu[:22]:22s} {status:20s} {late}")
    if not finished:
        return
    total = finished[0][2]
    steps = sorted(s for s in set.intersection(*(set(p) for _, _, _, p in finished)) if s >= 0.9 * total)
    print(f"\n{'step':>6} {'n':>3} {'mean':>9} {'(3.28-mean)*sqrt(n)':>20}  passes (>= 0.004)")
    earliest = None
    for s in steps:
        values = [p[s] for _, _, _, p in finished]
        n, mean = len(values), sum(values) / len(values)
        score = (3.28 - mean) * n ** 0.5
        if score >= 0.004 and earliest is None:
            earliest = s
        print(f"{s:>6} {n:>3} {mean:>9.5f} {score:>20.4f}  {'yes' if score >= 0.004 else 'no'}")
    print(f"\nearliest step passing the Track 3 criterion with the {len(finished)} finished runs: {earliest}")


if __name__ == "__main__":
    main()
