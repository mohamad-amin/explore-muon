"""Step lead of one run over another at matched loss (review, 2026-09-28 19:56 CDT: quote effects in steps, not loss).

Each step's training loss before the update is on unseen data. Both curves are smoothed with a centered running mean
(--half on each side). The lead of A over B at step t is (the first, interpolated step at which B's smoothed loss
reaches A's smoothed loss at t) − t: positive when A is ahead. If B never reaches it, the lead is reported as ≥ B's
last step − t.

usage: step_lead.py ARM_A ARM_B [--steps 46,75,83] [--half 3]   (arm directories relative to logs/muon_spectra or absolute)
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def curve(arm, half):
    path = Path(arm) if Path(arm).is_absolute() else ROOT / arm
    rows = {}
    for f in (path / "scientific" / "steps").glob("step*.json"):
        r = json.loads(f.read_text())
        if "train_nll" in r:
            rows[r["step"]] = r["train_nll"]
    steps = np.array(sorted(rows))
    loss = np.array([rows[s] for s in steps])
    smooth = np.array([loss[max(0, i - half):i + half + 1].mean() for i in range(len(loss))])
    return steps, smooth


def lead(steps_a, loss_a, steps_b, loss_b, t):
    target = float(np.interp(t, steps_a, loss_a))
    below = np.nonzero(loss_b <= target)[0]
    if len(below) == 0:
        return None, float(steps_b[-1] - t)
    j = below[0]
    if j == 0:
        return float(steps_b[0] - t), None
    s0, s1, l0, l1 = steps_b[j - 1], steps_b[j], loss_b[j - 1], loss_b[j]
    reach = s0 + (l0 - target) / max(l0 - l1, 1e-12) * (s1 - s0)
    return float(reach - t), None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("a")
    parser.add_argument("b")
    parser.add_argument("--steps", default="46,75,83")
    parser.add_argument("--half", type=int, default=3)
    args = parser.parse_args()
    sa, la = curve(args.a, args.half)
    sb, lb = curve(args.b, args.half)
    out = {}
    for t in [int(x) for x in args.steps.split(",")]:
        if t > sa[-1]:
            continue
        value, at_least = lead(sa, la, sb, lb, t)
        out[t] = round(value, 1) if value is not None else f">={at_least:.0f}"
    print(json.dumps({"a": args.a.split("/")[-1], "b": args.b.split("/")[-1], "lead_steps": out}))


if __name__ == "__main__":
    main()
