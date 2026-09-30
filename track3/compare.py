"""Compare Track 3 validation curves: our runs (track3/logs) against the published #36 baseline runs.

Usage: python track3/compare.py [LOG ...]   (default: every Track 3 log in track3/logs)
The #36 reference is the mean over its 10 published H100 logs (records/.../20260610_tuned_baseline_3250).
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
UPSTREAM = Path("/home-nfs/mohamadamin/.tmp/claude-1457/-share-data-dl-theory-amin-projects-explore-muon/"
                "740185e2-c249-4cf1-b369-ce504ecea942/scratchpad/mng/records/track_3_optimization/results/"
                "20260610_tuned_baseline_3250")
LINE = re.compile(r"^step:(\d+)/(\d+) val_loss:([\d.]+)")
MARKS = (250, 500, 1000, 1500, 2000, 2500, 2925, 3000, 3050, 3100, 3150, 3200, 3250)


def curve(path):
    """{step: val_loss} and a label (script name and device) from one Track 3 logfile."""
    text = path.read_text(errors="replace")
    points = {int(m[1]): float(m[3]) for m in map(LINE.match, text.splitlines()) if m}
    device = re.search(r"on (NVIDIA [^\n]*?) with world_size (\d+)", text)
    muon = re.search(r"optimizer2 = Muon\(\[p for p in model\.blocks\.parameters\(\) if p\.ndim >= 2\],"
                     r"\s*lr=([\d.]+), weight_decay=([\d.]+)", text)
    alpha = re.search(r"^pd_alpha, pd_damping, pd_refresh = ([\d.]+)", text, re.M)
    cooldown = re.search(r"def set_hparams\(step, cooldown_frac=([\d.]+)\)", text)
    name = ("PD" if "partial data-norm Muon" in text else "Muon #36") + (f" lr{muon[1]} wd{muon[2]}" if muon else "")
    name += f" a{alpha[1]}" if alpha and alpha[1] != "0.25" else ""
    name += f" cd{cooldown[1]}" if cooldown and cooldown[1] != "0.7" else ""
    gpu = device[1].replace("NVIDIA ", "").replace(" Generation", "") if device else ""
    return points, f"{name} ({gpu} x{device[2]})" if device else name


def first_below(points, target=3.28):
    return next((s for s in sorted(points) if s > 0 and points[s] < target), None)


def main():
    logs = [Path(a) for a in sys.argv[1:]] or sorted(p for p in (HERE / "logs").glob("*.txt")
                                                     if not p.name.startswith(("console_", "queue")))
    runs = [curve(p) for p in logs]
    reference = [curve(p)[0] for p in sorted(UPSTREAM.glob("*.txt"))]
    reference = [r for r in reference if 3250 in r]
    mean = {s: sum(r[s] for r in reference) / len(reference) for s in MARKS if all(s in r for r in reference)}
    header = f"{'step':>6} | " + " | ".join(f"{label[:28]:>28}" for _, label in runs) + f" | {'#36 mean (n=%d)' % len(reference):>16}"
    print(header)
    for s in MARKS:
        cells = [f"{points[s]:>28.5f}" if s in points else f"{'':>28}" for points, _ in runs]
        print(f"{s:>6} | " + " | ".join(cells) + (f" | {mean[s]:>16.5f}" if s in mean else ""))
    for points, label in runs:
        print(f"{label}: first val < 3.28 at step {first_below(points)}; final {points.get(max(points))} at {max(points)}")


if __name__ == "__main__":
    main()
