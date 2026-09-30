"""Screen Track 3 runs: group by train_steps and report each run against the Muon control(s) of its group.

Usage: python track3/screen.py [--steps]     (reads every Track 3 logfile in track3/logs)
  --steps: show each run's lead in steps instead of loss differences, i.e. how many steps later the
  reference control reaches the run's validation loss (interpolated on the control's curve; + = ahead).
Labels come from the logged code: optimizer, Muon lr/wd, PD alpha/damping, layer subset, decay variant.
Runs are grouped by (train_steps, family): #36-style decoupled decay, or #37-style hyperball ("H").
Each group also shows the mean of that family's 10 published H100 logs (#36 or #37) as a reference row.
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LINE = re.compile(r"^step:(\d+)/(\d+) val_loss:([\d.]+)", re.M)
RESULTS = Path("/home-nfs/mohamadamin/.tmp/claude-1457/-share-data-dl-theory-amin-projects-explore-muon/"
               "740185e2-c249-4cf1-b369-ce504ecea942/scratchpad/mng/records/track_3_optimization/results")
PUBLISHED = {"": RESULTS / "20260610_tuned_baseline_3250", "H": RESULTS / "20260611_muonh_tuned_aux_3250"}


def label(text):
    muon = re.search(r"optimizer2 = Muon\(\[p for p in model\.blocks\.parameters\(\) if p\.ndim >= 2\],"
                     r"\s*lr=([\d.]+), weight_decay=([\d.]+)", text)
    hyper = "def scale_invariant_update_(" in text
    if hyper:
        lr = re.search(r"optimizer2 = MuonH\(\[p for p in model\.blocks\.parameters\(\) if p\.ndim == 2\], lr=([\d.]+)", text)
        muon = (None, lr[1], "0") if lr else None
    if "# MG control: Muon's own update direction" in text:   # checked first: its docstring does not say PD
        return "MG (Muon dir + PD-geometry decay)", muon
    if "partial data-norm Muon" not in text:
        return ("MuonH" if hyper else "Muon"), muon
    alpha, damping = re.search(r"^pd_alpha, pd_damping, pd_refresh = ([\d.e-]+), ([\d.e-]+)", text, re.M).groups()
    if "def spd_update(" in text:   # SOAP-Muon family: alpha = 0 is SOAP-Muon, otherwise S∘PD
        parts = ["SOAP-Muon" if alpha in ("0.0", "0") else "S∘PD"]
        if "PD-geometry weight decay" in text and alpha not in ("0.0", "0"):
            parts.append("pdwd")
        return " ".join(parts), muon
    parts = ["PD-H" if hyper else "PD"]
    if alpha != "0.25":
        parts.append(f"a{alpha}")
    if damping != "1e-3":
        parts.append(f"damp{damping}")
    subset = re.search(r"PD only on (\w+)\n", text)
    if subset:
        parts.append(f"{subset[1]}-only")
    if "PD-geometry weight decay, first power" in text:
        parts.append("pdwd1")
    elif "PD-geometry weight decay" in text:
        parts.append("pdwd")
    return " ".join(parts), muon


def runs():
    for path in sorted(HERE.glob("logs/*.txt")):
        if path.name.startswith(("console_", "queue")):
            continue
        text = path.read_text(errors="replace")
        points = {int(m[1]): float(m[3]) for m in LINE.finditer(text)}
        total = next((int(m[2]) for m in LINE.finditer(text)), None)
        device = re.search(r"on NVIDIA ([^\n]*?) with world_size (\d+)", text)
        name, muon = label(text)
        if muon:
            name += f" lr{muon[1]}" + ("" if name.startswith(("MuonH", "PD-H")) else f" wd{muon[2]}")
        cooldown = re.search(r"def set_hparams\(step, cooldown_frac=([\d.]+)\)", text)
        if cooldown and cooldown[1] != "0.7":
            name += f" cd{cooldown[1]}"
        gpu = device[1].replace(" Generation", "").replace("GeForce ", "") if device else "?"
        family = "H" if "def scale_invariant_update_(" in text else ""
        batch = re.search(r"^batch_size = (\d+) \* 64 \* 1024", text, re.M)
        if batch and batch[1] != "8":   # non-#36 batch: its own group (diagnostic runs)
            family += f" batch {int(batch[1]) * 64 * 1024:,}"
        yield {"name": name, "gpu": gpu, "total": total, "points": points, "file": path.name[:8], "family": family}


def published(family, total):
    """Mean curve of a family's 10 published H100 logs, if they ran this many steps."""
    curves = []
    folder = PUBLISHED.get(family)   # diagnostic families (other batch sizes) have no published reference
    for path in sorted(folder.glob("*.txt")) if folder and folder.exists() else []:
        lines = list(LINE.finditer(path.read_text(errors="replace")))
        points = {int(m[1]): float(m[3]) for m in lines}
        if lines and int(lines[0][2]) == total and total in points:
            curves.append(points)
    if not curves:
        return None
    return {s: sum(c[s] for c in curves) / len(curves) for s in curves[0] if all(s in c for c in curves)}, len(curves)


def lead(ref, step, value):
    """Steps by which a run at `step` with loss `value` is ahead of the reference curve `ref` (+ = ahead)."""
    points = sorted((s, v) for s, v in ref.items() if s > 0)
    for (s0, v0), (s1, v1) in zip(points, points[1:]):
        if v0 >= value >= v1 and v0 > v1:
            return s0 + (s1 - s0) * (v0 - value) / (v0 - v1) - step
    return None


def main():
    steps = "--steps" in sys.argv[1:]
    groups = {}
    for run in runs():
        if run["total"]:
            groups.setdefault((run["total"], run["family"]), []).append(run)
    for (total, family), members in sorted(groups.items()):
        controls = [r for r in members if r["name"].startswith("Muon")]
        reference = published(family, total)
        if reference:
            members = members + [{"name": f"published #{37 if family else 36} H100 mean (n={reference[1]})",
                                  "gpu": "H100", "total": total, "points": reference[0], "file": "", "family": family}]
        marks = sorted({s for r in members for s in r["points"] if s and (s % 250 == 0 or s == total)})
        print(f"\n== {total} steps{(' (' + family.replace('H', 'hyperball family', 1).strip() + ')') if family else ''} "
              f"({len(members)} rows; controls: {len(controls)})")
        print(f"{'run':44s} {'gpu':14s} {'final':>8s} " + " ".join(f"{'@' + str(s):>8s}" for s in marks if s != total))
        controls.sort(key=lambda r: -len(r["points"]))   # most complete first
        public = reference[0] if reference else None

        def ref_for(r, s):
            """Reference curve for run r at step s: the pooled mean of this family's own controls that reached s
            (RTX 6000 Ada and L40S are the same AD102 chip; single controls differ by up to 0.005), else the
            published mean. For the step view the reference must also extend past s."""
            if r in controls or r["name"].startswith("published"):
                return None
            have = [c["points"] for c in controls if s in c["points"]]
            if have:
                steps_all = sorted(set.intersection(*(set(c) for c in have)))
                return {t: sum(c[t] for c in have) / len(have) for t in steps_all}
            return public if public and s in public else None

        for r in sorted(members, key=lambda r: r["points"].get(total, 99)):
            last = max(r["points"]) if r["points"] else 0
            final = r["points"].get(total)
            cells = []
            for s in marks:
                if s == total:
                    continue
                v = r["points"].get(s)
                ref = ref_for(r, s)
                if steps and v is not None and ref:
                    ahead = lead(ref, s, v)
                    cells.append(f"{ahead:>+8.0f}" if ahead is not None else f"{'?':>8s}")
                else:
                    cells.append(f"{v - ref[s]:>+8.4f}" if (v is not None and ref) else (f"{v:>8.4f}" if v else f"{'':>8s}"))
            ref = ref_for(r, total)
            final_s = f"{final:>8.5f}" if final else f"{'(' + str(max(r['points'])) + ')':>8s}"
            delta = f" ({final - ref[total]:+.4f})" if (final and ref) else ""
            print(f"{r['name'][:44]:44s} {r['gpu'][:14]:14s} {final_s}{delta:10s} " + " ".join(cells))
        print("cells: difference from the pooled mean of the group's Muon/MuonH controls that reached the step "
          "(else the published mean); plain values for controls")


if __name__ == "__main__":
    main()
