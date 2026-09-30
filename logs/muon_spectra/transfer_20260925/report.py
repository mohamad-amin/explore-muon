"""Transfer-test report: PD@0.01 - Muon@0.007 with one Track 3 feature, vs the reference pair (-0.0172)."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REF = {"M": HERE.parent / "improve_w4_20260925/M_lr0.007_s260925_a6000",
       "PD": HERE.parent / "improve_w5_20260925/PD_a0.25_lr0.01_s260925_a6000"}


def final(arm):
    pipeline = arm / "pipeline.json"
    phase = json.loads(pipeline.read_text()).get("phase") if pipeline.exists() else "missing"
    steps = sorted((arm / "scientific/steps").glob("step*.json")) if (arm / "scientific/steps").exists() else []
    rows = [json.loads(p.read_text()) for p in steps]
    val = {r["step"]: r["validation_nll"] for r in rows if "validation_nll" in r}
    return phase, val


def main():
    features = sorted({p.name.split("_")[1] for p in HERE.glob("M_*_s260925_a6000")})
    pairs = {"reference": REF, **{f: {m: HERE / f"{m}_{f}_s260925_a6000" for m in ("M", "PD")} for f in features}}
    print(f"{'feature':>10} {'Muon':>9} {'PD':>9} {'PD-Muon':>8} {'kept':>6}  mid-run PD-Muon")
    for feature, arms in pairs.items():
        (pm, vm), (pp, vp) = final(arms["M"]), final(arms["PD"])
        last = max(set(vm) & set(vp)) if set(vm) & set(vp) else None
        if last is None:
            print(f"{feature:>10}  Muon {pm}, PD {pp}")
            continue
        delta = vp[last] - vm[last]
        marks = [s for s in (250, 500, 1000, 2000) if s in vm and s in vp and s < last]
        mid = ", ".join(f"@{s} {vp[s] - vm[s]:+.4f}" for s in marks)
        status = "" if pm == pp == "complete" else f" [{pm}/{pp}, step {last}]"
        print(f"{feature:>10} {vm[last]:>9.5f} {vp[last]:>9.5f} {delta:>+8.4f} {delta / -0.0172:>6.2f}  {mid}{status}")


if __name__ == "__main__":
    main()
