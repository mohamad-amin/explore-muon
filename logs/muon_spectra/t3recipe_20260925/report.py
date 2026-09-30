"""Track 3 recipe in our setup (wd 0.2 + cooldown 0.7 + batch 524,288): PD and PD+geo minus Muon, with the 1M-batch cells."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
W = HERE.parent / "wdgeo2_20260925"
ROWS = {"batch 524,288 (this cohort)": {m: HERE / f"{m}_t3recipe_s260925_a6000" for m in ("M", "PD", "PDgeo2")},
        "batch 1,048,576 (wdgeo2)": {"M": W / "M_wd0.2_cd0.7_s260925_a6000", "PD": W / "PD_wd0.2_cd0.7_s260925_a6000",
                                     "PDgeo2": W / "PDgeo2_wd0.2_cd0.7_s260925_a6000"}}


def curve(arm):
    pipeline = arm / "pipeline.json"
    phase = json.loads(pipeline.read_text()).get("phase") if pipeline.exists() else "missing"
    rows = [json.loads(p.read_text()) for p in sorted((arm / "scientific/steps").glob("step*.json"))] \
        if (arm / "scientific/steps").exists() else []
    return phase, {r["step"]: r["validation_nll"] for r in rows if "validation_nll" in r}


def main():
    for label, arms in ROWS.items():
        pm, vm = curve(arms["M"])
        print(f"{label}: Muon {'%.5f' % vm[max(vm)] if vm else '-'} [{pm}, step {max(vm) if vm else '-'}]")
        for m in ("PD", "PDgeo2"):
            phase, v = curve(arms[m])
            common = set(v) & set(vm)
            if common:
                last = max(common)
                mid = ", ".join(f"@{s} {v[s] - vm[s]:+.4f}" for s in (500, 1000, 1500, 2000, 2500) if s in common and s < last)
                print(f"  {m:7s} {v[last]:.5f}  {v[last] - vm[last]:+.4f}  {mid}  [{phase}, step {last}]")
            else:
                print(f"  {m:7s} {phase}")


if __name__ == "__main__":
    main()
