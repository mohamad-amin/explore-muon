"""Local fix screen in the Track 3 recipe setting: every arm vs Muon@0.007 and vs PD + geometry decay p=2 (t3recipe)."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROUND1 = HERE.parent / "t3screen_20260926"
ROUND2 = HERE.parent / "t3screen2_20260926"
REF = HERE.parent / "t3recipe_20260925"
ARMS = {"Muon@0.007 (t3recipe)": REF / "M_t3recipe_s260925_a6000", "PD decoupled (t3recipe)": REF / "PD_t3recipe_s260925_a6000",
        "PDgeo2 (t3recipe)": REF / "PDgeo2_t3recipe_s260925_a6000",
        **{p.name.replace("_t3recipe_s260925_a6000", ""): p for p in sorted(ROUND1.glob("*_t3recipe_s260925_a6000"))},
        **{"r2 " + p.name.replace("_t3recipe_s260925_a6000", ""): p for p in sorted(ROUND2.glob("*_t3recipe_s260925_a6000"))},
        **{"r3 " + p.name.replace("_t3recipe_s260925_a6000", ""): p for p in sorted(HERE.glob("*_t3recipe_s260925_a6000"))}}


def curve(arm):
    pipeline = arm / "pipeline.json"
    phase = json.loads(pipeline.read_text()).get("phase") if pipeline.exists() else "missing"
    rows = [json.loads(p.read_text()) for p in sorted((arm / "scientific/steps").glob("step*.json"))] \
        if (arm / "scientific/steps").exists() else []
    return phase, {r["step"]: r["validation_nll"] for r in rows if "validation_nll" in r}


def main():
    _, muon = curve(ARMS["Muon@0.007 (t3recipe)"])
    _, geo = curve(ARMS["PDgeo2 (t3recipe)"])
    print(f"{'arm':>26} {'final/last':>10} {'vs Muon@0.007':>14} {'vs PDgeo2':>10}  status")
    for name, arm in ARMS.items():
        phase, v = curve(arm)
        if not v:
            print(f"{name:>26} {'':>10} {'':>14} {'':>10}  {phase}")
            continue
        last = max(v)
        dm = f"{v[last] - muon[last]:+.4f}" if last in muon else ""
        dg = f"{v[last] - geo[last]:+.4f}" if last in geo else ""
        print(f"{name:>26} {v[last]:>10.5f} {dm:>14} {dg:>10}  {phase}, step {last}")


if __name__ == "__main__":
    main()
