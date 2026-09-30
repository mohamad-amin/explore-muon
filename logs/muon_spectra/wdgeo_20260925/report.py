"""Decay-geometry report: each arm - Muon@0.007 wd0.2 (transfer_20260925), with PD decoupled wd0.2 and the reference pair."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
T = HERE.parent / "transfer_20260925"
MUON = T / "M_wd0.2_s260925_a6000"
ARMS = {"PD wd0.2 decoupled": T / "PD_wd0.2_s260925_a6000",
        **{p.name.replace("_s260925_a6000", ""): p for p in sorted(HERE.glob("PD_*_s260925_a6000"))}}


def curve(arm):
    pipeline = arm / "pipeline.json"
    phase = json.loads(pipeline.read_text()).get("phase") if pipeline.exists() else "missing"
    rows = [json.loads(p.read_text()) for p in sorted((arm / "scientific/steps").glob("step*.json"))] \
        if (arm / "scientific/steps").exists() else []
    return phase, {r["step"]: r["validation_nll"] for r in rows if "validation_nll" in r}


def main():
    pm, vm = curve(MUON)
    print(f"Muon@0.007 wd0.2: {pm}, last step {max(vm) if vm else None}; reference pair PD-Muon at wd0.01: -0.0172")
    print(f"{'arm':>22} {'PD':>9} {'PD-Muon':>8}  mid-run PD-Muon")
    for name, arm in ARMS.items():
        phase, vp = curve(arm)
        common = set(vm) & set(vp)
        if not common:
            print(f"{name:>22}  {phase}")
            continue
        last = max(common)
        mid = ", ".join(f"@{s} {vp[s] - vm[s]:+.4f}" for s in (250, 500, 1000) if s in common and s < last)
        status = "" if phase == pm == "complete" else f" [{phase}, step {last}]"
        print(f"{name:>22} {vp[last]:>9.5f} {vp[last] - vm[last]:>+8.4f}  {mid}{status}")


if __name__ == "__main__":
    main()
