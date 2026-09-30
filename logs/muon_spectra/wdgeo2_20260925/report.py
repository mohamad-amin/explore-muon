"""2x2 and cooldown report (MUON_CASE.md 22:30 CDT): {Muon, PD} x {decoupled, geometry decay} at wd 0.2, cooldown 0.1 and 0.7."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
S = HERE.parent
CELLS = {  # cooldown -> method -> arm directory
    "cd0.1": {"Muon": S / "transfer_20260925/M_wd0.2_s260925_a6000", "PD": S / "transfer_20260925/PD_wd0.2_s260925_a6000",
              "PD+geo": S / "wdgeo_20260925/PD_wd0.2_geo2_s260925_a6000", "Muon+geo": HERE / "MG_wd0.2_s260925_a6000"},
    "cd0.7": {"Muon": HERE / "M_wd0.2_cd0.7_s260925_a6000", "PD": HERE / "PD_wd0.2_cd0.7_s260925_a6000",
              "PD+geo": HERE / "PDgeo2_wd0.2_cd0.7_s260925_a6000", "Muon+geo": HERE / "MG_wd0.2_cd0.7_s260925_a6000"},
}


def curve(arm):
    pipeline = arm / "pipeline.json"
    phase = json.loads(pipeline.read_text()).get("phase") if pipeline.exists() else "missing"
    rows = [json.loads(p.read_text()) for p in sorted((arm / "scientific/steps").glob("step*.json"))] \
        if (arm / "scientific/steps").exists() else []
    return phase, {r["step"]: r["validation_nll"] for r in rows if "validation_nll" in r}


def main():
    print("wd 0.2, seed 260925, A6000; each cell: final val NLL and difference from Muon (decoupled) at the same cooldown")
    for cd, cells in CELLS.items():
        pm, vm = curve(cells["Muon"])
        print(f"\n{cd}: Muon {'%.5f' % vm[max(vm)] if vm else '-'} [{pm}, step {max(vm) if vm else '-'}]")
        for method in ("PD", "PD+geo", "Muon+geo"):
            phase, v = curve(cells[method])
            common = set(v) & set(vm)
            if not common:
                print(f"  {method:9s} {phase}")
                continue
            last = max(common)
            mid = ", ".join(f"@{s} {v[s] - vm[s]:+.4f}" for s in (500, 1000) if s in common and s < last)
            print(f"  {method:9s} {v[last]:.5f}  {v[last] - vm[last]:+.4f}  {mid}  [{phase}, step {last}]")


if __name__ == "__main__":
    main()
