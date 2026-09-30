"""WD x cooldown grid in the Track 3 recipe setting: PD + geometry decay (alpha 1/8) and Muon (LR 0.01), and PD - Muon."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CENTER = HERE.parent / "t3screen_20260926"
METHODS = {"PD+geo a1/8": "PDgeo2_a0.125", "Muon@0.01": "M_lr0.01"}
WDS, CDS = (0.1, 0.2, 0.3), (0.5, 0.7, 0.9)


def arm(method, wd, cd):
    """Center from t3screen (A6000); corners from this cohort, any GPU suffix (two cells moved to g20 / priv-g14)."""
    if (wd, cd) == (0.2, 0.7):
        return CENTER / f"{method}_t3recipe_s260925_a6000"
    found = sorted(HERE.glob(f"{method}_wd{wd}_cd{cd}_t3recipe_s260925_*"))
    return found[0] if found else HERE / f"{method}_wd{wd}_cd{cd}_t3recipe_s260925_missing"


def final(path):
    pipeline = path / "pipeline.json"
    phase = json.loads(pipeline.read_text()).get("phase") if pipeline.exists() else "missing"
    rows = [json.loads(p.read_text()) for p in sorted((path / "scientific/steps").glob("step*.json"))] \
        if (path / "scientific/steps").exists() else []
    v = {r["step"]: r["validation_nll"] for r in rows if "validation_nll" in r}
    return (v.get(2938), phase, max(v) if v else None)


def main():
    table = {m: {(wd, cd): final(arm(key, wd, cd)) for wd in WDS for cd in CDS
                 if (wd, cd) in ((0.2, 0.7), (0.1, 0.5), (0.1, 0.9), (0.3, 0.5), (0.3, 0.9))} for m, key in METHODS.items()}
    for m in METHODS:
        print(f"\n{m}: final val NLL (rows WD, columns cooldown); unfinished cells show [phase, last step]")
        print("   WD  " + "".join(f"{'cd ' + str(cd):>22}" for cd in CDS))
        for wd in WDS:
            cells = []
            for cd in CDS:
                r = table[m].get((wd, cd))
                gpu = arm(METHODS[m], wd, cd).name.rsplit("_", 1)[-1]
                cells.append(f"{'':>22}" if r is None else (f"{'%.5f' % r[0] + ' (' + gpu + ')':>22}" if r[0] else f"{'[' + str(r[1]) + ', ' + str(r[2]) + ']':>22}"))
            print(f"  {wd:<4}" + "".join(cells))
    print("\nPD+geo - Muon@0.01 per cell (center -0.0102):")
    for wd in WDS:
        cells = []
        for cd in CDS:
            a, b = table["PD+geo a1/8"].get((wd, cd)), table["Muon@0.01"].get((wd, cd))
            cells.append(f"{a[0] - b[0]:>+22.4f}" if a and b and a[0] and b[0] else f"{'':>22}")
        print(f"  {wd:<4}" + "".join(cells))


if __name__ == "__main__":
    main()
