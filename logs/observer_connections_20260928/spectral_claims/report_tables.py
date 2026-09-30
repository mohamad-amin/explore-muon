"""Compact tables generated from audit.py; all values are observed quadratures."""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent


def main():
    rows = json.loads((OUT / "results.json").read_text())
    group_order = [(16, 46), (16, 83), (4, 183), (1, 500)]
    arm_order = ["M", "PD", "SPD", "TS", "S", "SPD_b08"]
    selected = sorted([r for r in rows if (r["batch"], r["step"]) in group_order and r["arm"] in arm_order],
                      key=lambda r: (group_order.index((r["batch"], r["step"])), arm_order.index(r["arm"])))
    lines = ["# Computed evidence tables", "", "M = Muon; SPD = S∘PD; S = SOAP∘Muon. Band: eigenvalue < 1e-4 × own-state top Ritz value, held fixed across prefixes. These are quadrature estimates, not exact GN-projector measurements.", "",
             "| Batch / step | Arm | Top Ritz | Energy at48 (%) | Slope share at32 /40 /48 (%) | c* at32 /40 /48 | PSD lower bound on band energy (%) |",
             "|---|---|---:|---:|---|---|---:|"]
    for r in selected:
        p = r["prefixes"]
        v = p["48"]["relative_1e-4"]
        slopes = " / ".join(f'{100*p[k]["relative_1e-4"]["slope_share"]:.2f}' for k in ("32", "40", "48"))
        cs = " / ".join(f'{p[k]["relative_1e-4"]["cstar"]:.2f}' for k in ("32", "40", "48"))
        lines.append(f'| {r["batch"]}M / {r["step"]} | {r["arm"]} | {r["top"]:.2f} | {100*v["energy_share"]:.2f} | {slopes} | {cs} | {100*r["energy_lower_bounds"]["relative_1e-4"]:.1f} |')
    lines += ["", "The PSD bound is max(0, 1 − DᵀGD / (cutoff × ||D||²)). It is valid for the sampled PSD GN without resolving the projector; a weak bound does not establish a low energy share.", "",
              "| 16M @46 arm | Cutoff own-relative | First Ritz node | Slope share, abs 1e-4 (%) | abs 1e-3 (%) | abs 1e-2 (%) | shared Muon cutoff (%) |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for r in selected:
        if (r["batch"], r["step"]) != (16, 46):
            continue
        vals = [100*r["prefixes"]["48"][k]["slope_share"] for k in ("absolute_1e-4", "absolute_1e-3", "absolute_1e-2", "shared_muon_relative_1e-4")]
        lines.append(f'| {r["arm"]} | {r["cutoffs"]["relative_1e-4"]:.6g} | {r["first_nodes"][0]:.6g} | ' + " | ".join(f"{v:.2f}" for v in vals) + " |")
    lines += ["", "Zero at abs 1e-4 usually means that the quadrature contains no node there; it does not show that the original GN has no spectral mass or descent there.", ""]
    (OUT / "TABLES.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
