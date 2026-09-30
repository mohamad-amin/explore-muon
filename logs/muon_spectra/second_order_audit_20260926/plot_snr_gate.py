"""The SNR gate's online readings along training: tail (u < 1) mean Wiener weight and median SNR per step, and the
validation gap to PD and PD-top at the same α and hardware (2026-09-28 22:4x CDT).

usage: plot_snr_gate.py OUT_PNG GATE_ARM [PD_ARM PDTOP_ARM]   (arm directories relative to logs/muon_spectra)
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def rows(arm):
    out = []
    for f in sorted((ROOT / arm / "scientific" / "steps").glob("step*.json")):
        out.append(json.loads(f.read_text()))
    return out


def main():
    out, gate = sys.argv[1], sys.argv[2]
    refs = sys.argv[3:]
    g = rows(gate)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    steps = [r["step"] for r in g if r.get("snr_gate")]
    axes[0].plot(steps, [r["snr_gate"]["tail_mean_w"] for r in g if r.get("snr_gate")], label="tail mean w", color="C0")
    ax2 = axes[0].twinx()
    ax2.plot(steps, [r["snr_gate"]["tail_median_snr"] for r in g if r.get("snr_gate")], label="tail median SNR", color="C3", lw=0.8)
    ax2.set_yscale("log")
    ax2.set_ylabel("tail median SNR (per step, at the run's batch)", color="C3")
    axes[0].set(xlabel="step", ylabel="tail mean Wiener weight w", ylim=(0, 1.02), title=f"{Path(gate).name}: gate readings")
    val = {r["step"]: r["validation_nll"] for r in g if "validation_nll" in r}
    for ref in refs:
        rv = {r["step"]: r["validation_nll"] for r in rows(ref) if "validation_nll" in r}
        common = sorted(set(val) & set(rv) - {0})
        axes[1].plot(common, [val[s] - rv[s] for s in common], marker="o", ms=3, label=f"gate − {Path(ref).name}")
    axes[1].axhline(0, color="grey", lw=0.8)
    axes[1].set(xlabel="step", ylabel="validation loss difference", title="gate against its references")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()
