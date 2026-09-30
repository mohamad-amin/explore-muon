"""Exact totals do not qualify band projectors: an analytic two-dimensional case."""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent


def main():
    eigenvalues = np.array([1e-5, 1.0])
    step = np.array([1.0, 0.1])
    gradient = np.array([0.0, -1.0])
    cutoff = 0.02
    energy, slope, curvature = step**2, step * gradient, eigenvalues * step**2
    theta1 = curvature.sum() / energy.sum()
    true_low = eigenvalues < cutoff
    true = {"energy": float(energy[true_low].sum()), "slope": float(slope[true_low].sum()),
            "curvature": float(curvature[true_low].sum())}
    one_node = {"energy": float(energy.sum()), "slope": float(slope.sum()),
                "curvature": float(curvature.sum())}
    for row in (true, one_node):
        row["energy_share"] = row["energy"] / energy.sum()
        row["slope_share"] = row["slope"] / slope.sum()
        row["curvature_share"] = row["curvature"] / curvature.sum()
        row["cstar"] = -row["slope"] / row["curvature"]
    # The sole Ritz node is below the cutoff. All totals are exact, yet
    # its apparent flat-band slope comes entirely from the stiff direction.
    assert theta1 < cutoff
    assert true["cstar"] == 0 and one_node["cstar"] > 9
    out = {"eigenvalues": eigenvalues.tolist(), "step": step.tolist(),
           "gradient": gradient.tolist(), "cutoff": cutoff,
           "one_node_ritz_value": float(theta1), "exact_flat_band": true,
           "one_node_reported_flat_band": one_node,
           "interpretation": "A counterexample to the logical implication exact totals => exact spectral bands; not a model of the 48-step GN error."}
    (OUT / "synthetic.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
