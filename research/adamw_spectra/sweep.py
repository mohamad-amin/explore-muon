"""Bounded paired LR pilots; select by held-out NLL, never by spectral shape."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .train import atomic_json, load_config, run


RATES = (0.0003, 0.0006, 0.0012)


def select_candidate(candidates):
    valid = [c for c in candidates if c.get("valid")]
    if not valid:
        raise ValueError("No valid LR candidate; scientific run is not qualified")
    return min(valid, key=lambda c: (c["score"], c["learning_rate"]))


def sweep(config, out, device="cuda", steps=300):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a fresh sweep directory; existing pilots are preserved")
    every = config["validation_every"]
    if steps < 3 * every or steps % every:
        raise ValueError("Pilot must end on an evaluation, with at least three evaluations")
    out.mkdir(parents=True, exist_ok=True)
    eval_steps = [steps - 2 * every, steps - every, steps]
    candidates = []
    fig, ax = plt.subplots(figsize=(7, 4))
    for lr in RATES:
        pilot = {**config, "seed": config["seed"] - 1, "learning_rate": lr, "spectra_every": 0}
        folder = out / f"lr_{lr:g}"
        candidate = {"learning_rate": lr, "valid": False, "pilot_seed": pilot["seed"]}
        try:
            status = run(pilot, folder, device=device, stop_after=steps)
            if status["step"] != steps or status["status"] != "stopped_at_requested_step":
                raise ValueError("Pilot did not run the intended prefix of the full budget")
            records = [json.loads(p.read_text()) for p in sorted((folder / "steps").glob("step*.json"))]
            evaluations = {r["step"]: r["validation_nll"] for r in records if "validation_nll" in r}
            scores = [evaluations[s] for s in eval_steps]
            if not np.isfinite(scores).all():
                raise ValueError("Non-finite selection loss")
            ax.plot(list(evaluations), list(evaluations.values()), label=f"LR={lr:g}")
            candidate.update(valid=True, score=float(np.mean(scores)), evaluation_nll=scores,
                             clipping_fraction=float(np.mean([r["gradient_clipped"] for r in records[1:]])))
        except (RuntimeError, ValueError) as exc:
            candidate["error"] = f"{type(exc).__name__}: {exc}"
        candidates.append(candidate)
        atomic_json(out / "candidates.json", candidates)
    ax.set(xlabel="step", ylabel="validation NLL", title="Paired AdamW learning-rate pilots")
    if any(c["valid"] for c in candidates):
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / "validation.png", dpi=150)
    plt.close(fig)
    selected = select_candidate(candidates)
    report = {"evaluation_steps": eval_steps, "validation_tokens": config["validation_tokens"],
              "criterion": "minimum mean validation NLL; exact tie chooses lower LR",
              "candidates": candidates, "selected_learning_rate": selected["learning_rate"],
              "boundary_winner": selected["learning_rate"] in (min(RATES), max(RATES)),
              "scope": "Early-learning operating point; not a certificate of endpoint-optimal LR",
              "maximum_pilot_tokens": len(RATES) * steps * config["batch_tokens"]}
    atomic_json(out / "selection.json", report)
    atomic_json(out / "selected_config.json", {**config, "learning_rate": selected["learning_rate"]})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(json.dumps(sweep(load_config(args.config), args.out, args.device), indent=2))


if __name__ == "__main__":
    main()
