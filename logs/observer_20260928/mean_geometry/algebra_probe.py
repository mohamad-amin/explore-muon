"""Checkpoint-free algebra and small JSON reads; no training or GPU calls."""
import json
import os
from pathlib import Path

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "2"
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
base = ROOT / "logs/muon_spectra"
paths = {
    "baseline": base / "soaudit_batch16m_20260927/PD_a0.5_b16M_lr0.028_mom0.9_s260925_l40s",
    "head_half": base / "soaudit_headwhite16m_20260928/PDhw0.5_a0.5_b16M_lr0.028_mom0.9_s260925_l40s",
    "head_quarter": base / "soaudit_headwhite16m_20260928/PDhw0.25_a0.5_b16M_lr0.028_mom0.9_s260925_l40s",
    "center_half": base / "soaudit_headcenter16m_20260928/PDhwc0.5_a0.5_b16M_lr0.028_mom0.9_s260925_l40s",
}
results = {"source_dirs": {k: str(v) for k, v in paths.items()}, "curves": {}}
for name, path in paths.items():
    rows = []
    for step in (1, 2, 3, 5, 10, 15, 20, 30, 46, 60, 75, 83, 92):
        file = path / "scientific/steps" / f"step{step:06d}.json"
        if file.exists():
            data = json.loads(file.read_text())
            rows.append({k: data.get(k) for k in ("step", "train_nll", "gradient_norm_before_clip", "validation_nll")})
    results["curves"][name] = rows

# A centered covariance does not preserve the mean direction's Adam step.
cov_center = np.diag([4.0, 1.0])
mean = np.array([1.0, 0.0])
root = np.diag((np.diag(cov_center) / np.trace(cov_center) * 2 + 1e-3) ** -0.5)
grad = np.array([[1.0, 1.0]])
plain = grad / (np.abs(grad) + 1e-8)
white_step = (grad @ root) / (np.abs(grad @ root) + 1e-8)
update = white_step @ root
update *= np.linalg.norm(white_step) / np.linalg.norm(update)
results["centering_counterexample"] = {
    "centered_cov": cov_center.tolist(), "mean": mean.tolist(),
    "plain_adam_step": plain.tolist(), "centered_whitened_step": update.tolist(),
    "mean_response_ratio": float((update @ mean)[0] / (plain @ mean)[0]),
}

# Fixed non-diagonal R: matching ||white Adam|| is not matching ||ordinary Adam||.
angle = np.pi / 4
q = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
fixed_root = q @ np.diag([1.0, 8.0]) @ q.T
grads = [np.array([[1.0, 10.0], [2.0, -1.0]]), np.array([[-1.0, 10.0], [2.0, 1.0]])]
m, vo, vw = (np.zeros((2, 2)) for _ in range(3))
rows = []
for t, g in enumerate(grads, 1):
    m = .9 * m + .1 * g
    vo = .95 * vo + .05 * g ** 2
    vw = .95 * vw + .05 * (g @ fixed_root) ** 2
    plain = (m / (1 - .9 ** t)) / (np.sqrt(vo / (1 - .95 ** t)) + 1e-8)
    white = (m / (1 - .9 ** t) @ fixed_root) / (np.sqrt(vw / (1 - .95 ** t)) + 1e-8)
    rows.append({"step": t, "ordinary_norm": float(np.linalg.norm(plain)),
                 "whitened_coordinate_norm": float(np.linalg.norm(white)),
                 "ratio": float(np.linalg.norm(white) / np.linalg.norm(plain))})
results["norm_matching_counterexample"] = {"fixed_root": fixed_root.tolist(), "rows": rows}

# The covariance EMA is updated per microbatch forward, not per optimizer step.
results["covariance_memory"] = {}
for name, batch in (("1M", 1048576), ("16M", 16777216)):
    forwards = batch / (4 * 16 * 512)
    decay = .998 ** forwards
    results["covariance_memory"][name] = {"forwards_per_rank_per_step": forwards,
        "effective_step_decay": decay, "stationary_lag_steps": decay / (1 - decay)}

(HERE / "evidence.json").write_text(json.dumps(results, indent=2) + "\n")
print(json.dumps({k: v for k, v in results.items() if k not in ("source_dirs", "curves")}, indent=2))
