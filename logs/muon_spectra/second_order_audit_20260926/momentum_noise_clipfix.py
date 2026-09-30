"""Clip-corrected momentum-noise fractions for direction_snr_probe (v2) and direction_drift_probe outputs
(2026-09-28 21:2x CDT).

Both probes estimate the noise energy in the momentum buffer M as r / (1 - beta^2): white gradient noise at today's
level, entering M unscaled. With gradient clipping, the gradient of update s enters M multiplied by
c_s = min(1, clip / |g_s|) (|g_s| = gradient_norm_before_clip in the run's step record for step s). The checkpoint at
step N holds M_N = sum_k beta^k c_{N-k} g_{N-k}, so its noise energy is r sum_k beta^(2k) c_{N-k}^2. The correction
factor applied to the reported fractions is (1 - beta^2) sum_k beta^(2k) c_{N-k}^2 (noise level assumed constant).

usage: momentum_noise_clipfix.py JSON [JSON ...]
"""
import json
import sys
from pathlib import Path


def factor(arm, step, beta, clip):
    total = 0.0
    for k in range(step):
        s = step - k
        path = Path(arm) / "scientific" / "steps" / f"step{s:06d}.json"
        if not path.exists():
            continue
        norm = json.loads(path.read_text()).get("gradient_norm_before_clip")
        c = 1.0 if not norm or clip <= 0 else min(1.0, clip / norm)
        total += beta ** (2 * k) * c * c
    return (1 - beta ** 2) * total


def main():
    for name in sys.argv[1:]:
        r = json.loads(Path(name).read_text())
        config = json.loads((Path(r["arm"]) / "config.json").read_text())
        f = factor(r["arm"], r["step"], r["beta"], config["grad_clip"])
        rows = r.get("bins") or r.get("bins_all")
        tail = [b for b in rows if b["u_hi"] <= 1 and b["count"] >= 20]
        top = [b for b in rows if b["u_lo"] >= 1 and b["count"] >= 20]
        pooled = lambda bs: (sum(b["momentum_noise_fraction"] * b.get("momentum_energy", 1) for b in bs)  # noqa: E731
                             / max(sum(b.get("momentum_energy", 1) for b in bs), 1e-30))
        print(f"{Path(name).parent.name}/{Path(name).stem}: beta {r['beta']} clip factor {f:.3f}  "
              f"M noise tail {min(b['momentum_noise_fraction'] for b in tail) * f:.2f}-{max(b['momentum_noise_fraction'] for b in tail) * f:.2f}  "
              f"top {min(b['momentum_noise_fraction'] for b in top) * f:.2f}-{max(b['momentum_noise_fraction'] for b in top) * f:.2f}"
              + (f"  all {r['all']['momentum_noise_fraction'] * f:.2f}" if "all" in r else ""))


if __name__ == "__main__":
    main()
