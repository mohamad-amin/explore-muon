"""How much of each optimizer's path is net progress? Displacement between kept states over the path length (2026-09-28).

For an arm with kept states s1 < s2 (and the weights one step after each, W_{s+1}), per hidden matrix and in total:
  efficiency = |W_s2 - W_s1| / ((s2 - s1) x mean single-step norm)
with the single-step norm the mean of |W_{s+1} - W_s| at s1 and s2 (normalized maps take near-constant step norms at a
constant LR). 1 = every step points the same way; 1/sqrt(n) = a random walk; oscillation along stiff directions lowers
it. Also the loss drop between the states per unit of net displacement. CPU only.

usage: displacement_efficiency.py OUT_JSON ARM_DIR:S1:S2 [...]
"""
import json
import math
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
KINDS = ("attn.q", "attn.k", "attn.v", "attn.o", "mlp.up", "mlp.down")


def hidden(path):
    s = torch.load(path, map_location="cpu", weights_only=False)
    s = s.get("model", s)
    return {k.replace("_orig_mod.", ""): v.float() for k, v in s.items() if k.replace("_orig_mod.", "").startswith("blocks.") and v.ndim == 2}


def loss_at(arm, step):
    f = Path(arm) / "scientific" / "steps" / f"step{step + 1:06d}.json"
    return json.loads(f.read_text())["train_nll"] if f.exists() else None


def main():
    out = Path(sys.argv[1])
    result = {}
    for item in sys.argv[2:]:
        arm, s1, s2 = item.rsplit(":", 2)
        s1, s2 = int(s1), int(s2)
        kept = Path(arm) / "scientific" / "kept"
        a, b = hidden(kept / f"step{s1:06d}.pt"), hidden(kept / f"step{s2:06d}.pt")
        a1, b1 = hidden(kept / f"step{s1 + 1:06d}_weights.pt"), hidden(kept / f"step{s2 + 1:06d}_weights.pt")
        per_kind, total_d, total_s = {}, 0.0, 0.0
        for k in a:
            d = float((b[k] - a[k]).pow(2).sum())
            step = 0.5 * (float((a1[k] - a[k]).norm()) + float((b1[k] - b[k]).norm()))
            kind = next(x for x in KINDS if x in k)
            per_kind.setdefault(kind, [0.0, 0.0])
            per_kind[kind][0] += d
            per_kind[kind][1] += ((s2 - s1) * step) ** 2
            total_d += d
            total_s += ((s2 - s1) * step) ** 2
        l1, l2 = loss_at(arm, s1), loss_at(arm, s2)
        result[item] = {"efficiency": math.sqrt(total_d / total_s), "displacement": math.sqrt(total_d),
                        "path": math.sqrt(total_s), "steps": s2 - s1, "loss": [l1, l2],
                        "loss_drop_per_displacement": (l1 - l2) / math.sqrt(total_d) if l1 and l2 else None,
                        "by_kind": {kind: math.sqrt(v[0] / v[1]) for kind, v in per_kind.items()}}
        r = result[item]
        kinds = ", ".join("%s: %.3f" % (k.split(".")[-1], v) for k, v in r["by_kind"].items())
        print(f"{arm.split('/')[-1][:34]:34s} {s1}->{s2}: efficiency {r['efficiency']:.3f} (random walk {1 / math.sqrt(s2 - s1):.3f})  "
              f"displacement {r['displacement']:.2f} path {r['path']:.1f}  loss {l1:.3f}->{l2:.3f}  by kind {{{kinds}}}", flush=True)
    out.write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
