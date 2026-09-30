"""Step-equivalent speedup over Muon at 1M, 4M and 16M-token batches, from per-step training loss on unseen data.

Every training batch is new data (one pass), so each step's training loss before the update estimates the population
loss; it is smoothed with a centered running mean over ~3% of the run. For an arm and the Muon reference at the same
batch size, speedup(t) = t / the step at which the arm's smoothed loss first reaches the reference's at step t, reported
against the fraction of training t / total steps (warmup and the 10% cooldown excluded).

usage: speedup_by_batch.py OUT_JSON [OUT_PNG]
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REFERENCES = {   # batch: Muon candidates; the one with the lowest final validation is the reference
    "1M": ["soaudit_traj_20260926/M_lr0.007_s260925_l40s"],
    "4M β0.95": ["soaudit_batch4m_20260927/M_b4M_lr0.007_s260925_l40s", "soaudit_batch4m_20260927/M_b4M_lr0.014_s260925_l40s",
                 "soaudit_batch4m_20260927/M_b4M_lr0.028_s260925_l40s"],
    "4M": ["soaudit_mom4m_20260927/M_b4M_lr0.014_mom0.9_s260925_l40s", "soaudit_mom4m3_20260927/M_b4M_lr0.02_mom0.9_s260925_l40s"],
    "16M long": ["soaudit_b16mlong_20260928/M_b16M_T2x_lr0.02_mom0.9_s260925_l40s", "soaudit_b16mlong_lr_20260928/M_b16M_T2x_lr0.014_mom0.9_s260925_l40s"],
    "16M seed 260926": ["soaudit_seed16m_20260928/M_b16M_lr0.02_mom0.9_s260926_l40s"],
    "16M": ["soaudit_batch16m_20260927/M_b16M_lr0.014_mom0.9_s260925_l40s", "soaudit_batch16m_20260927/M_b16M_lr0.02_mom0.9_s260925_l40s",
            "soaudit_batch16m_20260927/M_b16M_lr0.028_mom0.9_s260925_ada", "soaudit_batch16m_20260927/M_b16M_lr0.04_mom0.9_s260925_ada",
            "soaudit_mom16m_20260928/M_b16M_lr0.02_mom0.8_s260925_l40s"],
}
GROUPS = {   # batch: (reference Muon arm, {label: arm})
    "1M": ("soaudit_traj_20260926/M_lr0.007_s260925_l40s",
           {"PD α¼ @0.01": "soaudit_traj_20260926/PD_a0.25_lr0.01_s260925_ada",
            "S∘PD α¼ @0.01": "soaudit_traj_20260926/SPD_a0.25_lr0.01_s260925_ada"}),
    "4M β0.95": ("soaudit_batch4m_20260927/M_b4M_lr0.014_s260925_l40s",
                 {"PD α¼ @0.02": "soaudit_batch4m_20260927/PD_b4M_lr0.02_s260925_ada",
                  "S∘PD α¼ @0.01": "soaudit_batch4m_20260927/SPD_b4M_lr0.01_s260925_a6000"}),
    "4M": ("soaudit_mom4m_20260927/M_b4M_lr0.014_mom0.9_s260925_l40s",
           {"PD α½ @0.02": "soaudit_mom4m4_20260927/PD_a0.5_b4M_lr0.02_mom0.9_s260925_ada",
            "TS ½/½ @0.02": "soaudit_strength4m_20260927/TS_a0.5_b0.5gn_b4M_lr0.02_mom0.9_s260925_ada",
            "S∘PD α½ @0.02": "soaudit_strength4m_20260927/SPD_a0.5_b4M_lr0.02_mom0.9_s260925_l40s"}),
    "16M": ("soaudit_batch16m_20260927/M_b16M_lr0.028_mom0.9_s260925_ada",
            {"S∘PD α½ @0.04": "soaudit_batch16m_20260927/SPD_a0.5_b16M_lr0.04_mom0.9_s260925_ada",
             "TS ½/½ @0.04": "soaudit_batch16m_20260927/TS_a0.5_b0.5gn_b16M_lr0.04_mom0.9_s260925_ada",
             "PD α½ @0.04": "soaudit_batch16m_20260927/PD_a0.5_b16M_lr0.04_mom0.9_s260925_l40s",
             "S∘PD α½ @0.028": "soaudit_batch16m_20260927/SPD_a0.5_b16M_lr0.028_mom0.9_s260925_ada",
             "TS ½/½ @0.028": "soaudit_batch16m_20260927/TS_a0.5_b0.5gn_b16M_lr0.028_mom0.9_s260925_l40s",
             "PD α½ @0.028": "soaudit_batch16m_20260927/PD_a0.5_b16M_lr0.028_mom0.9_s260925_l40s",
             "Muon @0.02": "soaudit_batch16m_20260927/M_b16M_lr0.02_mom0.9_s260925_l40s",
             "Muon @0.04": "soaudit_batch16m_20260927/M_b16M_lr0.04_mom0.9_s260925_ada",
             "S_left∘PD α½ @0.028": "soaudit_soapside16m_20260928/SleftPD_a0.5_b16M_lr0.028_mom0.9_s260925_ada",
             "S_right∘PD α½ @0.028": "soaudit_soapside16m_20260928/SrightPD_a0.5_b16M_lr0.028_mom0.9_s260925_l40s",
             "S∘PD α½ @0.028 β0.8": "soaudit_prefilter16m_20260928/SPD_a0.5_b16M_lr0.028_mom0.8_s260925_ada",
             "S∘PD α½ @0.028 β0.7": "soaudit_mom16m_20260928/SPD_a0.5_b16M_lr0.028_mom0.7_s260925_ada",
             "PD α½ @0.028 β0.8": "soaudit_mom16m_20260928/PD_a0.5_b16M_lr0.028_mom0.8_s260925_l40s",
             "Muon @0.02 β0.8": "soaudit_mom16m_20260928/M_b16M_lr0.02_mom0.8_s260925_l40s"}),
    "16M seed 260926": ("soaudit_seed16m_20260928/M_b16M_lr0.02_mom0.9_s260926_l40s",
                        {"S∘PD α½ @0.028": "soaudit_seed16m_20260928/SPD_a0.5_b16M_lr0.028_mom0.9_s260926_ada"}),
    "16M long": ("soaudit_b16mlong_20260928/M_b16M_T2x_lr0.02_mom0.9_s260925_l40s",
                 {"S∘PD α½ @0.028": "soaudit_b16mlong_20260928/SPD_a0.5_b16M_T2x_lr0.028_mom0.9_s260925_ada",
                  "S∘PD α½ @0.02": "soaudit_b16mlong_lr_20260928/SPD_a0.5_b16M_T2x_lr0.02_mom0.9_s260925_ada"}),
}


def train_curve(arm):
    rows = {}
    for f in (ROOT / arm / "scientific" / "steps").glob("step*.json"):
        r = json.loads(f.read_text())
        if "train_nll" in r:
            rows[r["step"]] = r["train_nll"]
    if not rows:
        return None, None
    steps = np.array(sorted(rows))
    loss = np.array([rows[s] for s in steps])
    width = max(1, int(round(0.03 * steps[-1])) | 1)
    pad = width // 2
    padded = np.pad(loss, pad, mode="edge")
    smooth = np.convolve(padded, np.ones(width) / width, mode="valid")
    return steps, smooth


def final_validation(arm):
    best = None
    for f in (ROOT / arm / "scientific" / "steps").glob("step*.json"):
        r = json.loads(f.read_text())
        if "validation_nll" in r and (best is None or r["step"] > best[0]):
            best = (r["step"], r["validation_nll"])
    return best


def speedup(ref, arm):
    rs, rl = ref
    as_, al = arm
    total = rs[-1]
    out = []
    for t, target in zip(rs, rl):
        frac = t / total
        if frac < 0.08 or frac > 0.9:
            continue
        below = np.nonzero(al <= target)[0]
        if not len(below):
            continue
        i = below[0]
        if i == 0:
            continue
        f = (al[i - 1] - target) / (al[i - 1] - al[i])
        step = as_[i - 1] + f * (as_[i] - as_[i - 1])
        out.append([round(frac, 4), round(t / step, 4), round(float(target), 4)])
    return out


def main():
    out = {}
    for batch, (ref_arm, arms) in GROUPS.items():
        finals = {a: final_validation(a) for a in REFERENCES[batch]}
        finals = {a: f for a, f in finals.items() if f and f[0] > 0 and f[0] >= max(v[0] for v in finals.values() if v)}
        if finals:
            ref_arm = min(finals, key=lambda a: finals[a][1])
        ref = train_curve(ref_arm)
        if ref[0] is None:
            continue
        entry = {"reference": ref_arm, "reference_final": final_validation(ref_arm), "arms": {}}
        for label, arm in arms.items():
            curve = train_curve(arm)
            if curve[0] is None or curve[0][-1] < ref[0][-1] or arm == ref_arm:
                continue
            entry["arms"][label] = {"speedup": speedup(ref, curve), "final": final_validation(arm)}
        out[batch] = entry
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1) + "\n")
    for batch, entry in out.items():
        print(batch, "reference final", entry["reference_final"])
        for label, a in entry["arms"].items():
            sp = a["speedup"]
            pick = []
            for x in (0.25, 0.5, 0.75, 0.88):
                near = [v for f, v, _ in sp if abs(f - x) < 0.03]
                pick.append(f"{x:.0%} {sum(near) / len(near):.2f}x" if near else f"{x:.0%} -")
            print("   ", label.ljust(18), "final %.4f" % a["final"][1] if a["final"] else "", " | ".join(pick))
    if len(sys.argv) > 2:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(out) + 1, figsize=(6 * (len(out) + 1), 4.6), squeeze=False)
        styles = {"1M": ":", "4M": "--", "4M β0.95": "-.", "16M": "-", "16M long": "-", "16M seed 260926": "-"}
        for batch, entry in out.items():     # matched-loss view: speedup against the reference's (smoothed) loss
            for label, a in entry["arms"].items():
                if a["speedup"] and any(k in label for k in ("S∘PD", "PD α¼", "TS")):
                    _, v, loss = zip(*a["speedup"])
                    axes[0][-1].plot(loss, v, styles[batch], label=f"{batch}: {label}")
        axes[0][-1].invert_xaxis()
        axes[0][-1].axhline(1, color="k", lw=0.5)
        axes[0][-1].set_xlabel("Muon's (smoothed) training loss at step t")
        axes[0][-1].set_ylabel("step-equivalent speedup")
        axes[0][-1].set_title("matched loss: speedup vs where training is")
        axes[0][-1].legend(fontsize=7)
        for ax, (batch, entry) in zip(axes[0], out.items()):
            for label, a in entry["arms"].items():
                if a["speedup"]:
                    f, v, _ = zip(*a["speedup"])
                    ax.plot(f, v, "-", label=label)
            ax.axhline(1, color="k", lw=0.5)
            ax.set_title(f"{batch}: speedup over {Path(entry['reference']).name.split('_s26')[0]}")
            ax.set_xlabel("fraction of training (Muon's step / total)")
            ax.set_ylabel("step-equivalent speedup")
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(sys.argv[2], dpi=110)
        print(sys.argv[2])


if __name__ == "__main__":
    main()
