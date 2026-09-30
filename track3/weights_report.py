"""Final-weight norms of Track 3 checkpoints, by matrix kind (CPU; reads track3/logs/*_final.pt).

For each run: mean Frobenius and spectral norm of q/k/v/o/up/down over the 12 blocks. When the checkpoint
carries PD input statistics, also the share of each matrix's squared norm that lies along the top 1% of
its input eigendirections (tr(P W^T W) / ||W||^2 for the projector P onto them), i.e. whether the weights
are small along the high-variance inputs that PD updates less.
"""
import re
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
KINDS = {"q": "attn.q", "k": "attn.k", "v": "attn.v", "o": "attn.proj", "up": "mlp.fc", "down": "mlp.proj"}
STATS = {"q": "attn.q", "k": "attn.q", "v": "attn.q", "o": "attn.proj", "up": "mlp.fc", "down": "mlp.proj"}


def label(checkpoint):
    """Run label as in screen.py (optimizer, LR/WD and PD variant, parsed from the logged code)."""
    from screen import label as parse
    log = checkpoint.with_name(checkpoint.name.replace("_final.pt", ".txt"))
    text = log.read_text(errors="replace") if log.exists() else ""
    name, muon = parse(text)
    if muon:
        name += f" lr{muon[1]}" + ("" if name.startswith(("MuonH", "PD-H")) else f" wd{muon[2]}")
    steps = re.search(r"^    train_steps = (\d+)", text, re.M)
    return f"{name} ({steps[1] if steps else '?'} steps)"


def report(checkpoint):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    weights, stats = saved["model"], saved.get("pd_stats", {})
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tops = {}   # top-1% input eigenvectors per statistics module (q, k and v share attn.q), computed once
    for name, s in stats.items():
        cov = (s["cov"] / s["weight"].clamp_min(1e-12)).float().to(device)
        _, vectors = torch.linalg.eigh(0.5 * (cov + cov.T))
        tops[name] = vectors[:, -max(1, cov.shape[0] // 100):]
    rows = {}
    for kind, path in KINDS.items():
        fro, spec, top = [], [], []
        for b in range(12):
            W = weights[f"blocks.{b}.{path}.weight"].float().to(device)
            fro.append(float(W.norm()))
            spec.append(float(torch.linalg.svdvals(W)[0]))
            P = tops.get(f"blocks.{b}.{STATS[kind]}")
            if P is not None:
                top.append(float((W @ P).square().sum() / W.square().sum()))
        rows[kind] = (sum(fro) / 12, sum(spec) / 12, sum(top) / len(top) if top else None)
    return rows


def main():
    files = [Path(a) for a in sys.argv[1:]] or sorted(HERE.glob("logs/*_final.pt"))
    for f in files:
        print(f"\n{label(f)}  [{f.name[:8]}]")
        print(f"{'kind':>5} {'||W||_F':>9} {'||W||_2':>9} {'top-1% input share':>19}")
        for kind, (fro, spec, top) in report(f).items():
            print(f"{kind:>5} {fro:>9.2f} {spec:>9.2f} {'' if top is None else f'{top:>19.4f}'}")


if __name__ == "__main__":
    main()
