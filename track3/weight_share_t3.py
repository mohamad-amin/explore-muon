"""Where do the final weights sit relative to the input spectrum? (read-only; one GPU)

For each hidden matrix W of a final Track 3 checkpoint, the input second moment C = E[x x^T] is measured
on the model itself (32 fresh validation sequences, position 0 excluded, as in gamma_probe_t3.py). With
q_j the eigenvectors of C (descending eigenvalue), report the share of ||W||_F^2 that lies along the top
1, 8 and 64 input directions, sum_{j<=k} ||W q_j||^2 / ||W||_F^2 (weights spread uniformly give k/d).

Decay-equilibrium prediction (MUON_CASE.md, 20:45 CDT): under decoupled decay PD's weights are depleted
along the high-variance inputs where PD steps less; geometry decay restores them.

Usage: python track3/weight_share_t3.py LABEL=track3/logs/<uuid>_final.pt [...]   (prints a table)
"""
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gamma_probe_t3 as probe   # the exact #36 architecture and the probe's held-out tokens

if not torch.cuda.is_available():   # CPU fallback (the probe module defaults to CUDA)
    probe.device = torch.device("cpu")
    torch.set_num_threads(8)

KINDS = {"q": "attn.q", "k": "attn.k", "v": "attn.v", "o": "attn.proj", "up": "mlp.fc", "down": "mlp.proj"}
SOURCE = {"q": "attn_in", "k": "attn_in", "v": "attn_in", "o": "o", "up": "up", "down": "down"}   # q, k, v share inputs
TOPS = (1, 8, 64)


@torch.no_grad()
def shares(checkpoint):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = probe.GPT(vocab_size=50304, num_layers=12, model_dim=768).to(probe.device)
    model.embed = model.embed.float()
    model.load_state_dict({k: v.float() for k, v in saved["model"].items()})
    model.eval()
    x, y = probe.tokens()
    modules = {(b, kind): model.blocks[b].get_submodule(path) for b in range(12) for kind, path in KINDS.items()}
    hooked = {(b, SOURCE[kind]): m for (b, kind), m in modules.items()}   # one hook per distinct input
    sums = {key: 0 for key in hooked}
    counts = {key: 0 for key in hooked}

    def hook(key):
        def record(module, args):
            a = args[0][:, 1:].reshape(-1, args[0].shape[-1]).double()
            sums[key] = sums[key] + a.T @ a
            counts[key] += a.shape[0]
        return record

    handles = [m.register_forward_pre_hook(hook(key)) for key, m in hooked.items()]
    for i in range(0, len(x), probe.MICRO):
        model(x[i:i + probe.MICRO], y[i:i + probe.MICRO])
    for h in handles:
        h.remove()
    bases = {}
    for key in hooked:
        cov = sums[key] / counts[key]
        cov = cov if probe.device.type == "cuda" else cov.float()   # FP32 eigh on CPU is ample for the top 64
        bases[key] = torch.linalg.eigh(cov)[1].flip(1).double()
    table = {}
    for (b, kind), module in modules.items():
        vectors = bases[(b, SOURCE[kind])]
        W = module.weight.double()
        along = (W @ vectors).square().sum(0) / W.square().sum()   # share of ||W||^2 per input direction
        table.setdefault(kind, []).append([float(along[:k].sum()) for k in TOPS])
    return {kind: [sum(r[i] for r in rows) / len(rows) for i in range(len(TOPS))] for kind, rows in table.items()}


def main():
    results = {}
    for arg in sys.argv[1:]:
        label, checkpoint = arg.split("=", 1)
        results[label] = shares(Path(checkpoint))
        print(f"done {label}", flush=True)
    print("share of ||W||_F^2 along the top-k input eigendirections, mean over the 12 blocks (uniform: k/d)")
    print(f"{'kind':>5} " + " ".join(f"{label[:14] + ' k=' + str(k):>22}" for label in results for k in TOPS))
    for kind in KINDS:
        print(f"{kind:>5} " + " ".join(f"{results[label][kind][i]:>22.4f}" for label in results for i in range(len(TOPS))))


if __name__ == "__main__":
    main()
