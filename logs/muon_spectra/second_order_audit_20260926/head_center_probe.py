"""Is the head input's anisotropy its mean? Uncentered vs centered second moment of the final normalized hidden state,
and the mean direction's share (2026-09-28, after the whitened-head arm's early deficit). CPU.

usage: head_center_probe.py ARM_DIR:STEP [...]
"""
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
from research.adamw_spectra import gn_probe as P  # noqa: E402
from research.adamw_spectra.data import TokenStream  # noqa: E402
from one_step_gn import CURV  # noqa: E402

torch.set_num_threads(16)
for item in sys.argv[1:]:
    arm, step = item.rsplit(":", 1)
    model, _ = P.load_checkpoint(Path(arm) / "scientific" / "kept" / f"step{int(step):06d}.pt", torch.device("cpu"))
    T = model.config.seq_len
    rows = []
    handle = model.head.register_forward_pre_hook(lambda m, a: rows.append(a[0][:, 1:].reshape(-1, a[0].shape[-1]).double()))
    val = TokenStream(str(REPO / "data/fineweb10B/fineweb_val_*.bin"))
    with torch.no_grad():
        for first in range(0, 32, 8):
            model(val.batch(CURV + first * T, 8, T, torch.device("cpu"))[0])
    handle.remove()
    h = torch.cat(rows)
    mean = h.mean(0)
    second = h.T @ h / h.shape[0]
    cov = second - torch.outer(mean, mean)
    ev_u = torch.linalg.eigvalsh(second).flip(0)
    ev_c = torch.linalg.eigvalsh(cov).flip(0)
    top_vec = torch.linalg.eigh(second)[1][:, -1]
    print(f"{arm.split('/')[-1][:30]} @{step}: |mean|^2 / tr = {float(mean @ mean / second.trace()):.3f}; "
          f"cos(mean, top eigvec) = {abs(float((mean / mean.norm()) @ top_vec)):.3f}; "
          f"uncentered top/mean {float(ev_u[0] / ev_u.mean()):.1f}, PR {float(ev_u.sum() ** 2 / (ev_u ** 2).sum() / len(ev_u)):.3f}; "
          f"centered top/mean {float(ev_c[0] / ev_c.mean()):.1f}, PR {float(ev_c.sum() ** 2 / (ev_c ** 2).sum() / len(ev_c)):.3f}", flush=True)
