"""How anisotropic is the unembedding's input? The final normalized hidden state's second moment C_h, against the hidden
matrices' inputs that PD whitens (2026-09-28). CPU is enough.

usage: head_input_probe.py OUT_JSON ARM_DIR:STEP [...]
"""
import json
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


def summary(cov):
    values = torch.linalg.eigvalsh(cov.double()).clamp_min(0).flip(0)
    unit = values / values.mean()
    return {"top_over_mean": float(unit[0]), "participation_ratio": float(values.sum() ** 2 / (values ** 2).sum()) / len(values),
            "top1_share": float(values[0] / values.sum()), "top8_share": float(values[:8].sum() / values.sum()),
            "cond_1e-3": float((unit[0] + 1e-3) / (unit[-1] + 1e-3)), "unit_quantiles": [float(unit[int(q * (len(unit) - 1))]) for q in (0.0, 0.1, 0.5, 0.9, 1.0)]}


def main():
    out, results = Path(sys.argv[1]), {}
    torch.set_num_threads(16)
    for item in sys.argv[2:]:
        arm, step = item.rsplit(":", 1)
        model, _ = P.load_checkpoint(Path(arm) / "scientific" / "kept" / f"step{int(step):06d}.pt", torch.device("cpu"))
        T = model.config.seq_len
        captured = []
        handle = model.head.register_forward_pre_hook(lambda m, a: captured.append(a[0].detach()))
        recorder = P.Recorder(model)
        val = TokenStream(str(REPO / "data/fineweb10B/fineweb_val_*.bin"))
        sums = {}
        with torch.no_grad():
            for first in range(0, 32, 8):
                x, _ = val.batch(CURV + first * T, 8, T, torch.device("cpu"))
                model(x)
                h = captured[-1][:, 1:].reshape(-1, captured[-1].shape[-1]).double()
                sums["head"] = sums.get("head", 0) + h.T @ h
                for n in ("block01.q", "block04.q", "block08.q", "block04.down", "block08.down"):
                    xi = recorder.inputs[n][:, 1:].reshape(-1, recorder.inputs[n].shape[-1]).double()
                    sums[n] = sums.get(n, 0) + xi.T @ xi
        handle.remove()
        recorder.remove()
        results[item] = {k: summary(v) for k, v in sums.items()}
        print(arm.split("/")[-1][:30], step, {k: (round(v["top_over_mean"], 1), round(v["participation_ratio"], 3), round(v["top8_share"], 3))
                                             for k, v in results[item].items()}, flush=True)
    out.write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()
