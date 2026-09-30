"""Loss stored in the stiff-mode oscillation: held-out loss at W_t, W_{t+1} and points on the segment between them.

With a period-2 limit cycle in the stiff modes, the midpoint (W_t + W_{t+1}) / 2 sits near the valley floor of the
oscillating modes; its loss drop against the endpoints is the loss the oscillation stores (what an LR cooldown or
weight averaging releases). All parameters are interpolated (hidden matrices, embeddings, gains, head).

With --split, the midpoint is also taken for the hidden matrices only (everything else at W_t) and for everything but
the hidden matrices only (embeddings, gains, head), to separate the two parts' stored loss (review, 11:24 CDT).

usage: midpoint_loss.py OUT_JSON ARM_DIR:STEP [...] [--sequences 1024] [--split]
"""
import argparse
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
from one_step_gn import EVAL  # noqa: E402

POINTS = (-0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5)


@torch.no_grad()
def mean_loss(model, batches):
    total, count = 0.0, 0
    for x, y in batches:
        total += float(P.token_losses(model(x), y).mean(1).sum())
        count += x.shape[0]
    return total / count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("items", nargs="+")
    parser.add_argument("--sequences", type=int, default=1024)
    parser.add_argument("--split", action="store_true")
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda")
    stream = TokenStream(str(REPO / "data/fineweb10B/fineweb_val_*.bin"))
    results = json.loads(args.out.read_text()) if args.out.exists() else {}
    for item in args.items:
        arm, step = item.rsplit(":", 1)
        step = int(step)
        kept = Path(arm) / "scientific" / "kept"
        model, saved = P.load_checkpoint(kept / f"step{step:06d}.pt", device)
        config = saved["config"]
        del saved
        nxt, _ = P.load_checkpoint(kept / f"step{step + 1:06d}_weights.pt", device, config["model"])
        T = model.config.seq_len
        batches = [stream.batch(EVAL + first * T, 16, T, device) for first in range(0, args.sequences, 16)]
        start = {k: v.detach().clone() for k, v in model.state_dict().items()}
        end = nxt.state_dict()
        losses = {}
        for c in POINTS:
            model.load_state_dict({k: (start[k] + c * (end[k] - start[k])) if start[k].is_floating_point() else start[k]
                                   for k in start})
            losses[str(c)] = mean_loss(model, batches)
        if args.split:
            hidden = {k for k in start if k.startswith("blocks.") and start[k].ndim == 2}     # the 48 body matrices
            for part, keys in (("hidden", hidden), ("aux", {k for k in start if k not in hidden})):
                for c in (0.5, 1.0):
                    model.load_state_dict({k: (start[k] + c * (end[k] - start[k])) if (k in keys and start[k].is_floating_point())
                                           else start[k] for k in start})
                    losses[f"{part}_{c}"] = mean_loss(model, batches)
            model.load_state_dict(start)
        entry = {"losses": losses, "midpoint_drop": 0.5 * (losses["0.0"] + losses["1.0"]) - losses["0.5"],
                 "step_change": losses["1.0"] - losses["0.0"], "beta": config["muon_momentum"],
                 "lr_config": config["learning_rate"], "batch_tokens": config["batch_tokens"]}
        results[f"{Path(arm).name}:{step}"] = entry
        print(json.dumps({"item": f"{Path(arm).name}:{step}", **{k: round(v, 5) for k, v in losses.items()},
                          "midpoint_drop": round(entry["midpoint_drop"], 5), "step_change": round(entry["step_change"], 5)}),
              flush=True)
        args.out.write_text(json.dumps(results, indent=1) + "\n")
        del model, nxt


if __name__ == "__main__":
    main()
