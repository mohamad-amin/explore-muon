"""Edge-of-stability check in each optimizer's own coordinates.

PD with input power alpha is Muon in the coordinates W~ = W R^-1, R = (C / mean eig + 1e-3 I)^-alpha: its update is
dW~ = s polar(grad W~) with s the norm-matching factor (s = 1 / pre_rescale_norm_ratio from the run's log). If the
state self-organizes to the stability edge of its own optimizer, LR * s * lambda_max(R G R) should take the same value
for Muon (alpha = 0, s = 1) and PD states. lambda_max by Lanczos on the exact GN (all hidden matrices) in the rescaled
coordinates, from a random start. Prints and writes JSON.

usage: eos_invariant.py OUT_JSON ARM_DIR:STEP:ALPHA[:BETA] [...] [--steps 30]
(BETA > 0: the two-sided rule's coordinates, W = L W~ R with L = (B / mean + 1e-3 I)^-beta, B from sampled labels)
"""
import argparse
import json
import math
import sys
from collections import OrderedDict
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
from research.adamw_spectra import gn_probe as P  # noqa: E402
from research.adamw_spectra.data import TokenStream  # noqa: E402
from research.adamw_spectra.train import learning_rate, token_budget  # noqa: E402
from one_step_gn import CURV, Flat, GaussNewton, inverse_roots, lanczos  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("items", nargs="+")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--curvature-sequences", type=int, default=256)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda")
    stream = TokenStream(str(REPO / "data/fineweb10B/fineweb_val_*.bin"))
    results = {}
    for item in args.items:
        parts = item.split(":")
        beta = float(parts.pop()) if len(parts) == 4 else 0.0
        arm, step, alpha = parts
        step, alpha = int(step), float(alpha)
        kept = Path(arm) / "scientific" / "kept"
        model, saved = P.load_checkpoint(kept / f"step{step:06d}.pt", device)
        config = saved["config"]
        T = model.config.seq_len
        names = list(P.hidden_linears(model))
        layers = P.hidden_linears(model)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for n in names:
            layers[n].weight.requires_grad_(True)
        curv = [stream.batch(CURV + first * T, 8, T, device) for first in range(0, args.curvature_sequences, 8)]
        roots, lefts = {}, {}
        if alpha > 0:
            recorder = P.Recorder(model)
            sums, count = {n: 0 for n in names}, 0
            with torch.no_grad():
                for x, _ in curv:
                    model(x)
                    for n in names:
                        inp = recorder.inputs[n][:, 1:].reshape(-1, recorder.inputs[n].shape[-1]).double()
                        sums[n] = sums[n] + inp.T @ inp
                    count += x.shape[0] * (T - 1)
            recorder.remove()
            roots = {n: inverse_roots(sums[n] / count, (alpha,))[alpha] for n in names}
        if beta > 0:
            # the two-sided rule's output factor: B = E[e~ e~^T] with model-sampled labels (e~ = T e)
            recorder = P.Recorder(model)
            gen = torch.Generator(device=device).manual_seed(0)
            sums, count = {n: 0 for n in names}, 0
            for x, y in curv:
                P.gradient_passes(model, recorder, x, y, gen, draws=1)
                for n in names:
                    e = (T * recorder.errors[n][1]).reshape(-1, recorder.errors[n][1].shape[-1]).double()
                    sums[n] = sums[n] + e.T @ e
                count += x.numel()
            recorder.remove()
            lefts = {n: inverse_roots(sums[n] / count, (beta,))[beta] for n in names}
        flat = Flat(OrderedDict((n, layers[n].weight) for n in names))
        gn = GaussNewton(model, names, curv, flat)

        def rescale(v):
            if not roots and not lefts:
                return v
            out = []
            for n, x in flat.dict(v).items():
                x = x @ roots[n] if roots else x
                x = lefts[n] @ x if lefts else x
                out.append(x.reshape(-1))
            return torch.cat(out)

        gen = torch.Generator(device=device).manual_seed(0)
        start = torch.randn(sum(flat.sizes), device=device, generator=gen)
        V, Tk = lanczos(lambda v: rescale(gn(rescale(v))), start, args.steps)
        ritz = torch.linalg.eigvalsh(Tk).flip(0)
        del V
        record = json.loads((kept / f"step{step:06d}.json").read_text()) if (kept / f"step{step:06d}.json").exists() else {}
        steps_log = Path(arm) / "scientific" / "steps" / f"step{step:06d}.json"
        log = json.loads(steps_log.read_text()) if steps_log.exists() else {}
        ratio = (log.get("data_norm") or {}).get("pre_rescale_norm_ratio")
        s = 1.0 / ratio if (alpha > 0 and ratio) else 1.0
        count_params = sum(p.numel() for p in model.parameters())
        lr = learning_rate(config, step + 1, step * config["batch_tokens"], token_budget(config, count_params, T))
        entry = {"alpha": alpha, "beta": beta, "lr": lr, "s": s, "top_ritz": ritz[:8].tolist(), "invariant": lr * s * float(ritz[0]),
                 "batch_tokens": config["batch_tokens"], "gradient_norm_logged": log.get("gradient_norm_before_clip")}
        results[f"{Path(arm).name}:{step}"] = entry
        print(json.dumps({"item": f"{Path(arm).name}:{step}", **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in entry.items() if k != 'top_ritz'},
                          "top_ritz": [round(x, 2) for x in entry["top_ritz"][:4]]}), flush=True)
        args.out.write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()
