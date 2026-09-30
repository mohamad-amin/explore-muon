"""Does spike suppression lose late because the implicit bias lags? (read-only probe)

In a bias-free network the component of W along the mean input x̂ = x̄/‖x̄‖ acts as a
bias: (W + δ x̂ᵀ) x shifts every token's output by about ‖x̄‖ δ. At a final checkpoint,
fit one output vector δ per body matrix along x̂ only (48 matrices, all weights frozen)
on fresh training tokens, and measure the drop in held-out loss. Control: the same fit
along a fixed random unit input direction z ⊥ x̂.

Prediction if C's late deficit (+0.0094 vs its twin M) is a lagging implicit bias: C's
bias-only gain exceeds M's by roughly that deficit, while the control gains match.

v2: v1 (probe.py) stepped the mean-direction delta about 17x harder in output units than
the control (the projection x.xhat is about ||xbar|| ~ 14, x.z about 0.8) and made M's
held-out loss worse. Here each direction d is normalized by r_d = RMS(x.d), so the fitted
vector is an output shift in the same units for both directions, with more fit data and
two learning rates.

One GPU, FP32. From the project root:
.venv/bin/python logs/muon_spectra/bias_lag_probe_20260925/probe_v2.py LABEL=RUN_DIR [LABEL=RUN_DIR ...]
"""
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from research.adamw_spectra.data import TokenStream, token_views
from research.adamw_spectra.model import GPT, ModelConfig
from research.adamw_spectra.spike_diagnostics import body_layers
from research.adamw_spectra.train import load_config

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
device = torch.device("cuda")
SEQ = 512
FIT_OFFSET = 2_400_000_000  # fresh training tokens (runs used the first 1.54e9; curvature probe 2.2e9)
FIT_SEQUENCES, BATCH, STEPS, LRS = 4096, 32, 400, (1e-4, 3e-4)
HELD_SEQUENCES, MICRO = 256, 16  # first validation sequences


def loss_on(model, x, y):
    total = 0.0
    for i in range(0, x.shape[0], MICRO):
        total += float(model(x[i:i + MICRO], y[i:i + MICRO]).double()) * x[i:i + MICRO].shape[0]
    return total / x.shape[0]


def probe(run, fit, held):
    saved = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=False)
    config = load_config(overrides=saved["config"])
    model = GPT(ModelConfig(**config["model"])).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    layers = body_layers(model)
    sums = {name: torch.zeros(layer.weight.shape[1], device=device, dtype=torch.float64)
            for name, layer in layers.items()}
    def accumulate(name):
        def hook(module, args):
            sums[name].add_(args[0][:, 1:].double().sum(dim=(0, 1)))  # returns None: input unchanged
        return hook
    hooks = [layer.register_forward_pre_hook(accumulate(name)) for name, layer in layers.items()]
    with torch.no_grad():
        for i in range(0, 256, MICRO):
            model(fit[0][i:i + MICRO], fit[1][i:i + MICRO])
    for h in hooks:
        h.remove()
    xbar = {n: (s / (256 * (SEQ - 1))).float() for n, s in sums.items()}
    squares = {n: torch.zeros(2, device=device, dtype=torch.float64) for n in layers}
    xhat = {n: v / v.norm() for n, v in xbar.items()}
    generator = torch.Generator(device=device).manual_seed(0)
    rand = {}
    for n, v in xhat.items():
        z = torch.randn(v.numel(), device=device, generator=generator)
        z -= (z @ v) * v
        rand[n] = z / z.norm()
    def project(name):
        def hook(module, args):
            sample = args[0][:, 1:].double()
            squares[name].add_(torch.stack([(sample @ xhat[name].double()).pow(2).sum(),
                                            (sample @ rand[name].double()).pow(2).sum()]))
        return hook
    hooks = [layer.register_forward_pre_hook(project(name)) for name, layer in layers.items()]
    with torch.no_grad():
        for i in range(0, 256, MICRO):
            model(fit[0][i:i + MICRO], fit[1][i:i + MICRO])
    for h in hooks:
        h.remove()
    rms = {n: (q / (256 * (SEQ - 1))).sqrt().float() for n, q in squares.items()}
    with torch.no_grad():
        base_held = loss_on(model, *held)
        base_fit = loss_on(model, fit[0][:256], fit[1][:256])
    out = {"run": str(run), "base_held": base_held, "base_fit_subset": base_fit, "fits": {},
           "projection_rms": {n: [float(r[0]), float(r[1])] for n, r in rms.items()}}
    for tag, directions, column, lr in [(t, d, c, lr) for lr in LRS
                                        for t, d, c in (("mean_direction", xhat, 0), ("random_direction", rand, 1))]:
        tag = f"{tag}_lr{lr:g}"
        scaled = {n: directions[n] / rms[n][column] for n in directions}
        deltas = {n: torch.zeros(layer.weight.shape[0], device=device, requires_grad=True)
                  for n, layer in layers.items()}
        hooks = [layer.register_forward_hook(
            lambda m, a, o, n=name: o + (a[0] @ scaled[n]).unsqueeze(-1) * deltas[n])
            for name, layer in layers.items()]
        optimizer = torch.optim.Adam(list(deltas.values()), lr=lr, betas=(0.9, 0.99))
        schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, 2 * (1 - s / STEPS)))
        start, first_grad, trace = time.time(), None, []
        for step in range(STEPS):
            i = (step * BATCH) % FIT_SEQUENCES
            loss = model(fit[0][i:i + BATCH], fit[1][i:i + BATCH])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if step == 0:
                first_grad = {n: float(d.grad.norm()) for n, d in deltas.items()}
            optimizer.step()
            schedule.step()
            if step % 40 == 0 or step == STEPS - 1:
                trace.append((step, float(loss.detach())))
        with torch.no_grad():
            after_held = loss_on(model, *held)
            after_fit = loss_on(model, fit[0][:256], fit[1][:256])
        for h in hooks:
            h.remove()
        shift = {n: float(deltas[n].detach().norm()) for n in deltas}  # RMS output change (normalized d)
        out["fits"][tag] = {"held_after": after_held, "held_gain": base_held - after_held,
                            "fit_subset_after": after_fit, "fit_subset_gain": base_fit - after_fit,
                            "fit_trace": trace, "grad_norm_at_zero": first_grad,
                            "output_shift_norm": shift, "seconds": time.time() - start}
        print(f"{run.parent.name} {tag}: held {base_held:.5f} -> {after_held:.5f} "
              f"(gain {base_held - after_held:+.5f}); fit subset gain {base_fit - after_fit:+.5f} "
              f"in {time.time() - start:.0f}s", flush=True)
    return out


def main():
    stream = TokenStream(str(REPO / "data/fineweb10B/fineweb_train_*.bin"))
    tokens = stream.device_tokens(FIT_OFFSET, FIT_SEQUENCES * SEQ + 1, device)
    fit = token_views(tokens, 0, FIT_SEQUENCES * SEQ, SEQ)
    val = TokenStream(str(REPO / "data/fineweb10B/fineweb_val_*.bin"))
    held = token_views(val.device_tokens(0, HELD_SEQUENCES * SEQ + 1, device), 0, HELD_SEQUENCES * SEQ, SEQ)
    results = {}
    for arg in sys.argv[1:]:
        label, run = arg.split("=", 1)
        results[label] = probe(REPO / run, fit, held)
        path = Path(__file__).with_name("results_v2.json")
        previous = json.loads(path.read_text()) if path.exists() else {}
        previous[label] = results[label]
        path.write_text(json.dumps(previous, indent=1) + "\n")


if __name__ == "__main__":
    main()
