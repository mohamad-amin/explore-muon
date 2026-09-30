"""Regime check on Candidate D (PROTOCOL.md, "Regime check for second-order-structure ideas", with the
review addendum). Measurements on the per-step model snapshots of one run:

  edge      Fixed-probe loss at every snapshot. Top Gauss-Newton (GN) eigenvalue of the 48 body matrices
            through training, with a disjoint-probe stability check. In two windows, with the top-16 GN
            basis recomputed every 5 steps: the step-to-step cosine of fixed-probe gradients inside and
            outside that subspace across two disjoint probes, the body-only step's loss change and its
            multiple c* = -g.d / d.G.d of the one-dimensional optimum, and the stiff energy of the steps
            relative to chance.
  momentum  Momentum rebuilt from replayed training gradients (checked against the optimizer's own
            Newton-Schulz step and the saved final momentum), the stale-free momentum (same batches at the
            current weights), and the split of its error into staleness and noise per direction class.
  gains     RMSNorm / QK-norm gain spread through training, and the gauge check: Muon's and PD's step with
            and without the ln1/ln2 gains folded into the q/k/v/up columns (an exact reparametrization).

Fresh probes are training-stream blocks beyond the run's 96M-target horizon; the development and sealed
test splits are never read. Run from tiny_surrogate/ through ./run.
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.func import functional_call, jvp, vjp
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.nn import functional as F

from research.tiny_spectra.model import GPT, ModelConfig
from research.tiny_spectra.fineweb import load_training_corpus
from research.adamw_spectra.muon import newton_schulz, NS_COEFFICIENTS

COHORT = Path("logs/tiny_spectra/regime_check_20260929")
TOP_K = 16


def cos(a, b):
    return float(F.cosine_similarity(a.flatten().double(), b.flatten().double(), dim=0))


# ----------------------------------------------------------------------------- run access

class Run:
    def __init__(self, run_dir, device, smoke=False):
        self.smoke = smoke
        self.dir = Path(run_dir)
        self.cfg = json.loads((self.dir / "config.json").read_text())
        self.metrics = {}
        for line in (self.dir / "metrics.jsonl").read_text().splitlines():
            record = json.loads(line)
            self.metrics[record["step"]] = record
        self.device = torch.device(device)
        self.corpus = load_training_corpus(self.cfg["data_path"], expected_manifest_sha256=self.cfg["data_sha256"])
        self.seq = self.cfg["seq_len"]
        self.total_sequences = self.cfg["total_tokens"] // self.seq
        self.batch_sequences = self.cfg["batch_tokens"] // self.seq
        self.total_steps = math.ceil(self.total_sequences / self.batch_sequences)
        self.starts = torch.load(self.dir / "windows.pt")["train"]
        # The same default stream, continued beyond the horizon, gives never-trained blocks.
        generator = torch.Generator().manual_seed(self.cfg["seed"] + 1729)
        stream = self.corpus.window_positions("train", self.total_sequences + 40960, self.seq, generator=generator)
        if not torch.equal(stream[:self.total_sequences], self.starts):
            raise RuntimeError("Stream prefix does not reproduce the run's training windows")
        f = stream[self.total_sequences:]
        if smoke:
            self.probe = dict(C1=f[0:2], C2=f[2:4], A=f[4:8], B=f[8:12])
            self.fresh_pool, self.band_probe = f[12:-4], f[-4:]
        else:
            self.probe = dict(C1=f[0:64], C2=f[64:128], A=f[128:384], B=f[384:640])
            self.fresh_pool, self.band_probe = f[640:-64], f[-64:]
        mc = ModelConfig(vocab_size=self.corpus.vocab_size, n_layer=self.cfg["n_layer"], n_embd=self.cfg["n_embd"],
                         n_head=self.cfg["n_head"], seq_len=self.seq)
        self.model = GPT(mc).to(self.device)
        self.body_names = list(self.model.body_parameters())
        self.params = dict(self.model.named_parameters())
        self.shapes = {n: self.params[n].shape for n in self.body_names}
        self.sizes = [self.shapes[n].numel() for n in self.body_names]
        self.dim = sum(self.sizes)
        self.beta = self.cfg["momentum"]
        self.clip = self.cfg["grad_clip"]
        self.decay = self.cfg.get("decay", 0.0)
        self._cache = {}

    def snapshot(self, step):
        if step not in self._cache:
            if len(self._cache) > 4:
                self._cache.pop(next(iter(self._cache)))
            self._cache[step] = torch.load(self.dir / "models" / f"step{step:06d}.pt", map_location="cpu")["model"]
        return self._cache[step]

    def load(self, step, body_from=None):
        state = dict(self.snapshot(step))
        if body_from is not None:
            other = self.snapshot(body_from)
            state.update({n: other[n] for n in self.body_names})
        self.model.load_state_dict(state, strict=True)

    def body_vector(self, step):
        state = self.snapshot(step)
        return torch.cat([state[n].reshape(-1).float() for n in self.body_names]).to(self.device)

    def batch_positions(self, step):
        begin = (step - 1) * self.batch_sequences
        positions = self.starts[begin:begin + self.batch_sequences]
        return positions[:4] if self.smoke else positions

    def is_partial(self, step):
        begin = (step - 1) * self.batch_sequences
        return len(self.starts[begin:begin + self.batch_sequences]) < self.batch_sequences

    def tokens(self, positions):
        return self.corpus.batch("train", len(positions), self.seq, positions=positions, device=self.device)

    def kappa(self, step):
        norm = self.metrics[step]["gradient_norm_before_clip"]
        return min(1.0, self.clip / (norm + 1e-6))

    def flat(self, tensors):
        return torch.cat([tensors[n].reshape(-1).float() for n in self.body_names])

    def unflat(self, vector):
        out, offset = {}, 0
        for n, size in zip(self.body_names, self.sizes):
            out[n] = vector[offset:offset + size].view(self.shapes[n])
            offset += size
        return out


# ----------------------------------------------------------------------------- gradients and GN products

def body_gradient(run, positions, *, micro=32, precision="bf16"):
    """Mean-loss gradient of the body matrices (flat), the loss, and the all-parameter gradient norm."""
    model = run.model
    for p in model.parameters():
        p.grad = None
    total = len(positions)
    loss_sum = 0.0
    for chunk in positions.split(micro):
        x, y = run.tokens(chunk)
        if precision == "bf16" and run.device.type == "cuda":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
        else:
            logits = model(x)
        loss = F.cross_entropy(logits.float().flatten(0, 1), y.flatten())
        (loss * (len(chunk) / total)).backward()
        loss_sum += float(loss.detach()) * len(chunk) / total
    all_norm = float(torch.sqrt(sum(p.grad.float().square().sum() for p in model.parameters() if p.grad is not None)))
    grad = torch.cat([run.params[n].grad.reshape(-1).float() for n in run.body_names])
    for p in model.parameters():
        p.grad = None
    return grad, loss_sum, all_norm


@torch.no_grad()
def probe_loss(run, positions, micro=64):
    total, count = 0.0, 0
    for chunk in positions.split(micro):
        x, y = run.tokens(chunk)
        logits = run.model(x)
        total += float(F.cross_entropy(logits.float().flatten(0, 1), y.flatten(), reduction="sum"))
        count += y.numel()
    return total / count


class GaussNewton:
    """Exact GN-vector products of the mean cross entropy on a fixed probe, over the body matrices (FP32)."""

    def __init__(self, run, positions, chunk=8):
        self.run = run
        params = {n: p.detach() for n, p in run.model.named_parameters()}
        self.body = {n: params[n] for n in run.body_names}
        self.others = {n: v for n, v in params.items() if n not in self.body}
        self.inputs = [run.tokens(c)[0] for c in positions.split(chunk)]
        self.count = sum(x.numel() for x in self.inputs)

    def __call__(self, vector):
        v = self.run.unflat(vector)
        out = torch.zeros_like(vector)
        with sdpa_kernel(SDPBackend.MATH):
            for x in self.inputs:
                f = lambda b: functional_call(self.run.model, {**self.others, **b}, (x,))
                logits, jv = jvp(f, (self.body,), (v,))
                p = torch.softmax(logits.float(), -1)
                hjv = p * jv - p * (p * jv).sum(-1, keepdim=True)
                _, pullback = vjp(f, self.body)
                (g,) = pullback(hjv)
                out += self.run.flat(g)
        return out / self.count


def lanczos(operator, dim, iterations, device, k=1, seed=0, residuals=False):
    """Top-k Ritz pairs of a symmetric PSD operator (full reorthogonalization) and optional relative residuals."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(dim, generator=generator).to(device)
    q /= q.norm()
    basis, alphas, betas = [q], [], []
    for j in range(iterations):
        w = operator(basis[-1])
        alpha = torch.dot(w, basis[-1])
        w = w - alpha * basis[-1] - (betas[-1] * basis[-2] if betas else 0)
        stack = torch.stack(basis)
        w = w - stack.T @ (stack @ w)
        w = w - stack.T @ (stack @ w)
        beta = w.norm()
        alphas.append(alpha)
        if j == iterations - 1 or beta < 1e-10:
            break
        betas.append(beta)
        basis.append(w / beta)
    m = len(alphas)
    t = torch.diag(torch.stack(alphas))
    if m > 1:
        off = torch.stack(betas[:m - 1])
        t = t + torch.diag(off, 1) + torch.diag(off, -1)
    values, vectors = torch.linalg.eigh(t.double())
    order = torch.argsort(values, descending=True)[:k]
    ritz = (torch.stack(basis[:m]).T @ vectors[:, order].float()).T.contiguous()
    ritz = ritz / ritz.norm(dim=1, keepdim=True)
    values = values[order].float().cpu()
    if not residuals:
        return values, ritz
    res = [float((operator(ritz[i]) - values[i] * ritz[i]).norm() / max(abs(float(values[i])), 1e-12))
           for i in range(min(k, 4))]
    return values, ritz, res


def overlap(u1, u2):
    """Mean squared cosine between two orthonormal k-subspaces (1 = identical, k/dim = chance)."""
    return float((u1 @ u2.T).square().sum() / u1.shape[0])


def _median(values):
    values = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(torch.tensor(values).median()) if values else None


def _mean(values):
    values = [v for v in values if v is not None]
    return float(sum(values) / len(values)) if values else None


# ----------------------------------------------------------------------------- edge

def edge(run, out):
    T = 4 if run.smoke else run.total_steps
    big = run.batch_sequences >= 2048
    result = dict(run=run.cfg["run_id"], total_steps=run.total_steps, chance_stiff_energy=TOP_K / run.dim,
                  sharpness=[], probe_stability=[], probe_loss=[], windows=[])
    write = lambda: (out / "edge.json").write_text(json.dumps(result, indent=1) + "\n")
    t0 = time.time()
    lanczos_its = 6 if run.smoke else 64
    k = 4 if run.smoke else TOP_K
    every = 4 if big else 16
    sharp_steps = [1, 2] if run.smoke else sorted(set([1, 2, 4] + list(range(every, T + 1, every)) + [T]))
    stability_steps = [1, 2] if run.smoke else ([20, 50, 80] if big else [60, 180, 300])
    loss_B = {}
    for step in range(0, T + 1):
        run.load(step)
        result["probe_loss"].append(dict(step=step, loss=probe_loss(run, run.probe["A"])))
        if step in sharp_steps:
            values, _ = lanczos(GaussNewton(run, run.probe["C1"]), run.dim, 4 if run.smoke else 20, run.device, k=1)
            lr_next = run.metrics.get(step + 1, run.metrics.get(step, {})).get("lr", float("nan"))
            result["sharpness"].append(dict(step=step, lambda_max=float(values[0]), lr_next=lr_next,
                                            lr_times_lambda=lr_next * float(values[0])))
        if step in stability_steps:
            v1, u1, r1 = lanczos(GaussNewton(run, run.probe["C1"]), run.dim, lanczos_its, run.device, k=k, seed=1,
                                 residuals=True)
            v2, u2, r2 = lanczos(GaussNewton(run, run.probe["C2"]), run.dim, lanczos_its, run.device, k=k, seed=3,
                                 residuals=True)
            result["probe_stability"].append(dict(step=step, eigenvalues_C1=v1.tolist(), eigenvalues_C2=v2.tolist(),
                                                  residuals_C1=r1, residuals_C2=r2, top_k_overlap=overlap(u1, u2),
                                                  top_4_overlap=overlap(u1[:4], u2[:4])))
            print(json.dumps(dict(stage="stability", step=step, lam=[round(v1[0].item(), 2), round(v2[0].item(), 2)],
                                  overlap=round(overlap(u1, u2), 3), res=[round(x, 3) for x in r1])), flush=True)
    losses = [r["loss"] for r in result["probe_loss"]]
    result["loss_increase_fraction_full_step"] = {}
    for name, (a, b) in dict(first_third=(1, T // 3), second_third=(T // 3, 2 * T // 3),
                             last_third=(2 * T // 3, T)).items():
        rises = sum(1 for s in range(a, b) if losses[s + 1] > losses[s])
        result["loss_increase_fraction_full_step"][name] = rises / max(1, b - a)
    write()
    print(json.dumps(dict(stage="sharpness_done", elapsed=round(time.time() - t0))), flush=True)

    windows = [(1, 3)] if run.smoke else ([(8, 28), (45, 75)] if big else [(20, 70), (180, 230)])
    for start, end in windows:
        record = dict(start=start, end=end, bases=[], pairs=[], steps=[])
        basis, previous = None, None
        for s in range(start, end + 1):
            run.load(s)
            if (s - start) % 5 == 0 and s < end:
                values, vectors, res = lanczos(GaussNewton(run, run.probe["C1"]), run.dim, lanczos_its, run.device,
                                               k=k, seed=1, residuals=True)
                record["bases"].append(dict(step=s, eigenvalues=values.tolist(), residuals=res,
                                            overlap_with_previous=None if basis is None else overlap(vectors, basis)))
                basis = vectors
            gA, lossA, _ = body_gradient(run, run.probe["A"], precision="fp32")
            gB, lossB, _ = body_gradient(run, run.probe["B"], precision="fp32")
            loss_B[s] = lossB
            if previous is not None:
                U = previous["basis"]
                a_prev, b_now, a_now = U @ previous["gA"], U @ gB, U @ gA
                record["pairs"].append(dict(
                    step=s - 1,
                    stiff_cos_cross=cos(a_prev, b_now),
                    rest_cos_cross=cos(previous["gA"] - U.T @ a_prev, gB - U.T @ b_now),
                    stiff_cos_same=cos(a_prev, a_now),
                    rest_cos_same=cos(previous["gA"] - U.T @ a_prev, gA - U.T @ a_now),
                    gradient_stiff_share=float(a_prev.square().sum() / previous["gA"].square().sum())))
            if s < end:
                d = run.body_vector(s + 1) - run.body_vector(s)
                entry = dict(step=s + 1, lr=run.metrics[s + 1]["lr"], kappa=run.kappa(s + 1),
                             stiff_energy=float((basis @ d).square().sum() / d.square().sum()))
                entry["stiff_energy_over_chance"] = entry["stiff_energy"] / result["chance_stiff_energy"]
                for name, g in (("A", gA), ("B", gB)):
                    curvature = float(torch.dot(d, GaussNewton(run, run.probe[name])(d)))
                    slope = float(torch.dot(g, d))
                    entry[f"c_star_{name}"] = -slope / curvature if curvature > 0 else float("nan")
                    entry[f"model_change_{name}"] = slope + 0.5 * curvature
                run.load(s, body_from=s + 1)
                entry["body_step_loss_change_A"] = probe_loss(run, run.probe["A"]) - lossA
                entry["body_step_loss_change_B"] = probe_loss(run, run.probe["B"]) - lossB
                record["steps"].append(entry)
            previous = dict(gA=gA, basis=basis)
        steps = record["steps"]
        record["summary"] = dict(
            c_star_A_median=_median([x["c_star_A"] for x in steps]),
            c_star_B_median=_median([x["c_star_B"] for x in steps]),
            stiff_cos_cross_median=_median([x["stiff_cos_cross"] for x in record["pairs"]]),
            rest_cos_cross_median=_median([x["rest_cos_cross"] for x in record["pairs"]]),
            stiff_cos_same_median=_median([x["stiff_cos_same"] for x in record["pairs"]]),
            rest_cos_same_median=_median([x["rest_cos_same"] for x in record["pairs"]]),
            gradient_stiff_share_median=_median([x["gradient_stiff_share"] for x in record["pairs"]]),
            body_loss_increase_fraction=_mean([(x["body_step_loss_change_A"] > 0) + 0.0 for x in steps] +
                                              [(x["body_step_loss_change_B"] > 0) + 0.0 for x in steps]),
            stiff_energy_median=_median([x["stiff_energy"] for x in steps]),
            stiff_energy_over_chance_median=_median([x["stiff_energy_over_chance"] for x in steps]),
            basis_overlap_median=_median([b["overlap_with_previous"] for b in record["bases"][1:]]),
            kappa_mean=_mean([x["kappa"] for x in steps]),
            clipped_fraction=_mean([(x["kappa"] < 1) + 0.0 for x in steps]))
        result["windows"].append(record)
        write()
        print(json.dumps(dict(stage="window", start=start, end=end,
                              **{k2: (round(v, 4) if isinstance(v, float) else v) for k2, v in record["summary"].items()},
                              elapsed=round(time.time() - t0))), flush=True)


# ----------------------------------------------------------------------------- momentum

def input_statistics(run, positions):
    """Per body matrix: input second moment at the current weights, and the eigenvectors above its mean."""
    captured, hooks = {}, []
    for name, module in run.model.body_modules().items():
        def hook(mod, args, key=name):
            x = args[0].detach().float().reshape(-1, args[0].shape[-1])
            captured.setdefault(key, []).append(x.T @ x / x.shape[0])
        hooks.append(module.register_forward_pre_hook(hook))
    with torch.no_grad():
        for chunk in positions.split(16):
            run.model(run.tokens(chunk)[0])
    for h in hooks:
        h.remove()
    bands, covariances = {}, {}
    for name, items in captured.items():
        c = torch.stack(items).mean(0)
        covariances[name] = c
        values, vectors = torch.linalg.eigh(c.double())
        bands[name] = vectors[:, values > values.mean()].float()
    return bands, covariances


def class_inner(run, a, b, top, bands):
    """Inner products of two flat body vectors restricted to direction classes."""
    total = float(torch.dot(a.double(), b.double()))
    gn_top = float(torch.dot((top @ a).double(), (top @ b).double()))
    A, B = run.unflat(a), run.unflat(b)
    band = sum(float(((A[n] @ bands[n]) * (B[n] @ bands[n])).double().sum()) for n in run.body_names)
    return dict(all=total, gn_top=gn_top, gn_rest=total - gn_top, input_band=band, input_rest=total - band)


def momentum(run, out, states, full_history=False, micro=32, tag=""):
    fresh_batches = 2 if run.smoke else (16 if run.batch_sequences >= 2048 else 32)
    size = 4 if run.smoke else run.batch_sequences
    beta = run.beta
    existing = out / f"momentum{tag}.json"
    results = json.loads(existing.read_text()) if existing.exists() else []
    classes = ("all", "gn_top", "gn_rest", "input_band", "input_rest")
    for t in states:
        t0 = time.time()
        K = t if full_history else min(40, t)
        window = list(range(t - K + 1, t + 1))
        weights = {s: beta ** (t - s) * run.kappa(s) for s in window}
        S = sum(weights.values())
        M, norms = None, []
        for s in window:
            run.load(s - 1)
            g, _, all_norm = body_gradient(run, run.batch_positions(s), micro=micro)
            M = g * weights[s] if M is None else M + g * weights[s]
            norms.append(dict(step=s, replayed=all_norm, logged=run.metrics[s]["gradient_norm_before_clip"]))
        run.load(t)
        current = torch.stack([body_gradient(run, run.batch_positions(s), micro=micro)[0] for s in window])
        w = torch.tensor([weights[s] for s in window], device=run.device)
        M_tilde = w @ current
        full = [i for i, s in enumerate(window) if not run.is_partial(s)]
        g_past = current[full].mean(0)
        fresh = torch.stack([body_gradient(run, run.fresh_pool[i * size:(i + 1) * size], micro=micro)[0]
                             for i in range(fresh_batches)])
        g_fresh = fresh.mean(0)
        top_values, top, res = lanczos(GaussNewton(run, run.probe["C1"]), run.dim, 6 if run.smoke else 64,
                                       run.device, k=4 if run.smoke else TOP_K, seed=2, residuals=True)
        bands, _ = input_statistics(run, run.band_probe)
        inner = lambda a, b: class_inner(run, a, b, top, bands)
        replay_spread = [inner(current[i] - g_past, current[i] - g_past) for i in full]
        fresh_spread = [inner(fresh[i] - g_fresh, fresh[i] - g_fresh) for i in range(fresh_batches)]
        noise_weight = float((w ** 2).sum())
        sq = {name: inner(v, v) for name, v in dict(fresh=g_fresh, stale=M - M_tilde, momentum=M,
                                                     stale_free=M_tilde, replay_gap=g_past - g_fresh).items()}
        ip = {name: inner(a, g_fresh) for name, a in dict(momentum=M, stale_free=M_tilde).items()}
        ip_ms = inner(M, M_tilde)
        record = dict(step=t, K=K, S=S, micro=micro, lr=run.metrics[t]["lr"], top_eigenvalues=top_values.tolist(),
                      lanczos_residuals=res, kappa_mean=_mean([run.kappa(s) for s in window]),
                      clipped_fraction=_mean([(run.kappa(s) < 1) + 0.0 for s in window]),
                      norm_check_max_relative=max(abs(n["replayed"] - n["logged"]) / n["logged"] for n in norms),
                      classes={})
        norm = lambda x: math.sqrt(max(x, 0.0))
        for cls in classes:
            v_replay = sum(x[cls] for x in replay_spread) / max(1, len(full) - 1)
            v_fresh = sum(x[cls] for x in fresh_spread) / (fresh_batches - 1)
            record["classes"][cls] = dict(
                signal=S * norm(sq["fresh"][cls] - v_fresh / fresh_batches),
                staleness=norm(sq["stale"][cls]),
                noise=norm(noise_weight * v_fresh), noise_from_replay=norm(noise_weight * v_replay),
                momentum=norm(sq["momentum"][cls]), stale_free=norm(sq["stale_free"][cls]),
                cos_momentum_fresh=ip["momentum"][cls] / max(1e-30, norm(sq["momentum"][cls]) * norm(sq["fresh"][cls])),
                cos_stale_free_fresh=ip["stale_free"][cls] / max(1e-30, norm(sq["stale_free"][cls]) * norm(sq["fresh"][cls])),
                cos_momentum_stale_free=ip_ms[cls] / max(1e-30, norm(sq["momentum"][cls]) * norm(sq["stale_free"][cls])),
                replay_gap_over_fresh=norm(sq["replay_gap"][cls]) / max(1e-30, norm(sq["fresh"][cls])),
                replay_gap_expected=norm(v_replay / max(1, len(full)) + v_fresh / fresh_batches)
                / max(1e-30, norm(sq["fresh"][cls])),
                replay_over_fresh_spread=v_replay / max(1e-30, v_fresh))
        record["momentum_stiff_share"] = sq["momentum"]["gn_top"] / sq["momentum"]["all"]
        record["momentum_stiff_share_over_chance"] = record["momentum_stiff_share"] / (top.shape[0] / run.dim)
        if run.cfg["method"] == "muon":
            # The optimizer's own map on the rebuilt momentum should reproduce the step it took at step t.
            lr = run.metrics[t]["lr"]
            before, after = run.unflat(run.body_vector(t - 1)), run.unflat(run.body_vector(t))
            m = run.unflat(M)
            values = []
            for n in run.body_names:
                actual = ((1 - lr * run.decay) * before[n] - after[n]) / lr
                values.append(cos(actual, newton_schulz(m[n][None], NS_COEFFICIENTS)[0]))
            record["ns_step_check"] = dict(median_cos=_median(values), min_cos=min(values))
        if t == run.total_steps and (run.dir / "final.pt").exists():
            payload = torch.load(run.dir / "final.pt", map_location="cpu", weights_only=False)
            saved = run.flat({n: payload["optimizer"]["state"][n]["momentum_buffer"].to(run.device)
                              for n in run.body_names})
            record["final_momentum_check"] = dict(relative_error=float((M - saved).norm() / saved.norm()),
                                                  cos=cos(M, saved), expected_truncation=beta ** K)
        torch.save(dict(step=t, momentum=M.cpu(), stale_free=M_tilde.cpu(), fresh_mean=g_fresh.cpu()),
                   out / f"momentum_state{tag}_{t:04d}.pt")
        record["seconds"] = time.time() - t0
        results = sorted([r for r in results if r["step"] != t] + [record], key=lambda r: r["step"])
        existing.write_text(json.dumps(results, indent=1) + "\n")
        brief = {c: {key: round(record["classes"][c][key], 3) for key in
                     ("signal", "staleness", "noise", "cos_momentum_fresh", "cos_stale_free_fresh")}
                 for c in ("all", "gn_top", "gn_rest")}
        print(json.dumps(dict(stage="momentum", step=t, **brief, ns=record.get("ns_step_check"),
                              final=record.get("final_momentum_check"),
                              norm_check=round(record["norm_check_max_relative"], 4),
                              seconds=round(record["seconds"]))), flush=True)


def ns_diagnostic(run, out, states, micro):
    """Muon only: rebuild the momentum with a given replay microbatch and repeat the Newton-Schulz step check."""
    existing = out / "ns_diagnostic.json"
    results = json.loads(existing.read_text()) if existing.exists() else []
    for t in states:
        K = min(40, t)
        M = None
        for s in range(t - K + 1, t + 1):
            run.load(s - 1)
            g, _, _ = body_gradient(run, run.batch_positions(s), micro=micro)
            weight = run.beta ** (t - s) * run.kappa(s)
            M = g * weight if M is None else M + g * weight
        lr = run.metrics[t]["lr"]
        before, after = run.unflat(run.body_vector(t - 1)), run.unflat(run.body_vector(t))
        m = run.unflat(M)
        values = [cos(((1 - lr * run.decay) * before[n] - after[n]) / lr, newton_schulz(m[n][None], NS_COEFFICIENTS)[0])
                  for n in run.body_names]
        saved = out / f"momentum_state_{t:04d}.pt"
        record = dict(step=t, K=K, micro=micro, median_cos=_median(values), min_cos=min(values),
                      cos_with_micro32_rebuild=cos(M, torch.load(saved)["momentum"].to(run.device)) if saved.exists() else None)
        results = [r for r in results if not (r["step"] == t and r["micro"] == micro)] + [record]
        existing.write_text(json.dumps(results, indent=1) + "\n")
        print(json.dumps(dict(stage="ns_diagnostic", **record)), flush=True)


# ----------------------------------------------------------------------------- gains

def polar(matrix):
    u, _, vh = torch.linalg.svd(matrix.double(), full_matrices=False)
    return (u @ vh).float()


def ns(matrix):
    return newton_schulz(matrix[None], NS_COEFFICIENTS)[0]


def pd_root(c, alpha, damping=1e-3):
    values, vectors = torch.linalg.eigh(c.double())
    values = values.clamp_min(0) / values.clamp_min(0).mean()
    return ((vectors * (values + damping).pow(-alpha)) @ vectors.T).float()


def data_cos(a, b, c):
    """Cosine of two steps under the data metric tr(A C B^T), i.e. of their output changes on the inputs."""
    a, b, c = a.double(), b.double(), c.double()
    return float(torch.trace(a @ c @ b.T) / torch.sqrt(torch.trace(a @ c @ a.T) * torch.trace(b @ c @ b.T)))


def gains(run, out, states):
    T = run.total_steps
    names = [n for n in run.params
             if n.endswith(("ln1.weight", "ln2.weight", "q_norm.weight", "k_norm.weight")) or n == "norm.weight"]
    trajectory = []
    for step in (range(0, 4) if run.smoke else range(0, T + 1, 1 if T <= 100 else 4)):
        state = run.snapshot(step)
        row = dict(step=step)
        for n in names:
            g = state[n].float()
            median = g.abs().median()
            row[n] = dict(cv=float(g.std() / g.mean().abs()), max_over_median=float(g.abs().max() / median),
                          min_over_median=float(g.abs().min() / median), mean=float(g.mean()))
        trajectory.append(row)
    fold = []
    for t in states:
        path = out / f"momentum_state_{t:04d}.pt"
        if not path.exists():
            continue
        M = run.unflat(torch.load(path)["momentum"].to(run.device))
        run.load(t)
        state = run.snapshot(t)
        _, covariances = input_statistics(run, run.band_probe)
        rows = []
        for i in range(run.cfg["n_layer"]):
            for kind, gain_name in (("q", f"blocks.{i}.ln1.weight"), ("k", f"blocks.{i}.ln1.weight"),
                                    ("v", f"blocks.{i}.ln1.weight"), ("up", f"blocks.{i}.ln2.weight")):
                name = f"blocks.{i}.{'mlp' if kind == 'up' else 'attn'}.{kind}.weight"
                m = M[name].float()
                g = state[gain_name].float().to(run.device)
                c = covariances[name].to(run.device)          # second moment of x = g * z (what PD sees)
                cz = c / g[:, None] / g[None, :]              # second moment of the normalized input z
                row = dict(layer=i, kind=kind, gain_max_over_median=float(g.abs().max() / g.abs().median()))
                for label, fmap in (("muon_ns", ns), ("muon_svd", polar)):
                    orig, folded = fmap(m) * g, fmap(m / g)
                    row[f"{label}_cos"] = cos(orig, folded)
                    row[f"{label}_data_cos"] = data_cos(orig, folded, cz)
                for alpha, damping, fmap, label in ((0.25, 1e-3, ns, "pd025"), (0.5, 1e-3, ns, "pd05"),
                                                    (0.5, 1e-8, polar, "pd05_algebra")):
                    r, r_fold = pd_root(c, alpha, damping), pd_root(cz, alpha, damping)
                    orig = (fmap(m @ r) @ r) * g
                    folded = fmap((m / g) @ r_fold) @ r_fold
                    row[f"{label}_cos"] = cos(orig, folded)
                    row[f"{label}_data_cos"] = data_cos(orig, folded, cz)
                rows.append(row)
        fold.append(dict(step=t, rows=rows))
        print(json.dumps(dict(stage="fold", step=t, **{key: round(_median([r[key] for r in rows]), 4) for key in
                                                       ("muon_ns_data_cos", "muon_svd_data_cos", "pd025_data_cos",
                                                        "pd05_data_cos", "pd05_algebra_data_cos")})), flush=True)
    (out / "gains.json").write_text(json.dumps(dict(trajectory=trajectory, fold=fold), indent=1) + "\n")


# ----------------------------------------------------------------------------- main

def default_states(run):
    if run.smoke:
        return [2, 3]
    if run.batch_sequences >= 2048:  # 1M: 92 steps, cooldown from step 83
        return [20, 46, 65, 83, run.total_steps]
    return [40, 120, 183, 280, run.total_steps]  # 262K: 368 steps, cooldown from step 332


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("edge", "momentum", "gains", "all", "ns_diagnostic"))
    parser.add_argument("run_id")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true", help="tiny sizes for a CPU code check; results are not science")
    parser.add_argument("--states", type=int, nargs="*")
    parser.add_argument("--micro", type=int, default=None, help="replay microbatch (momentum default 32, ns_diagnostic 4)")
    parser.add_argument("--full-history", action="store_true", help="momentum stage: rebuild from step 1 (no truncation)")
    parser.add_argument("--tag", default="", help="momentum stage: output suffix, e.g. _exact")
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    run = Run(COHORT / "runs" / args.run_id, args.device, smoke=args.smoke)
    out = COHORT / ("smoke" if args.smoke else "analysis") / args.run_id
    out.mkdir(parents=True, exist_ok=True)
    states = args.states or default_states(run)
    if args.stage in ("edge", "all"):
        edge(run, out)
    if args.stage in ("momentum", "all"):
        momentum(run, out, states, full_history=args.full_history, micro=args.micro or 32, tag=args.tag)
    if args.stage in ("gains", "all"):
        gains(run, out, states)
    if args.stage == "ns_diagnostic":
        ns_diagnostic(run, out, states, args.micro or 4)


if __name__ == "__main__":
    main()
