"""Short LR anneal from a kept checkpoint: a cooldown in miniature that relaxes the stiff-mode oscillation.

The run's own optimizer and state (momentum, AdamW moments; PD's input statistics from the kept stats file) continue
on the run's own next training batches (same data order, BF16 autocast, gradient clipping) for K steps, with the LR
taken linearly from its scheduled value toward zero: lr_k = lr_sched(t + k) (1 - (k - 1) / K). The end state
approximates the valley floor under the oscillation, for re-scoring one-step directions there (review, 11:24 CDT).
Writes BRANCH_DIR/scientific/kept/step{t+K}.pt (model, optimizer, config, step) and BRANCH_DIR/anneal.json with the
held-out loss (EVAL sequences, the one-step scripts' scoring set) before the branch and after every step.
Single process, world size 1; the frozen training step is replicated (DDP mean over the global batch, clip, step).

The Muon momentum follows the run's own schedule (train.muon_momentum: constant, or the momentum warmup), as the trainer
sets it before every update (added 2026-09-29 07:25 CDT for the floor ledger; constant-beta runs are unchanged).

usage: anneal_branch.py BRANCH_DIR ARM_DIR:STEP [--steps 16] [--eval-sequences 256]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
from research.adamw_spectra import gn_probe as P  # noqa: E402
from research.adamw_spectra.data import TokenStream, token_views  # noqa: E402
from research.adamw_spectra.model import GPT, ModelConfig, StatLinear  # noqa: E402
from research.adamw_spectra.train import learning_rate, make_optimizer, muon_momentum, token_budget  # noqa: E402
from one_step_gn import EVAL  # noqa: E402


@torch.no_grad()
def held_out(model, batches):
    model.eval()
    total, count = 0.0, 0
    for x, y in batches:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            total += float(model(x, y)) * x.shape[0]
        count += x.shape[0]
    model.train()
    return total / count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("branch", type=Path)
    parser.add_argument("item")
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--eval-sequences", type=int, default=256)
    parser.add_argument("--no-save", action="store_true", help="skip the end-state checkpoint (disk space; floor ledger)")
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    started = time.time()
    arm, step = args.item.rsplit(":", 1)
    step = int(step)
    kept = Path(arm) / "scientific" / "kept"
    saved = torch.load(kept / f"step{step:06d}.pt", map_location="cpu", weights_only=False)
    config = saved["config"]
    model = GPT(ModelConfig(**config["model"])).to(device)
    model.load_state_dict(saved["model"])
    stats_path = kept / f"step{step:06d}_input_stats_rank0.pt"
    if stats_path.exists():     # PD's data norm reads these non-persistent StatLinear buffers
        stats = torch.load(stats_path, map_location="cpu", weights_only=False)
        modules = dict(model.named_modules())
        for name, buffers in stats.items():
            module = modules[name]
            if not isinstance(module, StatLinear):
                raise RuntimeError(f"{name} is not a StatLinear")
            for key, value in buffers.items():
                getattr(module, key).copy_(value.to(device))
    elif config.get("data_norm_alpha", 0) and config["optimizer"] == "muon":
        raise RuntimeError("PD-family run without kept input statistics")
    optimizer, _ = make_optimizer(model, config, device)
    optimizer.load_state_dict(saved["optimizer"])
    for attribute, value in (("capture_parameters", set()), ("audit_distribution", False)):
        if hasattr(optimizer, attribute):
            setattr(optimizer, attribute, value)
    del saved
    model.train()
    T = model.config.seq_len
    B = config["batch_tokens"]
    budget = token_budget(config, sum(p.numel() for p in model.parameters()), T)
    train = TokenStream(str(REPO / config["train_pattern"]))
    stream = TokenStream(str(REPO / "data/fineweb10B/fineweb_val_*.bin"))
    evals = [stream.batch(EVAL + first * T, 16, T, device) for first in range(0, args.eval_sequences, 16)]
    record = {"arm": arm, "start_step": step, "steps": args.steps, "batch_tokens": B,
              "loss": [held_out(model, evals)], "lr": [], "gradient_norm": [], "momentum": []}
    micro = config["microbatch_sequences"] * T
    for k in range(1, args.steps + 1):
        s = step + k
        lr = learning_rate(config, s, (s - 1) * B, budget) * (1 - (k - 1) / args.steps)
        for group in optimizer.param_groups:
            group["lr"] = lr * group.get("lr_scale", 1.0)
        if hasattr(optimizer, "momentum") and config["optimizer"] == "muon":
            optimizer.momentum = muon_momentum(config, (s - 1) * B, budget)
            record["momentum"].append(optimizer.momentum)
        optimizer.zero_grad(set_to_none=True)
        tokens = torch.from_numpy(train.read((s - 1) * B, B + 1)).to(device)
        for offset in range(0, B, micro):
            x, y = token_views(tokens, offset, micro, T)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(x, y)
            (loss * (micro / B)).backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"], foreach=True)
        optimizer.step()
        record["lr"].append(lr)
        record["gradient_norm"].append(float(norm))
        record["loss"].append(held_out(model, evals))
        print(json.dumps({"k": k, "step": s, "lr": round(lr, 6), "gradient_norm": round(float(norm), 4),
                          "held_out": round(record["loss"][-1], 5), "seconds": round(time.time() - started)}), flush=True)
    out = args.branch / "scientific" / "kept"
    out.mkdir(parents=True, exist_ok=True)
    final = step + args.steps
    if not args.no_save:
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": config, "step": final,
                    "branch_of": f"{arm}:{step}"}, out / f"step{final:06d}.pt")
    record["seconds"] = time.time() - started
    (args.branch / "anneal.json").write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({"done": True, "saved": str(out / f"step{final:06d}.pt"), "drop": record["loss"][0] - record["loss"][-1]}),
          flush=True)


if __name__ == "__main__":
    main()
