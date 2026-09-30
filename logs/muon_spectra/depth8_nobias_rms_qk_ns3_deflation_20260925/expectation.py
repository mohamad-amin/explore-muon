"""Read-only expectation check on real momentum: does the gate fire, and what does 3-step NS keep?

Uses the frontier-norm Muon run's final momentum (step 1469). First-order descent of an
NS direction D along M, relative to exact polar, is <M, D> / ||M||_*.
Run from the project root with one GPU visible.
"""
import json, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from research.adamw_spectra.model import GPT, ModelConfig
from research.adamw_spectra.muon import JORDAN_QUINTIC, NS_COEFFICIENTS, deflated_newton_schulz, newton_schulz
from research.adamw_spectra.spike_diagnostics import body_layers
from research.adamw_spectra.train import load_config, make_optimizer

run = Path("logs/muon_spectra/depth8_w512_nobias_rms_qk_20260925/muon/scientific")
saved = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=False)
config = load_config(overrides=saved["config"])
device = torch.device("cuda")
model = GPT(ModelConfig(**config["model"])).to(device)
model.load_state_dict(saved["model"])
optimizer, _ = make_optimizer(model, config, device)
optimizer.load_state_dict(saved["optimizer"])
three = (JORDAN_QUINTIC,) * 3
rows = {}
for name, layer in body_layers(model).items():
    m = optimizer.state[layer.weight]["momentum_buffer"].float()[None]
    nuclear = float(torch.linalg.svdvals(m.double()).sum())
    keep = lambda d: float((m.double() * d.double()).sum()) / nuclear
    deflated, k = deflated_newton_schulz(m, three, generator=torch.Generator(device=device).manual_seed(0))
    rows[name] = {"paper5_steps5": keep(newton_schulz(m)), "jordan_steps3": keep(newton_schulz(m, three)),
                  "deflated_jordan_steps3": keep(deflated), "deflated_pairs": int(k[0])}
out = {"source": str(run), "step": saved["step"], "matrices": rows}
Path(__file__).with_name("expectation.json").write_text(json.dumps(out, indent=1) + "\n")
for kind in ("q", "k", "v", "o", "up", "down"):
    sel = [v for n, v in rows.items() if n.endswith("." + kind)]
    med = lambda key: sorted(v[key] for v in sel)[len(sel) // 2]
    print(f"{kind:5s} 5-step paper {med('paper5_steps5'):.3f} | 3-step plain {med('jordan_steps3'):.3f} | "
          f"3-step deflated {med('deflated_jordan_steps3'):.3f} | gate fired {sum(v['deflated_pairs'] > 0 for v in sel)}/{len(sel)}"
          f" | k values {sorted(v['deflated_pairs'] for v in sel)}")
