"""Staleness-oracle configs (PROTOCOL.md, 2026-09-30, revised after review): the regime check's PD 1M config
plus the oracle keys. Dropped after review (not launched): stiff-only, rest-only (unmatched), probe transport."""
import json
from pathlib import Path

ROOT = Path("logs/tiny_spectra")
OUT = ROOT / "staleness_oracle_20260930" / "configs"
base = json.loads((ROOT / "regime_check_20260929/configs/pd_b1048576_lr0.02_s20261001.json").read_text())
base.update(keep_model_every=4, eval_every=4, probe_sequences=128, oracle_micro=32, stiff_k=16, stiff_refresh=2,
            stiff_lanczos=64, diagnostic_basis_every=10)
arms = {
    "pd1m_control": dict(oracle="none"),
    "pd1m_beta08": dict(oracle="none", momentum=0.8),
    "pd1m_beta05": dict(oracle="none", momentum=0.5),
    "pd1m_beta0": dict(oracle="none", momentum=0.0),
    "pd1m_oracle_full": dict(oracle="full", oracle_K=40),
    "pd1m_oracle_rest_matched": dict(oracle="rest_matched", oracle_K=40),
}
for old in OUT.glob("pd1m_*.json"):
    if old.stem not in arms:
        old.unlink()
for name, changes in arms.items():
    cfg = dict(base, **changes, run_id=name)
    (OUT / f"{name}.json").write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")
    print(name, {k: cfg[k] for k in ("method", "batch_tokens", "lr", "momentum", "oracle")})
for name in ("none", "full", "rest_matched", "transport"):
    cfg = dict(base, oracle=name, oracle_K=3, stiff_k=2, stiff_refresh=2, stiff_lanczos=4, probe_sequences=4,
               batch_tokens=4 * 512, total_tokens=4 * 4 * 512, microbatch_sequences=2, validation_tokens=8 * 512,
               evaluation_microbatch_sequences=8, device="cpu", eval_every=2, keep_model_every=2, smoke=True,
               diagnostic_basis_every=2, run_id=f"smoke_{name}", oracle_micro=2)
    (OUT / f"smoke_{name}.json").write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")
