"""Regime-check run configs (2026-09-29): existing D recipes plus a model snapshot every step.

Only keep_model_every, eval_every (1M runs) and run_id change relative to the source configs,
except the new PD 1M arm, which is the batch cohort's SoapMuon+PD 1M config with method "pd".
"""
import json
from pathlib import Path

ROOT = Path("logs/tiny_spectra")
OUT = ROOT / "regime_check_20260929" / "configs"
SOURCES = {
    "muon_b262144_lr0.01_s20261001": (ROOT / "fineweb_d_ordering_20260928/configs/muon_D_lr0.01_m0.9_s20261001.json", {}),
    "pd_b262144_lr0.01_s20261001": (ROOT / "fineweb_d_ordering_20260928/configs/pd_D_lr0.01_m0.9_s20261001.json", {}),
    "muon_b1048576_lr0.01_s20261001": (ROOT / "fineweb_d_batch_20260928/configs/muon_Dbatch_b1048576_lr0.01_s20261001.json",
                                       {"eval_every": 4}),
    "pd_b1048576_lr0.02_s20261001": (ROOT / "fineweb_d_batch_20260928/configs/spd_Dbatch_b1048576_lr0.02_s20261001.json",
                                     {"eval_every": 4, "method": "pd"}),
}
for run_id, (source, changes) in SOURCES.items():
    cfg = json.loads(source.read_text())
    cfg.update(changes, keep_model_every=1, run_id=run_id)
    (OUT / f"{run_id}.json").write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")
    print(run_id, "from", source.name, {k: cfg[k] for k in ("method", "batch_tokens", "lr", "momentum", "alpha", "eval_every", "root_refresh")})
