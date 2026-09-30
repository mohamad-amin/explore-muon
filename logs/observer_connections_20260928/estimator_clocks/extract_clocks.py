"""CPU-only, standard-library audit of frozen estimator clocks; no model execution.

Run from anywhere: python /.../estimator_clocks/extract_clocks.py
Writes only beside this script. Source digests must match archived scientific metadata.
ESS values describe the weights under independent stationary observations, not measured
independent examples. All ages are ages of the estimator's observations, not matrix roots.
"""
import ast
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUNS = [
    ("1M", "PD", "soaudit_traj_20260926/PD_a0.25_lr0.01_s260925_ada"),
    ("1M", "SPD", "soaudit_traj_20260926/SPD_a0.25_lr0.01_s260925_ada"),
    ("1M", "TS", "soaudit_twosided_20260927/TS_a0.25_b0.25gn_lr0.01_s260925_a4000"),
    ("4M", "PD", "soaudit_mom4m4_20260927/PD_a0.5_b4M_lr0.02_mom0.9_s260925_ada"),
    ("4M", "SPD", "soaudit_strength4m_20260927/SPD_a0.5_b4M_lr0.02_mom0.9_s260925_l40s"),
    ("4M", "TS", "soaudit_strength4m_20260927/TS_a0.5_b0.5gn_b4M_lr0.02_mom0.9_s260925_ada"),
    ("16M", "PD", "soaudit_batch16m_20260927/PD_a0.5_b16M_lr0.028_mom0.9_s260925_l40s"),
    ("16M", "SPD", "soaudit_batch16m_20260927/SPD_a0.5_b16M_lr0.028_mom0.9_s260925_ada"),
    ("16M", "TS", "soaudit_batch16m_20260927/TS_a0.5_b0.5gn_b16M_lr0.028_mom0.9_s260925_l40s"),
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def span(source, name, cls=None):
    tree = ast.parse(source.read_text())
    scope = tree.body if cls is None else next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls).body
    node = next(n for n in scope if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)
    return {"path": str(source.relative_to(ROOT)), "start": node.lineno, "end": node.end_lineno}


def init_defaults(source):
    cls = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "StatLinear")
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    return dict(zip([a.arg for a in node.args.args][-len(node.args.defaults):], [ast.literal_eval(n) for n in node.args.defaults]))


def ema_summary(decay):
    return {"decay": decay, "mean_age_updates": decay / (1 - decay),
            "half_life_updates": math.log(0.5) / math.log(decay),
            "iid_weight_ess_updates": (1 + decay) / (1 - decay)}


def output_startup(step, refresh, beta, sequences):
    update_steps = list(range(1, step + 1, refresh))
    n = len(update_steps)
    weights = [beta ** (n - 1)] + [(1 - beta) * beta ** (n - i - 1) for i in range(1, n)]
    assert abs(sum(weights) - 1) < 1e-12
    ess = 1 / sum(w * w for w in weights)
    return {"step": step, "refresh_count": n, "first_sample_weight": weights[0],
            "mean_age_steps": sum(w * (step - at) for w, at in zip(weights, update_steps)),
            "iid_weight_ess_refreshes": ess, "iid_weight_ess_sequences": ess * sequences}


records = []
for batch_label, algorithm, run in RUNS:
    run_path = ROOT / "logs/muon_spectra" / run
    metadata_path = run_path / "scientific/metadata.json"
    metadata = json.loads(metadata_path.read_text())
    cfg = metadata["config"]
    frozen = run_path.parent / "frozen/adamw_spectra"
    sources = {}
    for name in ("model.py", "muon.py", "distributed.py"):
        actual = digest(frozen / name)
        expected = metadata["source_sha256"][name]
        assert actual == expected, (run, name, actual, expected)
        sources[name] = {"path": str((frozen / name).relative_to(ROOT)), "sha256": actual}
    defaults = init_defaults(frozen / "model.py")
    batch, world = cfg["batch_tokens"], metadata["world_size"]
    seq, micro = cfg["model"]["seq_len"], cfg["microbatch_sequences"]
    assert batch % (world * seq * micro) == 0
    rounds = batch // (world * seq * micro)
    cov_decay, stride = defaults["cov_decay"], defaults["cov_stride"]
    q = cov_decay ** rounds
    rows = micro * len(range(1, seq, stride))
    cov = ema_summary(cov_decay)
    cov.update({"clock": "one local training microbatch forward", "rows_per_forward": rows,
                "rank_reduction": "none; owner rank's local statistic", "forward_updates_per_full_step": rounds,
                "effective_step_decay": q, "mean_age_model_states_at_refresh": q / (1 - q),
                "root_refresh_steps": cfg["data_norm_refresh"],
                "mean_cache_age_steps": (cfg["data_norm_refresh"] - 1) / 2,
                "iid_weight_ess_token_rows": rows * cov["iid_weight_ess_updates"],
                "global_token_age_microbatch_clock": cov["mean_age_updates"] * world * micro * seq,
                "root_refresh_global_tokens": cfg["data_norm_refresh"] * batch})
    cov["mean_age_model_states_with_cache"] = cov["mean_age_model_states_at_refresh"] + cov["mean_cache_age_steps"]
    cov["mean_age_global_tokens_with_cache"] = cov["mean_age_model_states_with_cache"] * batch
    momentum = ema_summary(cfg["muon_momentum"])
    momentum.update({"clock": "one optimizer step", "mean_age_global_tokens": momentum["mean_age_updates"] * batch})
    soap = None
    if cfg["soap_precondition"]:
        soap = ema_summary(cfg["soap_beta2"])
        soap.update({"gram_clock": "one globally reduced and clipped gradient per step",
                     "denominator_clock": "one projected momentum/gradient per step, after first ready step",
                     "tracked": cfg["soap_second_moment"], "basis": cfg["soap_basis"],
                     "basis_refresh": "eigh after first step; sorted one QR iteration after each later step",
                     "gram_age_when_used_steps": soap["mean_age_updates"] + 1,
                     "denominator_age_steps": soap["mean_age_updates"],
                     "mean_age_global_tokens": soap["mean_age_updates"] * batch})
    output = None
    if cfg.get("data_norm_out_beta", 0) > 0:
        beta, refresh, count = cfg["data_norm_out_ema"], cfg["data_norm_out_refresh"], cfg["data_norm_out_sequences"]
        output = ema_summary(beta)
        output.update({"clock": "one extra owner-local statistics pass each refresh",
                       "source": cfg["data_norm_out_source"], "rank_reduction": "none",
                       "sequences_per_refresh_per_factor": count, "token_rows_per_refresh_per_factor": count * seq,
                       "refresh_steps": refresh, "refresh_global_tokens": refresh * batch,
                       "mean_age_steps_at_refresh": refresh * output["mean_age_updates"],
                       "mean_age_steps_with_cache": refresh * output["mean_age_updates"] + (refresh - 1) / 2,
                       "iid_weight_ess_sequences": count * output["iid_weight_ess_updates"],
                       "iid_weight_ess_token_rows": count * seq * output["iid_weight_ess_updates"],
                       "total_refreshes": (metadata["total_steps"] - 1) // refresh + 1,
                       "startup": [output_startup(t, refresh, beta, count) for t in sorted(set([1, max(1, round(150_000_000 / batch)), *cfg["keep_checkpoints"]]))]})
        output["mean_age_global_tokens_with_cache"] = batch * output["mean_age_steps_with_cache"]
    source_spans = {"StatLinear_forward": span(frozen / "model.py", "forward", "StatLinear"),
                    "Muon_step": span(frozen / "muon.py", "step", "MuonAdamW"),
                    "data_norm_root": span(frozen / "muon.py", "data_norm_root"),
                    "soap_precondition": span(frozen / "muon.py", "soap_precondition"),
                    "soap_update_statistics": span(frozen / "muon.py", "soap_update_statistics")}
    if output:
        source_spans.update({"output_second_moments": span(frozen / "muon.py", "output_second_moments"),
                             "update_output_statistics": span(frozen / "muon.py", "update_output_statistics", "MuonAdamW")})
    records.append({"batch_label": batch_label, "algorithm": algorithm, "run": run,
                    "metadata_path": str(metadata_path.relative_to(ROOT)), "metadata_sha256": digest(metadata_path),
                    "sources": sources, "source_spans": source_spans,
                    "total_steps": metadata["total_steps"], "batch_tokens": batch, "world_size": world,
                    "microbatch_sequences": micro, "seq_len": seq, "alpha": cfg["data_norm_alpha"],
                    "input_C": cov, "momentum": momentum, "SOAP": soap, "output_B": output})

payload = {"scope": "Frozen completed representative PD/SPD/TS cohorts; not a training experiment",
           "formula_caveats": ["Steady-state ages ignore finite-run warm start except explicit output startup rows.",
                               "Cache ages assume uniformly sampled refresh phase.",
                               "An EMA observation's mean age does not equal the nonlinear inverse root's lag.",
                               "ESS is weight ESS under iid observations; tokens and gradients are correlated.",
                               "Final incomplete batch has fewer forwards and is excluded from full-step clocks."],
           "runs": records}
(HERE / "evidence.json").write_text(json.dumps(payload, indent=2) + "\n")
lines = ["# Estimator clock table", "", "Generated by `extract_clocks.py`; source SHA256 verified against every run's scientific metadata.", "",
         "| Batch | Steps | C forwards/step | C effective step decay | C mean age at refresh, steps | C refresh, steps | C + cache mean age, steps | Momentum beta / age, steps | SOAP beta / Gram used age, steps | B refresh / mean age incl cache, steps |",
         "|---|---:|---:|---:|---:|---:|---:|---|---|---|"]
for label in ("1M", "4M", "16M"):
    rs = {r["algorithm"]: r for r in records if r["batch_label"] == label}
    r = rs["SPD"]; c, m, s, b = r["input_C"], r["momentum"], r["SOAP"], rs["TS"]["output_B"]
    lines.append(f"| {label} | {r['total_steps']} | {c['forward_updates_per_full_step']} | {c['effective_step_decay']:.6f} | {c['mean_age_model_states_at_refresh']:.3f} | {c['root_refresh_steps']} | {c['mean_age_model_states_with_cache']:.3f} | {m['decay']} / {m['mean_age_updates']:.1f} | {s['decay']} / {s['gram_age_when_used_steps']:.1f} | {b['refresh_steps']} / {b['mean_age_steps_with_cache']:.1f} |")
lines += ["", "C token-row iid weight ESS = 255,744 at each batch; B sequence iid weight ESS = 72 at each batch.",
          "C and B are owner-local. SOAP's Gram uses the globally reduced gradient. Every run has 4 ranks, 16 sequences/microbatch, sequence length 512.", "",
          "| Batch | C + cache age, M global tokens | Momentum age, M tokens | SOAP denominator age, M tokens | B + cache age, M tokens | B refresh count over run |",
          "|---|---:|---:|---:|---:|---:|"]
for label in ("1M", "4M", "16M"):
    rs = {r["algorithm"]: r for r in records if r["batch_label"] == label}
    r = rs["SPD"]; c, m, s, b = r["input_C"], r["momentum"], r["SOAP"], rs["TS"]["output_B"]
    lines.append(f"| {label} | {c['mean_age_global_tokens_with_cache']/1e6:.2f} | {m['mean_age_global_tokens']/1e6:.2f} | {s['mean_age_global_tokens']/1e6:.2f} | {b['mean_age_global_tokens_with_cache']/1e6:.2f} | {b['total_refreshes']} |")
lines += ["", "## Audited runs", ""]
lines.extend(f"- {r['batch_label']} {r['algorithm']}: `{r['run']}`" for r in records)
(HERE / "TABLE.md").write_text("\n".join(lines) + "\n")
print("Verified", len(records) * 3, "frozen source/metadata hash pairs; wrote evidence.json and TABLE.md.")
