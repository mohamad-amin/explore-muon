# Frontier-norm variant, depth 8, width 512 (2026-09-25)

The user authorized this variant on 2026-09-25: no biases, RMSNorm with gains, and Q/K norms. It is a paired Muon/AdamW cohort. The decision argument, gates and predeclared AdamW grid are in [MUON_CASE.md](../../../research/adamw_spectra/MUON_CASE.md#frontier-norm-architecture-variant-2026-09-25).

## Architecture change

The architecture differs from the depth-8 cohort (`depth8_w512_20260925_r2`) only in the `ModelConfig` flags `bias=False`, `norm="rmsnorm"` and `qk_norm=True`:
- q/k/v/o/up/down have no biases;
- ln1/ln2/final use RMSNorm with a learnable gain (ε 1e-6, FP32 compute) instead of LayerNorm;
- queries and keys get per-head RMSNorm with a shared (64,) gain after projection.

Everything else in `config_muon.json` is byte-identical to the old Muon scientific config: learned positions, GELU MLP, 8×64 heads, untied head, initialization, seed 260924, data, batch, tokens, schedule and the Muon recipe (LR .01, aux .002, momentum .95, five paper NS polynomials).

Default flags reproduce the old cohort's initial weights exactly (`test_arch.py`). This variant draws different initial weights because bias-free layers consume fewer random numbers. The Muon and AdamW arms here share identical initial weights, checked by hash at AdamW qualification.

## Pipeline

`run_variant.py` runs on allocation 2567578 (priv-g14, 4× L40S), launched detached (`launch.json`). Progress is in `pipeline.json`. Both arms use the same fresh-token diagnostic instrument as `../spike_diagnostics_20260925`.
- **Muon arm** (`muon/`): tiny CUDA replica-audited run, five-update full-size qualification, 1,469-update scientific run, update and momentum spectral analysis, then the spike diagnostic on GPU 3.
- **AdamW arm** (`adamw/`): three 300-update learning-rate pilots at {6e-4, 1.2e-3, 2.4e-3} on GPUs 0–2, concurrent with the Muon diagnostic. Selection uses the original rule. Then qualification paired to the Muon initial-weight hash, a four-rank scientific run, analysis and the diagnostic.

Frozen sources are in `frozen/` with hashes in `frozen_manifest.json`. Unit tests on the frozen copy passed (`unit_tests.log`, 26 tests; the hash-regression test skips there because it needs the live log tree, and passed on the live tree).

## Results

### Muon arm (complete 2026-09-25 06:20 UTC)

One seed per architecture; not significance-tested. Full table: [comparison_muon.md](comparison_muon.md).

- **Loss.** Final validation NLL 3.71310 versus 3.72099 for the old architecture (−0.0079 nats/token).
- **Diagnostic gates passed.** Loaded-weight NLL within 9e-6; reconstruction 3.5e-3; hook-versus-gradient 3e-6.
- **The momentum spike shrinks but stays.** Mean ρ1 over steps 1300–1469:
  - V 0.83–0.87 → 0.69–0.74 (final 0.74 → 0.55);
  - up 0.50–0.72 → 0.29–0.53;
  - mid/late Q 0.53–0.63 → 0.35–0.44;
  - final O 0.66 → 0.57; final down 0.47 → 0.40;
  - mid/late K 0.13–0.18 → 0.21–0.23.
  
  Normalized medians rise by up to about 60%.
- **The spike is still the token-mean product.** Readings are 35 mean / 7 sink / 5 distributed / 1 not reproduced (old: 33 / 5 / 7 / 3). Final-block mean shares: O 0.96, down 0.88, Q 0.86, V 0.74 (first-token share 0.35), up 0.69.
- **K changes route.** QK-norm breaks the key-shift invariance, so K's mean term is no longer exactly zero (fresh mean-product energy 0.07–0.48). Where reproduced in mid/late blocks, K's spike is now first-token dominated (0.75 and 1.00).
- **Attention sinks are stronger.** Mean attention mass on the first token in blocks 2–7 rises from 0.04–0.20 to 0.13–0.32. The first token's residual norm in blocks 2–4 is 5.1–5.4× the rest (old 2.1–3.4×). V's first-token share rises from 0.04–0.28 to 0.22–0.40.
- **The bulk is as before.** Cross-fitted/in-sample medians: mode 1 1.16, modes 2–5 0.87, top decile 0.53, q0.1–0.5 0.30, q0.5–0.9 0.21, bottom 0.19.

### AdamW arm (complete 2026-09-25 07:02 UTC)

Full table: [comparison_adamw.md](comparison_adamw.md).

- **Pilots.** The predeclared rule selected LR 1.2e-3. Mean validation NLL over updates 200/250/300 was 5.477 at 6e-4, 5.282 at 1.2e-3 and 5.674 at 2.4e-3, so the optimum is bracketed. Qualification confirmed the initial weights are identical to the Muon arm's (hash `3c51d98e…`).
- **Loss.** Final validation NLL 3.86532 versus 3.87925 for the old architecture (−0.0139 nats/token).
- **Diagnostic gates passed.** NLL within 1e-6; reconstruction 3.5e-3; hook-versus-gradient 5e-6.
- **Update spectra stay flat-topped.** AdamW update ρ1 is 0.03–0.19 in both architectures.
- **First-moment spikes are mostly mean-route.** Readings are 27 mean / 5 sink / 7 distributed / 9 not reproduced (old: 22 / 3 / 17 / 6). Final V/O/up mean shares are 0.72/0.82/0.83.
- **The bulk is as before.** Cross-fitted/in-sample medians are 0.13–0.37 for the bulk bands and 0.92 for mode 1.
- **Outlier channels remain an AdamW trait.** Residual channel max/mean-abs is 27–83 in the body, versus 15–25 for the Muon variant. The first token's residual norm rises to 6.5× the rest in mid blocks.

### Cross-arm summary (one seed per cell)

| | old architecture | variant | change |
|---|---|---|---|
| Muon val NLL | 3.72099 | 3.71310 | −0.0079 |
| AdamW val NLL | 3.87925 | 3.86532 | −0.0139 |
| Muon − AdamW | −0.1583 | −0.1522 | |

The frontier-norm change lowers both optimizers' loss slightly and leaves Muon's advantage at about 0.15 nats. It shrinks but does not remove the token-mean-product spike. It strengthens attention sinks under Muon. It does not change how much real signal the bulk carries.
