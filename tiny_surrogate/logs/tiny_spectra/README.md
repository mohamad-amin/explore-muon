# Tiny surrogate study

User objective: reproduce optimizer ordering, growing geometry gain with batch, and a lower useful momentum at larger batch in a small standalone setting, using only cluster GPUs below 48 GB.

- `qualification_20260928/`: 19 CPU contracts and paired end-to-end smoke.
- `shakespeare_smoke_20260928/`: completed GPU qualification, Slurm 2626581_0, A4000 (16 GB).
- `shakespeare_screen_20260928/`: frozen 15-arm discovery screen, Slurm array 2626585; inspect live state.

Code: `research/tiny_spectra`. Decisions and predeclared criteria: tiny-surrogate entry in `research/adamw_spectra/MUON_CASE.md`. A discovery result alone does not qualify the surrogate.

Completed discovery: `shakespeare_discovery_20260928/report/README.md` combines the 15-arm screen and five outward LR checks through symlinks to original evidence. The selection-bank chain holds; full validation reverses TS and PD. This is provisional, not a qualified surrogate. `shakespeare_batchprobe_20260928` (Slurm2626637) holds the next six baseline/batch sentinels; inspect live status.

The six baseline/batch probes (2626637) are complete. The endpoint geometry gap grows with batch, but common-loss speedups are not monotone; see `shakespeare_discovery_20260928/DISCOVERY_SPEEDUP.json`. The candidate remains unqualified. Current array2626669, `shakespeare_dynamics_20260928`, is the37-arm focused batch/momentum/LR qualification plus the large-batch missing methods. Recheck its live handle before any continuation.

**Current:** the75-run character candidate is frozen unqualified (`CHARACTER_CANDIDATE_VERDICT.json`). Its28-play independent panel remains sealed. Fresh subword candidate C is GPU-qualified;15-arm development screen Slurm2626881 in `stories_c_screen_20260928` is active. It uses a1.85M model and11GB2080Ti GPUs. Tokenizer training uses only training stories; every short/tail evaluation target is covered; selection uses full development NLL; independent test stories remain sealed. See current RESEARCH_STATE and the latest MUON_CASE entries before continuation.
