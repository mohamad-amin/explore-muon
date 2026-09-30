# The large-batch speedup in the late phase: 16M-token batch for 2x the tokens (second-order audit, 2026-09-28)

Decision note and predictions: `research/adamw_spectra/MUON_CASE.md`, "Decision argument: the large-batch speedup in
the late phase".

- **Arms.** Copies of `../soaudit_batch16m_20260927` arms with total_tokens 6,159,482,880 (368 steps at 16M, the same
  step count as the 4M runs) and kept checkpoints 37, 183 and 330.
  - priv-g14 (L40S): Muon @0.02.
  - g20 (Ada): S∘PD α ½ @0.028.
- **Code.** Frozen from the live source, which includes the 2026-09-28 SOAP-statistics fix for the two-sided map. That
  fix does not affect Muon or S∘PD.

**Revision (02:4x CDT).** The first attempt, kept as `../soaudit_b16mlong_20260928_failed_tokens`, asked for 4× the
tokens (6.16e9). The qualification refused it: "Insufficient tokens; distributed execution will not recycle data".
Only 3.2e9 training tokens are available locally (32 shards). This cohort uses total_tokens 3,079,741,440 (2×, 184 steps at 16M) and
kept checkpoints 37, 92 and 165. Arm names use T2x.
