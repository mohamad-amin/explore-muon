# Where the gains land: target-token frequency (2026-09-25)

`frequency_probe.py` → `frequency.json`. Read-only, on final checkpoints at seed 260925, A6000. Per-token validation NLL is computed on 512 sequences (262k tokens, FP32). Tokens are binned by the target's unigram frequency, estimated from 50M fresh training tokens.

Per-token Δ NLL vs tuned Muon@0.007 (mean over the bin):

| target frequency | share of tokens | Muon NLL | PD α=¼ | SOAP-Muon core | S∘PD |
|---|---|---|---|---|---|
| < 1e-6 | 0.6% | 6.48 | −0.003 | +0.008 | +0.016 |
| 1e-6 – 1e-5 | 11.4% | 6.19 | −0.033 | −0.038 | −0.056 |
| 1e-5 – 1e-4 | 24.2% | 5.04 | −0.019 | −0.032 | −0.036 |
| 1e-4 – 1e-3 | 22.6% | 3.91 | −0.011 | −0.019 | −0.018 |
| 1e-3 – 1e-2 | 18.2% | 2.54 | −0.020 | −0.019 | −0.024 |
| > 1e-2 | 23.1% | 1.53 | −0.016 | −0.011 | −0.021 |
| **all** | | 3.660 | **−0.018** | **−0.023** | **−0.028** |

**Reading.**
- SOAP's gain leans toward rarer targets: −0.032 to −0.038 in the 1e-6–1e-4 bins vs −0.011 on the most frequent. This is the pattern expected of Adam-like entrywise normalization on heavy-tailed data.
- The data norm's gain is broad. It is similar on frequent tokens (−0.016 to −0.020) and on mid-rare ones.
- The two differ in where they help, which fits their stacking.
- For the rarest bin (0.6% of tokens), all differences are within per-token noise.
