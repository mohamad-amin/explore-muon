# Whitening strength and step length at β 0.8, 16M (second-order audit, 2026-09-28)

The flattest curvature band is under-stepped (band c* 2.6–5.5 for S∘PD at 16M) while every stiffer band is at the edge.
Stronger input whitening (α ¾) shifts the step toward the flat band; the fresher momentum (β 0.8) may tolerate a longer
step. Decision argument: `research/adamw_spectra/MUON_CASE.md`, "Decision argument: stronger input whitening, or a
longer step" (12:53 CDT).

- **Arms.** 16M, seed 260925, β 0.8, Ada (g20), after the seed replicate: S∘PD α ¾ @0.028, then S∘PD α ½ @0.04.
  Reference: S∘PD α ½ @0.028 β 0.8, 4.4724 (`../soaudit_prefilter16m_20260928`).
- **13:50 CDT:** the g20 queue was stopped before either arm started. The α ¾ arm is dropped: its motivation, the flattest band's one-step under-stepping, is a Lanczos-resolution artifact per the 13:5x review. The LR 0.04 arm moved to `../soaudit_horizon16m_20260928`'s queue, after the S∘PD β 0.8 2× horizon arm.

## Result (15:31 CDT)
S∘PD α ½ β 0.8 @0.04: 4.4792, against @0.028 4.4724 (+0.007). The α ¾ arm was not run.
