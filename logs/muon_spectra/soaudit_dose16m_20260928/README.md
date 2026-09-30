# Dose-response of the momentum gain on whitening strength at 16M (second-order audit, 2026-09-28)

Candidate mechanism (MUON_CASE, 15:43): the whitening map filters the stiff oscillation spatially, so the momentum's
temporal filter can be shorter. Muon (α 0) gains nothing from β 0.8 and PD α ½ gains −0.124. If the mechanism holds,
PD α ¼, with partial decoupling, gains an intermediate amount.

- **Arms.** PD α ¼ @0.028 at β 0.9 and β 0.8, 16M, seed 260925, L40S. Pairs of the same hardware: Muon @0.02 (4.9104 /
  4.8901) and PD α ½ @0.028 (4.6711 / 4.5475).
- **Prediction.** The β 0.8 − β 0.9 gain at α ¼ lies between 0 and −0.124, i.e. in −0.02 to −0.10.
- **Queue.** priv-g14, after the 4M S∘PD β 0.8 arm.

## Results (16:45 CDT)
PD α ¼: β 0.9 4.6557, β 0.8 4.5694 (−0.086). This lies between Muon (−0.020) and PD α ½ (−0.124): the gain is monotone in
whitening strength. Whitening strength and window interact: α ¼ is better at β 0.9, α ½ at β 0.8.
