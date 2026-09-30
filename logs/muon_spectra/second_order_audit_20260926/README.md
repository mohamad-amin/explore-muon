# Second-order audit: planning material (2026-09-26, CDT)

Planning only; no training or GPU job was run for this folder. Protocol entry: `research/adamw_spectra/MUON_CASE.md`,
"Second-order audit: principles behind the progress and a measurement plan".

- `framework_draft_before_review.md`: the first draft. Its H1, part 3/H5 and H4 were corrected by the independent
  review; the protocol entry has the corrected version.
- `t3_principles_survey.md`: 23 Track 3 methods classified by lever (basis, magnitude, scale, averaging, coupling),
  checked against the records' code and PR texts (upstream commit bc3a0c2d).
- `nqm_alpha2.py`: the reviewer's exactly solvable noisy-quadratic model (per-direction curvature = input variance,
  noise ∝ variance^kappa, momentum, WSD cooldown, LR tuned per alpha). `nqm_check_output.txt` is the re-run for
  cooldown 0.7 and momentum 0.95 (kappa 1 and 0.5).
- `energy_share.py`: PD's output energy E||dW x||^2 relative to Muon at matched Frobenius norm, per alpha, from a saved
  Track 3 checkpoint (median 0.64 / 0.46 / 0.37 / 0.32 for alpha 1/8 / 1/4 / 3/8 / 1/2 on run e4ff2c15).
