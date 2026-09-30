# SOAP on the two-sided ½/½ map (S∘TS), second-order audit, 2026-09-28

Decision note and predictions: `research/adamw_spectra/MUON_CASE.md`, "Decision argument: SOAP on the two-sided ½/½
map". The live `muon.py` was patched first so that, with the output factor on, SOAP's gradient statistics are L G R
(both sides whitened, like the momentum it normalizes). No earlier configuration changes, and all 84 unit tests pass.
The frozen copy here matches that patched source.

- priv-g14 (L40S): S∘TS α ½ β_out ½ (GN labels), 4M, LR 0.02, β 0.9. Reference: S∘PD α ½ @0.02 3.8260 (L40S).
- g20 (Ada): S∘TS α ½ β_out ½, 16M, LR 0.028, β 0.9. Reference: S∘PD α ½ @0.028 4.5742 (Ada).

## Results (00:5x CDT)
4M @0.02: 3.8257 (S∘PD α ½ 3.8260). 16M @0.028: 4.6595 (S∘PD α ½ 4.5742, TS ½/½ 4.6677). Reading: MUON_CASE.md, S∘TS entry.
