# Is SOAP's large-batch gain sign-like equalization? (second-order audit, 2026-09-28)

Decision note and predictions: `research/adamw_spectra/MUON_CASE.md`, "Decision argument: is SOAP's large-batch gain
sign-like equalization?". S∘PD α ½ with `soap_second_moment` "gradient" (projected gradient's second moment) instead of
the default "update" (projected momentum's own squares, nearly sign). 16M @0.028 on g20 (reference 4.5742, Ada); 4M @0.02
β 0.9 on priv-g14 (reference 3.8260, L40S). Frozen from the live source.

## Results (05:0x CDT)
4M: 3.8238 (update 3.8260). 16M: 4.5686 (update 4.5742). Equivalent within noise; reading in MUON_CASE.md.
