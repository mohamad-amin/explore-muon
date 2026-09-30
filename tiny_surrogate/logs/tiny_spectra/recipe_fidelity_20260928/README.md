# Recipe-fidelity audit

Core SOAP, PD-root and NS algebra matches the original frozen implementations.
The small harness intentionally changed statistic clocks and sample budgets.
At the reference4M batch, four ranks each execute128 microforwards: input covariance
retention is0.998^128=0.7739435 per update (cross-update mean age3.42).
Candidate C pools once per update with retention0.9 (age9), while using momentum0.8
(age4), rather than reference momentum0.9 (age9).

Reference output statistics use8sequences×512tokens per owner every10updates;
C uses16×128 every5updates. SOAP beta2=.9 stays in optimizer-update units.
Reference covariance GEMMs use forward BF16 autocast; C explicitly usesFP32.
Reference exposure is20.01tokens/parameter; C is4.53. C batch/parameter is below
even reference1M, so it cannot simply be called a miniature4M regime.

These are mismatches, not evidence that any one causes failure. The next two-arm
diagnostic changes only C retention to the reference4M-derived value, retaining
all other choices and the prior controls. Sealed tests remain unused.
