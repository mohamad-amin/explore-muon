# Follow-up review of the executed finite-logit design

2026-09-28. Read-only review of `../aux_partition/PROTOCOL.md` and `probe.py`
while the parent monitors its CPU execution. No model evaluation was made by
this reviewer. This supersedes the optional-GN recommendation in `REVIEW.md`;
the original review remains preserved.

The exact finite CE decomposition is a better match to this narrow question
than adding an infinitesimal GN. Keeping all sixteen endpoint corners and all
Möbius contrasts preserves attribution scope. The thirteen logit planes add
an exact distinction between the loss interaction of additive logit changes
and the loss effect of the actual nonadditive logit response. No curvature
approximation is needed for that identity.

The implementation has no identified blocking scientific or indexing issue:

- The bit-mask condition enumerates exactly the four other-auxiliary contexts
  for each of B:H, B:E and B:N, plus B:HEN at the original base.
- CE's linear target terms cancel in the additive overlap term. The recorded
  overlap plus mixed-logit loss equals the original finite CE contrast.
- The four singleton finite-logit directions use base probabilities in the
  signed covariance Gram. This is a finite-response descriptor, not a
  Jacobian-derived predictive GN. The stored predictive KL is specifically
  D_KL(p_base || p_corner), not its reverse or a symmetric divergence.
- Reusing head inputs for the two settings of this untied, bias-free, linear
  readout is valid. Direct H/full shortcut checks and exact checkpoint-weight
  restoration provide useful execution qualification.
- The special head mixed response is D_head times the body-induced change
  of the head input, conditional on the chosen E/N values. Its checked RMS
  discrepancy gives an empirical floating-point floor for the first plane.
- The original loss reproduction, factorial identities and exact logit split
  checks are separately retained. Old and new data roles remain distinct.

Interpretation should retain four qualifications. First, positive additive
overlap is not necessarily waste: two useful changes can move predictions in
compatible directions and encounter diminishing or negative marginal loss
returns when combined. Second, a mixed-logit effect is a descriptive finite
response, not evidence about optimizer-state transport. Third, raw mixed-logit
RMS includes tokenwise constant shifts, which softmax cannot observe; the CE
effect is the primary functional quantity. The p-weighted centered Gram and
KL descriptors do remove that ambiguity for their respective directions.
Fourth, the first head/body numerical qualification is not a universal
resolution guarantee for much smaller E/N mixed responses. Report tiny terms
relative to the measured numerical floor without claiming exact zero.

The run writes complete banks incrementally; a failure within a bank may
leave only earlier banks in result.json. Treat any non-complete status as
incomplete and preserve logs/status rather than interpreting a selected
prefix. The per-sequence wall-time check is not a preemptive interruption of
an in-progress sequence; the parent's external monitoring controls that
remaining execution boundary. These are operational qualifications, not
reasons to expand or rerun the measurement.

Recommendation remains to interpret only the completed, qualified fixed run.
No extra direction grid, true Hessian, sample extension, or training arm is
needed to finish this point study.
