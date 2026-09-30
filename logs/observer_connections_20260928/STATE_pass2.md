# Observer current map

Updated 2026-09-28 after the second pass. The broad observer goal remains
active. No robust optimization/architecture improvement or novelty is claimed.

## Completed evidence

1. `SIGNAL_NOISE_FINDINGS.md`: the clipped/extrapolated SNR statistic produces
   near100% energy above threshold under zero signal. Real independently
   fitted frame measurements place69–87% of body movement in weak estimated
   coordinates despite >99% of raw signal in the strong group.
2. `estimator_clocks/NOTE.md`: TS factor sample budget stays eight local
   sequences per refresh while SOAP uses global gradients. C's apparent
   step-clock acceleration largely reflects its fixed token clock.
3. `spectral_claims/NOTE.md`: total quadratic forms are reliable; hard Krylov
   bands have not converged in length multiplier and their slope fractions
   do not rank all optimizers.
4. `PERSISTENCE_FINDINGS.md`: input-tail groups have positive next-step
   alignment at six original Muon/PD states, but dense lag products change
   sign within a few steps. They are not a steady slow gradient.
5. `attention_geometry/NOTE.md`: learned V maps suppress the input mean more
   strongly under Muon than PD/SOAP-PD; final mean-versus-centered gains are
   .077/.446/.603. Attention preserves a constant value route, whose local
   usefulness has not yet been independently scored.
6. `body_aux/REPORT.md`: completed CPU counterfactual at Muon4M183→184. Two
   eight-sequence banks have positive same-bank and negative cross-bank
   base/body and base/full gradient cosines. Auxiliary scope is not the main
   cause of that sign. A distinct joint-step loss interaction is positive
   on all16 sequences (+.01809 mean), while aux-only helps all16. This is a
   small-sample premise check, not a training-rate or improvement claim.

## Execution and resource state

All launched computations from both passes are complete; no observer process remains. The
latest model measurement used CPU FP32, two threads, no optimizer and no GPU;
runtime80.04s. Terminal record: `body_aux/status.json`, exec session34886
completed exit0. Main allocations2567578/2618555 were confirmed live at13:34
and were untouched. No scheduler allocations were used.

All new files live here. The separate older `../observer_20260928/` notebook,
main code/protocol/state, historical results and queues are preserved.
Future Python calls should set PYTHONDONTWRITEBYTECODE=1 and cap numerical
threads, including model probes.

## Next scientific decision

The stronger aux-causes-gradient-sign hypothesis lost this check. Do not
increase its sample simply for certainty. Next test whether the actual saved
V update's constant-value component earns independent loss descent, separately
from its centered component, retaining their interaction/curvature cross term.
Start with one mid-training state and CPU qualification, after a short decision
argument and fresh independent review. The proposed decomposition is in
`attention_geometry/NOTE.md`; no new model run for it has been started.

The body–auxiliary loss interaction is a distinct future attribution question
(head/embeddings/norm gains), not permission to start a tuning branch.

Both visible goal turns are **progress**: new reproducible evidence, independent
reviews, and a completed counterfactual changed the next decision. Completion
of the full open research objective is not established.
