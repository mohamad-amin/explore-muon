# Observer current map

Updated 2026-09-28 after the third pass. The broad observer goal remains
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
   usefulness was subsequently scored locally, and its amplitude interpretation
   was qualified by the exact V/O gauge (points7–9 below).
6. `body_aux/REPORT.md`: completed CPU counterfactual at Muon4M183→184. Two
   eight-sequence banks have positive same-bank and negative cross-bank
   base/body and base/full gradient cosines. Auxiliary scope is not the main
   cause of that sign. A distinct joint-step loss interaction is positive
   on all16 sequences (+.01809 mean), while aux-only helps all16. This is a
   small-sample premise check, not a training-rate or improvement claim.
7. `value_step_geometry/REPORT.md`: all224 saved V steps show much smaller
   current mean-route perturbations under PD/SOAP-PD, despite larger learned
   pre-O means. The stage500 Dmu energy difference is about10–15×; decay is
   too small to explain it. These internal energies require point9's gauge
   qualification.
8. `value_split/REPORT.md`: completed qualified CPU functional split of the
   actual8-V step at Muon/PD1M500→501. Constant and centered components have
   predictive-GN correlation~.56 in both methods. Their positive cross term
   predicts finite interaction closely. PD's conditional constant benefit
   changes sign between score banks; a repeatable extra local gain is not
   established. Do not extend the sample or launch a bias/centering arm.
9. `value_gauge/REPORT.md`: larger V-coordinate mean is not an invariant
   larger attention-branch route. After O, step500 summed constant energies
   M/PD/S/SPD are93.96/101.24/122.44/74.98. Smaller joint V/O perturbations
   survive (PD5.2× below Muon; SPD9.7× below SOAP). The exact symmetry and
   counterexample are in `value_gauge/IDENTITY.md`.
10. `coupling_archive/REPORT.md`: positive coupling is broad, repeatable and
    mostly aligned with descent in the archived48-direction Grams. A small
    selected-response Gram is not the full parameter GN spectrum; this is
    not evidence for an expendable interfering mean channel. Existing block,
    deflation and staging failures are retained as constraints.

## Execution and resource state

All launched computations are complete; no observer process remains. The
latest model measurement used CPU FP32, two threads, no optimizer and no GPU;
runtime98.80s. Terminal record: `value_split/status.json`, exec session61976
completed exit0. The earlier body/auxiliary probe took80.04s and remains
complete. Main allocations and new GPU array jobs were checked at13:53 and
were untouched. No scheduler allocations were used by this observer. The
main program is also beginning a small-GPU surrogate branch; continue using
CPU only rather than assuming sub48GB devices are uncontested.

All new files live here. The separate older `../observer_20260928/` notebook,
main code/protocol/state, historical results and queues are preserved.
Future Python calls should set PYTHONDONTWRITEBYTECODE=1 and cap numerical
threads, including model probes.

## Next scientific decision

The fixed value-route premise check is complete. Larger internal means and
generic positive interference do not support a new architecture intervention.
Do not expand this point probe just to seek a favorable sign. The strongest
remaining estimator question is whether TS's fixed small output-factor sample
budget makes its direction unstable relative to a pooled estimate. The clock
audit identified this difference but did not measure it. Design a bounded CPU
repeatability check at one retained state, fixing the input root and momentum,
and separate sampled-label randomness from sequence-sampling randomness.
Record source/means/roots and report function/direction effects rather than
only a B spectrum. Begin with a qualification/cost gate and a fresh independent
review. No such measurement has been launched yet.

The body–auxiliary loss interaction is a distinct future attribution question
(head/embeddings/norm gains), not permission to start a tuning branch.

All three visible goal turns are **progress**: new reproducible evidence,
independent reviews and completed counterfactuals changed the next decision. Completion
of the full open research objective is not established.
