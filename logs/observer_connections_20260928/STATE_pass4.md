# Observer current map

Updated 2026-09-28 after the fourth pass. The broad observer goal remains
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
7. `value_step_geometry/REPORT.md`: analysis of224 saved V steps finds much smaller
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
11. `ts_repeatability/REPORT.md`: completed fixed-state TS output-factor
    sampling test (all48 matrices). Fresh8-sequence maps have cosines
    .696–.702, same-input label repeats~.74, independent16~.822, EMA-refresh
    proxies .984–.987. This is conditional on lagged M46 and a rank0-C proxy,
    not online EMA variance. Source/numerical controls pass. Parameter-space
    sensitivity is real, but the subsequent predictive-GN cosines are~.99.
    Relative response difference remains~.155; the small fixed-norm noise
    ratio is partly a functional-amplitude effect. Close the immediate
    estimator branch; do not infer harmlessness or an optimization bottleneck.
12. `clipping_connection/COMPLETION_ADDENDUM.md`: completed main-study control
    has clip.1 Muon beta.8 worse than.9 by.00341; scalar history is shorter,
    not older. In the extended SOAP-PD run, beta.8's NLL advantage narrows
    from.09654 at50 to.00996 at184. Loss-slope flattening matters; an exploratory
    step-lead check suggests some actual lead attenuation but proves no optimum.
13. `surrogate_observation/NOTE.md`: old character-development TS/PD reversal
    is reproduced from871 saved per-window losses: fixed256 favors TS,
    remaining615 favors PD. Numerical error cannot explain it. This is
    scoring-composition sensitivity, not proof of the gradient-SNR mechanism.
    The main program has since declared all old character scores development,
    rejected that candidate's full qualification and begun separate data-only
    preparation. Do not inspect or score its sealed confirmation data.
14. `gaussian_fisher_peer/NOTE.md`: same-target Fisher half-factor propagation
    with random signs is established prior art and a possible variance-control
    candidate. No such model probe or training arm has been run; raw-factor
    variance guarantees do not establish a nonlinear optimizer improvement.

## Execution and resource state

All launched computations are complete; no observer process remains. Latest
TS factor run2 took392.30s (session77168, exit0); saved-matrix calibration
took65.70s (session86314, exit0); predictive-GN evaluation took110.13s
(session1657, exit0). Each used at most two CPU threads and no optimizer/GPU.
The original cost-gate stop and calibration operation-order failure remain
preserved, with unchanged scientific gates. Prior model probes remain complete.
Main allocations and new GPU arrays were checked at14:13 and left untouched;
completed clipping/horizon records were reverified at14:59. No scheduler
allocation was used by this observer. The
main program is also beginning a small-GPU surrogate branch; continue using
CPU only rather than assuming sub48GB devices are uncontested.

All new files live here. The separate older `../observer_20260928/` notebook,
main code/protocol/state, historical results and queues are preserved.
Future Python calls should set PYTHONDONTWRITEBYTECODE=1 and cap numerical
threads, including model probes.

## Next scientific decision

The TS point study is complete and does not justify another covariance sampler
or larger sample. Preserve its amplitude/shape qualifications and move to a
distinct observation: the positive body–auxiliary finite-loss interaction.
Which auxiliary group accounts for it: head, embeddings, or normalization
gains? Begin with saved actual displacement and configuration/scale inventory,
using the completed body_aux probe and current head/clipping records. Any
subsequent matched counterfactual needs its own decision argument and fresh
review; no attribution model run has yet been launched. Do not equate generic
positive coupling with waste or tune a head LR from one-step scores.

All four visible goal turns are **progress**: new reproducible evidence,
independent reviews and completed counterfactuals changed the next decision. Completion
of the full open research objective is not established.
