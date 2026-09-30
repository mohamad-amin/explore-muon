# Observer current map

Updated 2026-09-28 after the auxiliary factorial and momentum-map audit.
The broad observer goal remains active. No robust optimization/architecture
improvement or novelty is claimed. The preceding map is preserved in
`STATE_pass4.md`.

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
15. `aux_partition/REPORT.md`: completed full16-corner body/head/embedding/norm
    factorial on the original16 and fresh8 inputs at Muon4M183→184. Embeddings
    explain66.39% of fresh total interaction, head33.79%, norms.465%, with
    higher-order terms−.648%. Exact finite-logit additive overlap explains96.24%
    (old98.42%). Mixed representation/readout effects are small at this state.
    All gates pass, including exact reproduction of oldFP32losses. This closes
    the special-channel-removal/transport premise; no freezing/staging/head-LR
    intervention follows. See also `aux_inventory/REPORT.md` and fresh design
    reviews in `aux_partition_peer/` and `priority_review/`.
16. `embedding_clock/REPORT.md`: arithmetic-only support/Adam-state inventory.
    Rows absent from the next4Mtrainingbatch carry3.889% of input-embedding
    displacement energy; the zero-new-gradient recurrence matches to3.30e−6
    relative error. One fresh4-sequence bank contains no such input rows yet
    retains strong positive body/embedding interaction. Their movement cannot
    cause that bank's effect in an untied lookup. No lazy/sparseAdam remedy.
17. `momentum_maps/REPORT.md`: all18 declared16Mprofiles (M/PD/SPD, beta.9/.8,
    steps9/46/83) have a stronger within-state comparison than the main table:
    PD maps of the identical lagged momentum suppress normalized top16Ritz
    energy145–5305× against polar maps, and held-out curvature per norm53–4022×.
    This supports an additional map effect beyond co-adaptation. However,
    polar itself suppresses the raw-momentum fraction122–757×. Both maps use
    M_s, while actual steps use M_(s+1); 17/18polar and16/18PDmaps are uphill,
    whereas18/18actual steps descend. Positive a²/(2q) hides that sign.
    The precise missing feedback object is Dphi(M), not phi(M)'s projection.
    An exact fixed-norm3×3counterexample and joint weight/momentum Jacobian
    pass numerical checks (finest-step and joint errors<1e−8). These are analytic qualifications,
    not network measurements. Review: `momentum_map_peer/REVIEW.md`.

## Execution and resource state

All launched computations are complete; no observer process remains. The
auxiliary factorial took293.36s (session19525, exit0), saved displacement
inventory8.38s, and embedding-support inventory.63s (session36996, exit0).
The new18-profile scalar audit and analytic qualification completed in under
a second each, without loading checkpoints or importing model code. All
numerical work used at most two CPU threads; no observer GPU or scheduler
allocation was used. Prior TS/value/body probes and all their original
failures remain preserved. Final auxiliary scalar identities were checked
locally; independent reviews covered design and implementation.

Main allocations2567578(priv-g14)/2618555(g20) were rechecked live15:45CDT.
The4Mbeta.8arms were active; the16Mmomentum-warmup queue was waiting behind
theg20arm. The small-surrogate branch is now separated under`tiny_surrogate/`;
its19TinyStoriesdevelopment runs were complete at the last read, with the
full target ordering still unqualified. Sealed confirmation panels were not
read or scored. These main/other-branch observations are context, never
authorization to launch or continue their experiments. Continue using CPU
only rather than assuming sub48GB devices are uncontested.

All new files live here. The separate older `../observer_20260928/` notebook,
main code/protocol/state, historical results and queues are preserved.
Future Python calls should set PYTHONDONTWRITEBYTECODE=1 and cap numerical
threads, including model probes.

## Next scientific decision

The auxiliary attribution selects ordinary useful prediction overlap. Close
the channel-removal/transport story at this state; do not extend its bank,
split tiny norm subgroups or turn conditional losses into an LR prescription.
The momentum audit supports a real extra covariance-map effect while making
the missing causal measurement precise. A further step must distinguish
output suppression from response to incoming perturbations, with the correct
momentum time index and actual normalization/clipping/state conventions.
No derivative/network probe has been launched or preapproved by these notes.

Before a new model experiment, determine whether existing retained tensors
can support a source-faithful common-state comparison. The scalar profile
JSONs do not retain the Ritz vectors or incoming next-step gradient required
to recover the missing feedback derivative. A fresh experiment needs a
bounded scientific decision and peer discussion, not an automatic repair of
the latest interpretation. Broader unexplored artifacts remain legitimate
alternatives; the goal is understanding and robust improvement, not collecting
ever more local audits.

Progress audit: the intervening historical orientation fulfilled the user's
survey request but produced no new observer scientific artifact, so it is
not counted as discovery progress. This continuation is **progress**: the
completed factorial is now closed with a decision, and new all18-state
evidence plus an independently reviewed analytic discriminator changes the
next scientific question. The full open objective remains incomplete.
