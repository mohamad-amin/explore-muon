# Whitening strength changes predictive amplitude and early body–auxiliary interaction

2026-09-28. Completed CPU-only four-corner probe of eight saved own states:
PDalpha{.25,.5}×momentum{.9,.8}, steps9→10and46→47 at16Mbatch. Two fixed
four-sequence banks were scored for every state.256score forwards finished
in139.32seconds. No training, gradient, optimizer or GPU call.

## The training interaction is positive evidence; its mediator is still open

The main quarter-power experiment meets its recorded prediction. All four
PDarms shareLR.028,seed260925,L40S, initialization/data and other active
settings. Re-extracting their completed records gives:

| Validation point | Beta.8−.9, alpha¼ | Beta.8−.9, alpha½ | Difference in differences |
|---|---:|---:|---:|
| Step50 | −0.081191 | −0.121657 | −0.040467 |
| Final92 | −0.086270 | −0.123602 | −0.037332 |

The interaction in fixed training windows4–23,24–43,44–63,64–83and84–92is
−.00090,−.03432,−.04391,−.03159,−.03356. Thus it develops after the earliest
window and persists. This supports the alpha×momentum interaction in this
one-seed recipe. Alpha changes more than a selected stiff projection, so the
result does not uniquely identify spatial filtering as the causal mediator.

## Equal parameter budgets do not fix prediction changes

At each state, base/body-only/auxiliary-only/full-next corners use the actual
saved writes. Body comprises48hidden matrices; auxiliaries are the other36
tensors. Primary amplitude is forwardKL(p_base||p_corner), measured exactly
from FP64log probabilities. Base distributions and trained states differ
across arms; this is not a common-state intervention.

| State | Half/quarter body KL | Half/quarter auxiliary KL | Half/quarter full KL |
|---|---:|---:|---:|
| Step9, beta.9 | 0.845 | 0.461 | 0.655 |
| Step9, beta.8 | 1.085 | 0.400 | 0.924 |
| Step46, beta.9 | 0.716 | 1.586 | 0.856 |
| Step46, beta.8 | 0.684 | 0.793 | 0.714 |

Actual body parameter-norm ratios are1.00009–1.00036; the nominal norm is
5.376 in every cell. Stronger whitening gives smaller body predictive KL
in three cells, but **larger** KL early at beta.8, in both banks. Reverse and
symmetric KL agree on that exception. A blanket body-amplitude-shrink account
therefore does not hold over the declared cells. This does not exclude an
amplitude contribution later, where body KL falls by roughly30%.

Finite-step asymmetry matters: the early beta.8full-step ratio is.924 in
forwardKL but1.144 in reverseKL and1.023 symmetrically. No favorable divergence
is substituted for the primary one. All measures/banks are retained.

Shorter momentum roughly doubles body KL at step46for both powers (2.111×
at alpha¼,2.018×at alpha½), despite essentially identical body parameter
norms. Meanwhile the auxiliary KL is nearly unchanged at alpha¼ but halves
at alpha½. Identical auxiliary optimizer settings therefore do not supply
an identical auxiliary functional step. This is a consequence of the changed
trajectory, not an invalidation of the matched training treatment.

## An additional early interaction appears

The exact loss interaction is I=L_full−L_body−L_aux+L_base. Positive means
the joint loss change is worse than the sum of isolated changes; negative
means better. It does not rank training methods or imply a group is harmful.

| Step9 state | Body-only NLL change | Auxiliary-only change | Full change | Interaction I |
|---|---:|---:|---:|---:|
| alpha¼,beta.9 | +0.04523 | −0.09184 | +0.04055 | +0.08715 |
| alpha¼,beta.8 | −0.00773 | −0.09769 | −0.04406 | +0.06136 |
| alpha½,beta.9 | −0.06839 | −0.01582 | −0.13095 | −0.04674 |
| alpha½,beta.8 | −0.01790 | −0.02447 | −0.07766 | −0.03529 |

Both quarter-power interactions are positive on all8inputs; both half-power
interactions are negative on all8. By step46, all four interactions are
positive, with means.02247,.03109,.02959,.02530 respectively. The sign is
therefore phase/state dependent, not a generic property of a parameter group.

The early sign change is already present at both momenta, before the strong
alpha×beta training interaction develops. At beta.9, half-power later loses
to quarter-power despite its favorable early joint-step interaction. These
facts specifically prevent treating the early interaction as an explanation
of the training-rate ordering.

## Exact post-hoc evidence of a mixed model-output response

The retained CE/KLscalars permit one extra identity without model calls. Let
z0,zB,zA,zF denote the four logits, p0=softmax(z0), and e_y the true-label
vector. Define

    I_KL = KL(p0||pF) − KL(p0||pB) − KL(p0||pA),
    J = I − I_KL
      = mean[(p0−e_y) · (zF−zB−zA+z0)].

This follows by expanding CE and KL; it is not an independently measured
identity gate. J vanishes for exactly additive logits and is unchanged by
arbitrary common logit shifts at each corner. It measures the mixed output
response's pairing with the base loss residual.

| Step9 state | J |
|---|---:|
| alpha¼,beta.9 | +0.01563 |
| alpha¼,beta.8 | +0.03153 |
| alpha½,beta.9 | −0.03275 |
| alpha½,beta.8 | −0.02970 |

The half-power J is negative on all8inputs at both momenta, with nearly equal
bank means. Quarter-power J is positive on7/8and8/8. At step46all means are
positive and smaller (.00049–.00692). A purely additive-logit description
cannot reproduce this label-relevant nonadditivity at the early half-power
states. Body-only geometry need not describe the early whole-model step.

**This J is not the earlier aux_partition mixed-loss contribution.** That
quantity was CE(zF)−CE(zB+zA−z0), which additionally requires the predictive
KL of the additive logits. It cannot be recovered from the present retained
scalars. No new overlap fraction, residual-Hessian magnitude or auxiliary
subgroup attribution follows. The earlier late-Muon96–98%overlap result
remains a valid state-local finding; these quantities are not interchangeable.

## Qualification, limits and decision

Strict corner tensors equal the intended stored base/next tensors. Restoring
the base reproduces logits exactly. Maximum per-token FP64/FP32CEdifference
is2.32e−6, under the fixed5e−5gate; nullKLis0and all divergences are finite
and nonnegative within tolerance. Parameter/source/config pairing passed;
newer head/prefilter/warmup options are inactive. Helper code matches the
previous qualified observer source. Checkpoint tensor and token hashes,
per-token losses/divergences, norms, full trajectories and timings are saved.
Forecast287.82seconds; actual139.32, under600. Independent design,
implementation and result reviews passed.

The positive dose prediction is retained. The uniform body-amplitude account
is too simple, and fixed auxiliary settings conceal substantial functional
changes. The newly observed early mixed-logit interaction is a legitimate
additional phase-dependent phenomenon. None identifies the training mediator,
proves the stiff-filtering account false, or supports a step-rescaling,
freezing, staging or subgroup remedy.

Close this eight-state probe without expansion. A later mechanistic comparison
would need to distinguish additive prediction overlap from the full mixed
output contribution, localize it with a stated hypothesis, and test a
separate state/phase prediction before any optimization prescription. The
post-hoc J calculation alone does not do that.

Evidence:`run1/result.json`,`analysis.json`,`mixed_identity.json`,
`scalar_tokens.npz`,`tokens.npz`,`ratios.csv`,`alpha_kl_ratios.png`/`.pdf`;
scripts`probe.py`,`analyze.py`,`mixed_identity.py`. Independent reviews:
`../dose_interaction_peer/`.
