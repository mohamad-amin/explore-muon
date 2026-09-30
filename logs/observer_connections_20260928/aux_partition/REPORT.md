# Body–auxiliary interaction is mostly overlap of separate prediction changes

2026-09-28. Completed CPU-only factorial at the saved Muon 4M step 183→184.
This resolves the attribution question in the observer's fourth-pass state;
it establishes no optimizer improvement or training-rate mechanism.

The earlier four-corner experiment found that auxiliary updates helped alone
but hurt conditional on the body update. The complete 16-corner experiment
attributes about two thirds of this interaction to embeddings, one third to
the head, and less than one percent to normalization gains. An exact finite
logit decomposition explains 96–98% of the total interaction by the overlap
of the separate prediction changes. A large nonlinear representation/readout
transport defect is not needed at this state.

## Design and qualification

The checkpoint pair is `soaudit_mom4m_20260927/`
`M_b4M_lr0.014_mom0.9_s260925_l40s/scientific/kept/`
`step000183.pt` and `step000184_weights.pt`, under `logs/muon_spectra`.
The actual saved displacements include parameter-write precision and decay.
All 84 parameter tensors are partitioned exhaustively into:

- B: 48 hidden linear matrices;
- H: the untied `head.weight`;
- E: token and learned position embeddings;
- N: 33 RMSNorm gain tensors, including final and Q/K normalization.

Sixteen previously scored sequences reproduce the original anomaly. Eight
fresh sequences, chosen in the protocol before scoring, check consistency
at this selected state. These are not independent trajectory replications.
Offsets and exact input/target arrays are retained in `run1/result.json` and
`run1/tokens.npz`. Every sequence receives all 16 corners and all 12 conditional
body/subgroup planes, as well as body versus all auxiliaries.

All original gates passed. Old FP32 four-corner losses reproduce exactly.
Reusing the linear head input gives exactly the same logits as direct H-only
and full-model loads. The head's bilinear mixed response has RMS identity
error 1.255e-6, or 0.0346% of its 0.003629 RMS signal. Factorial and finite-loss
split identities pass; a separate scalar implementation finds maximum split
error 2.30e-15. Forward arithmetic is FP32; CE/logsumexp contrasts are FP64,
which does not remove forward-rounding error in extremely small contrasts.

The predeclared cost gate forecast 327.24 seconds; the complete probe took
293.36 seconds using two CPU threads, no CUDA and no optimizer step. Frozen
helper hashes, probe hash and checkpoint file metadata are in `result.json`.
Independent design/implementation reviews preceded execution; final numeric
identities were checked locally, not by a separate results reviewer.

## Attribution, retaining higher-order interactions

All values below are nats per token. Pair terms are factorial contrasts at
the base values of the other groups; the higher-order terms are kept, rather
than silently assigned to one subgroup.

| Contrast | Original 16 sequences | Fresh 8 sequences |
|---|---:|---:|
| Body × head | +0.00632151 | +0.00631429 |
| Body × embeddings | +0.01180312 | +0.01240764 |
| Body × normalization | +0.00008441 | +0.00008693 |
| Sum of body-containing higher-order terms | −0.00011822 | −0.00012108 |
| Total body × all auxiliaries | +0.01809081 | +0.01868778 |

On fresh inputs the pair shares are 33.79%, 66.39% and 0.465%; higher-order
terms contribute −0.648%. Do not interpret the smallest individual higher-order
signs, some of which approach FP32 forward precision. The embedding pair
interaction and its adverse conditional effect are positive on all 24 inputs.
Head interaction is positive on all 24, but its conditional loss effect has
exceptions (10/16 old and 7/8 fresh are adverse).

| Fresh-input loss change | Alone | Conditional on body |
|---|---:|---:|
| Head | −0.00479565 | +0.00151864 |
| Embeddings | −0.00690847 | +0.00549917 |
| Normalization | −0.00023213 | −0.00014520 |
| All auxiliaries | −0.01148227 | +0.00720551 |

Body alone changes loss by −0.01974301 on fresh inputs. These are local
counterfactual effects at the full saved step, not evidence that auxiliary
learning harms training or that another group should be frozen.

## Exact finite prediction-space split

For four logits z00,z10,z01,z11, form z_add=z10+z01−z00. Then

    I = CE(z11) − CE(z10) − CE(z01) + CE(z00)
      = [CE(z_add) − CE(z10) − CE(z01) + CE(z00)]
        + [CE(z11) − CE(z_add)].

The first term is label independent for softmax CE: the linear label terms
cancel. It is the interaction of additive prediction changes. The second
scores the mixed logit response added last. This is an exact finite identity
along this specified path, not a unique causal allocation or a GN approximation.

| Total interaction split | Original 16 | Fresh 8 |
|---|---:|---:|
| Additive prediction overlap | +0.01780528 | +0.01798452 |
| Mixed-logit loss contribution | +0.00028554 | +0.00070326 |
| Overlap / total | 98.42% | 96.24% |

The singleton finite-logit changes, measured in the base predictive-Fisher
metric, have body/head correlation 0.367 and body/embedding correlation 0.667
on fresh inputs. Head and embedding metric energies are nearly equal
(0.001724 and 0.001747); the embeddings' larger alignment explains their
larger interaction. Body energy is 0.16434. Exact KL from the base prediction
is 0.084917 for body alone, 0.000863 for head, 0.000874 for embeddings and
0.102953 for the full step. Similar parameter-displacement norms therefore
do not imply similar functional movement.

A post-hoc algebraic projection illustrates why removing overlap is not an
obvious remedy. Removing the component of each singleton response parallel
to the body response retains 86.5% of head metric energy but only 30.4% of
its label-linear descent; for embeddings it retains 55.5% of energy and
7.5% of descent. This is algebra in a finite-logit span, not an implemented
parameter update or a scored nonlinear projected loss.

## Decision

The prediction of ordinary useful response overlap in the independent
priority review is supported. Close the special-channel removal/transport
branch at this state. No staged update, head-only remedy, subgroup sweep or
additional local model probe follows. Relative scale and phase may still
matter, but a single full-step counterfactual cannot select their treatment.

This connects the earlier body-direction coupling and V-route cross terms:
positive interaction can be the curvature cost of moving toward the same
prediction correction. The present result makes that explanation explicit
with finite loss, rather than inferring it from a selected-direction Gram.

Evidence: `run1/result.json`, `run1/analysis.json`, `run1/validation.json`,
`run1/attribution.png` and `.pdf`; scripts `probe.py`, `analyze.py` and
`validate_and_qualify.py`. The original protocol and reviews remain unchanged.
