# Negative early interaction does not imply helpful finite nonlinearity

2026-09-28. The fixed four-state probe is complete. Its predeclared helpful
mixed-response premise fails and the branch closes without a subgroup
factorial, more inputs, more states or fitted rescaling. All positive and
negative results are retained. Independent review:
[RESULTS_REVIEW.md](../early_coupling_peer/RESULTS_REVIEW.md).

## Question and exact decomposition

The preceding dose probe found negative body–auxiliary loss interaction
for early half-power PD, with a negative pairing of the mixed logits with
the base loss residual. That pairing, J, is a linear score at the base.
This experiment asks whether adding the mixed response after the additive
prediction changes actually helps finite cross-entropy.

For logits at the base, body-only, auxiliary-only and full-next corners,
define zadd = zB + zA − z0. Exactly:

    I_total   = CE(zF) − CE(zB) − CE(zA) + CE(z0)
    I_overlap = CE(zadd) − CE(zB) − CE(zA) + CE(z0)
    I_mixed   = CE(zF) − CE(zadd)
    I_total   = I_overlap + I_mixed
    I_mixed   = J + KL(p0 || pF) − KL(p0 || padd).

The overlap term is label independent. The mixed term is the finite loss
effect along this specified logit path. The synthetic additive logits need
not correspond to a realizable parameter update. Negative means helpful
in the defined interaction, not a faster training method.

The four own states are quarter- and half-power PD at beta .9, steps 9→10
and 46→47. Eight original inputs reproduce the preceding probe, and two
fixed banks of four fresh sequences each assess the prospective predictions.
All states use the same inputs. These are not independent training seeds.

## Fresh-input result

All entries are mean NLL per token over eight fresh sequences.

| State | Total interaction | Additive overlap | Finite mixed loss | Base-linear J | Mixed KL correction |
|---|---:|---:|---:|---:|---:|
| Quarter, step 9 | +.091008 | +.058449 | +.032559 | +.007524 | +.025035 |
| Half, step 9 | −.047696 | −.053006 | +.005310 | −.038475 | +.043785 |
| Quarter, step 46 | +.017875 | +.016257 | +.001617 | +.002507 | −.000890 |
| Half, step 46 | +.026797 | +.024265 | +.002532 | +.008208 | −.005677 |

The early half-power negative total interaction reproduces in both banks.
Its mean benefit comes from complementary additive prediction changes.
The mixed response is slightly adverse on average: +.018310 in one bank
and −.007689 in the other, pooled +.005310.

This conclusion is not solely a fresh-input fluctuation. On the original
eight inputs, early half-power J is −.032749 and negative on every sequence,
but I_mixed is +.014743 and positive on every sequence. Its +.047492 KL
correction reverses the tempting finite-benefit interpretation of J.
J remains an exact descriptor of label-relevant output nonadditivity.

## Prospective decisions

| Prediction | Observation | Verdict |
|---|---|---|
| P1: early half-power mixed term negative in both fresh banks, pooled ≤ −.01 | Bank means +.018310 / −.007689; pooled +.005310 | Fails |
| P2: half-minus-quarter early mixed term negative in both banks, pooled ≤ −.01 | −.025490 / −.029008; pooled −.027249, all eight paired signs negative | Passes as relative improvement |
| P3: half-power mixed term less favorable at step 46 in both banks | Late-minus-early −.015690 / +.010133 | Fails |

P2 says half power has a smaller adverse mean mixed contribution than
quarter power. It does not establish absolute helpfulness or rescue P1.
The early dose contrast in total interaction is −.111455 additive plus
−.027249 mixed. These are absolute contrasts, not fractions of a training
gain. No causal mediation or universal phase clock follows.

## Qualification and execution

All 256 score forwards finished in 207.47 seconds, within the 600-second
cap and below the 315.91-second initial forecast. Two CPU threads; no CUDA,
gradients, optimizer or training. Session 97830 ended with exit code 0.
The reported PID 3 belongs to the sandbox namespace, not the host namespace.

Every original per-token corner NLL and forward KL reproduces exactly.
Maximum per-token split, mixed-identity and J-identity errors are
4.89e−15, 5.14e−15 and 1.89e−15, below the fixed 1e−10 gate. Frozen helper
hashes and checkpoint tensor identities are preserved. Strict body/auxiliary
masks partition all 84 parameter tensors into 48 and 36 tensors.

The implementation and result reviews found no indexing, label,
normalization or precision explanation for the failed prediction.
See [protocol](PROTOCOL.md), `run1/result.json`, `run1/analysis.json`,
`run1/scalars.npz`, `run1/summary.csv` and
[finite split figure](run1/finite_split.png).

## Connection and stopping decision

The earlier late-Muon analysis found mostly positive additive overlap.
Early half-power PD now supplies the complementary example: additive
prediction interaction can be negative and useful. Its sign is state and
phase dependent. Neither sign alone identifies a disposable parameter
group or warrants freezing/staging it.

The apparent helpful nonlinear channel does not survive its finite-loss
test. Close that premise at the declared scope. The main positive
whitening-by-momentum training interaction and longer-horizon warmup result
are unaffected; their mediator remains open. No new optimizer or
architecture improvement is claimed.
