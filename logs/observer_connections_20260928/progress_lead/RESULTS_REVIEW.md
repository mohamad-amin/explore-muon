# Independent progress-lead result review

2026-09-28. Reviewed `PROTOCOL.md`, `analyze.py`, `result.json`, `analysis.log`,
all four CSV schemas/counts, and both figures. Independently recomputed
validation inverse secants and training anchors from the original scalar
step JSONs using a separate standard-library implementation, without
importing the analyzer. No model, tensor, GPU, or training call. Only this
review is written; existing results and code remain unchanged.

**Verdict: the reduction follows the fixed readout and has no identified
scientific or arithmetic bug. The primary conclusion is appropriately
unresolved.** Warmup's endpoint benefits remain established. The available
fixed-bank anchors cannot distinguish constant, eroding, and increasing
horizontal lead in the common-beta, pre-cooldown interval. Secondary
changing-batch curves show a positive apparent lead with a net decline over
the chosen anchors, whose size depends strongly on smoothing.

## Reproduction and coverage

- All **12 primary validation leads**, plus all 24 omission-sensitivity
  leads, reproduce from raw validation rows with maximum discrepancy
  **5.69e−14 updates**.
- All **36 training anchors** reproduce, including every pair, width, and
  states 110/120/130/140; maximum discrepancy was zero in this independent
  calculation. All 477 exported smoothed means also reproduce.
- All **1,283 manifest inputs** currently match their recorded hashes.
  The analyzer's original execution took 1.77 seconds.
- CSV coverage is complete for the declared reductions: 552 raw training
  rows, 477 smoothed centers, 36 training anchors, 36 validation-level rows.
  The raw CSV preserves every training step in every pair, with selected
  scalar fields; the original complete JSONs are preserved and hashed.
- No arms, widths, fixed targets, or fixed training anchors were dropped.
  Training fixed-level results remain in `result.json`, including missing
  high-loss coverage for some longer windows.

Reviewed hashes:

- `analyze.py`: `585a3120ec3071274263049621530d1328bbf45e8b9c03f4aa1069d80960f2df`
- `PROTOCOL.md`: `2990c683ba77b30f4f92a0b2664524270e42ce286a9335077f28cb027b6eed54`
- `result.json`: `37078931064ff169bddbc5dc2e8df5f824c40f25767603afef49145ef9ff22cf`

Minor provenance note: the source-clock review itself is not among the
manifest inputs, although the protocol cites it. The frozen trainer sources
and every scalar source it qualifies are hashed. This omission does not
change any result and requires no scientific rerun.

## Primary fixed-bank result

Independent calculation used, separately for each arm,

    crossing(level) = 100 + 50*(L100−level)/(L100−L150),
    lead = crossing_constant − crossing_warmup.

All targets are bracketed by 100/150 in both arms; both endpoints lie inside
the declared 93–166 validation-state range. Ordered by decreasing loss:

| Pair | Lead at 4.40 | Lead at 4.35 | Lead at 4.30 | Lead at 4.25 |
|---|---:|---:|---:|---:|
| PD 260925 |8.3470|7.4989|6.6508|5.8027|
| SOAP-PD 260925 |8.0328|7.2693|6.5058|5.7423|
| SOAP-PD 260926 |10.3746|9.3178|8.2610|7.2042|

These numbers describe the selected linear secants. Four targets within
the same two-anchor segment do not constitute four independent measurements
of decline. The exported monotonicity-conditional lead bracket is [-50,50]
for each primary level, correctly reflecting the wide observation interval;
it is not a confidence interval or a certified first-passage bound.

Omitting 100 gives leads ranging 2.781–3.977; omitting 150 gives 7.259–12.208.
All omission cases were retained. Their intervals include beta changes or
cooldown, so they are cadence sensitivities, not valid common-regime rate
estimates or guaranteed bounds on the primary error. The omitted-anchor
inverse-time errors are large for each arm, and paired cancellation is only
partial: paired inverse errors are −.747 to −2.373 updates when omitting 100,
and +1.399 to +1.658 when omitting 150.

## Constructive nonidentification is valid

For each pair the code checks the required strict anchor ordering and keeps
all three declared witness families. Their reference knots and warmup curves
match the four recorded 100/150 validation values exactly. Independent
piecewise-linear forward/inverse checks on 101 points per construction gave
maximum identity error **1.67e−13**. The time-map derivatives are 1, .84, 1.16,
and all constructed reference states are ≤159.

Thus the same four anchors admit constant 5, eroding 10→2, and growing 1→9
lead. The figure clearly calls them mathematical completions rather than
new observations. They establish nonidentification from those anchors;
they do not claim to match the dense changing-batch training series or
demonstrate three realizable optimizer dynamics.

## Secondary training curves and boundary handling

Every smoothed window uses exactly 11/21/31 training records inside 94–166,
which measure states 93–165. Center ranges are respectively 98–160,
103–155, and 108–150 in state units. Every reference crossing bracket uses
valid centers, whose complete raw windows remain in the allowed range.
No partial final batch or cooldown row enters this reduction.

All 36 declared training-anchor crossings are unique, downward, and positive.
The fixed state-140 minus state-110 lead changes are:

| Pair | 11-row mean | 21-row mean | 31-row mean |
|---|---:|---:|---:|
| PD 260925 |−1.5126|−.6989|−.4255|
| SOAP-PD 260925 |−1.7935|−.9217|−.1643|
| SOAP-PD 260926 |−1.0304|−.6214|−.2635|

All nine **net changes** are negative. The intermediate lead curves are
not monotonically decreasing: some rise between 110 and 130 and later fall.
The small 31-row net changes do not establish equivalence to a constant
head start. Overlapping windows and shared corpus order are not independent
replicates, and changing widths averages over different amounts of time.

Across all 477 eligible centers, 413 have unique crossings and 64 are explicitly
`below_observed_range`. They remain in CSV/JSON with absent numerical leads;
the plot uses gaps rather than extrapolated values. There are no upward
segments in these nine smoothed loss-curve pairs, but the generic crossing
routine and synthetic checks retain upward, flat, and multiple crossings
rather than silently selecting a favorable one. Some fixed-level training
comparisons are `above_observed_range` for the longer windows; these are
also preserved.

All three warmup endpoint scores fall below every recorded constant-arm
validation score. The output correctly records missing observed-range
coverage instead of inventing a post-184 match. This is censoring of the
observed piecewise-linear description, not proof that the actual control
could never have crossed that level between sparse evaluations.

## Reporting decision

Report a positive retained endpoint benefit and a positive apparent
common-regime training lead. Qualify its apparent erosion by the large
smoothing/cadence dependence. Do not promote either the linear validation
secants or a near-flat 31-row curve into a uniquely identified learning-rate
or phase mechanism. Current beta/LR equality does not equalize historical
momentum, adaptive statistics, parameter geometry, or clipping.

This meets the protocol's stopping condition: close the scalar archive
route without a finer smoother, extra fitting family, or new evaluation
chosen to decide among the remaining interpretations.
