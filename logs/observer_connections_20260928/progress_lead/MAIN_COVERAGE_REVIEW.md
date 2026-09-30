# Main-study coverage of step lead and warmup progress

2026-09-28, bounded read-only review. No model, tensor, new score, script
execution, or job. Current state/notebook, plotting/analysis source, and
saved result schemas were inspected. Only this file is written.

**The main now has a generic step-lead implementation, and already plots
all three warmup pairs' loss differences. It does not yet have a located,
retained three-pair analysis that confines both smoothing and crossing to
the common-beta, pre-cooldown interval.** Such an analysis can clarify the
“head start, not faster rate” reading, but should not be presented as the
first measurement of these trajectories or a newly identified phase clock.

Let `A = logs/muon_spectra/second_order_audit_20260926`.

## Four distinct readouts

1. **New direct step lead: `A/step_lead.py`.** This file appeared during
   this review; filesystem mtime is 2026-09-28 20:01:48 CDT. It cites the
   19:56 correction and accepts arbitrary arm A/B paths. Default anchors
   are 46,75,83 and default smoothing half-width is 3.

   `curve` reads each saved `train_nll`, orders by step, and averages the
   surrounding seven **array entries**, with shorter windows at endpoints.
   `lead` interpolates A's smoothed loss at requested t, finds the first
   B sample at or below that loss, and linearly interpolates the crossing
   between adjacent B samples. Output is

       lead_A_over_B(t) = crossing_step_B(loss_A(t)) − t.

   Positive means A reaches that training-loss level earlier. This is
   neither validation interpolation nor loss-gap divided by a local
   derivative. If no B crossing exists, it prints a lower bound based on
   last_B−t. The CLI rounds leads to .1 step and bounds to integer steps.
   No output JSON or fixed three-warmup-pair results from this script were
   located in the audit directory during this bounded read.

2. **Warmup phase plots: `A/make_atlas.py:504–546`.** `smoothed(half=3)`
   averages existing rows whose step IDs fall in t−3…t+3, then `phase_pairs`
   saves `[t, round(loss_A(t)−loss_B(t),4)]`. It includes all three completed
   184-step warmup-versus-own-.9 pairs: original SOAP-PD seed260925,
   SOAP-PD seed260926, and PD seed260925. It also includes constant-momentum,
   short-horizon, dose, and 4M comparisons. These are **vertical NLL gaps**,
   not horizontal step leads. No special filter starts at 93 or ends at166.
   `atlas_template.html:508–518` renders these differences as phase curves.

3. **Cross-method speedup: `A/speedup_by_batch.py` and `.json`.**
   `train_curve` uses an odd running-mean width near 3% of the available
   run length, edge-padded. `speedup` takes the reference loss at t and
   finds the arm's first linearly interpolated crossing s; output is t/s.
   It retains reference fractions .08 through .9. The Muon reference is
   selected from available candidates by final validation loss, not fixed
   as the paired .9 schedule. The saved JSON is from 13:15 CDT, covers
   1M/4M/16M comparisons, and contains **no warmup arms**. Its long-horizon
   group is constant-beta SOAP-PD versus Muon. This is not the provenance
   of a three-pair warmup head-start estimate.

4. **Older validation speedup: `A/make_atlas.py:254–297`.**
   `val_curve` reads sparse validation rows. `steps_to` finds the first
   bracketing segment and interpolates **log(step)** linearly in loss.
   It reports t/s for 1M PD and 4M PD/TS/SOAP-PD versus Muon. This is a
   different interpolation convention and different family; it also does
   not cover the three warmup pairs.

## What the main has already claimed

`research/adamw_spectra/MUON_CASE.md:4186–4208` records the three successful
warmup endpoints, their similarly shrinking NLL gaps, and the 19:56
corrections. Those corrections state dose leads near step80 of 0/4/6.5/5.5
steps and a warmup head start of 4–7 steps after beta matches. The new
`step_lead.py` supplies an explicit reproducible definition going forward;
the notebook entry itself does not preserve the original command, smoothing
choice, exact anchors, or numerical table behind those particular quotes.
Do not silently certify them as produced by a later-created script.

`make_atlas.py` already includes the completed replication pairs as of
19:43, but the located readable atlas build is older (16:01–16:02) and lacks
their labels. Source coverage and currently built presentation coverage
therefore differ. No atlas was rebuilt here.

## Limits relevant to the proposed narrow analysis

- In the qualified runs, beta first equals .9 at **update93**, and the
  applied LR first declines at **update167**. Saved training loss at row t
  is measured before update t, on weights after t−1. Preserve that indexing.
- A centered seven-point value at t uses rows through t+3. The reference
  crossing also has its own smoothing support. Merely choosing centers
  inside 93–166 does not keep all contributing rows or states inside the
  desired regime. The existing lead code has no boundary checks.
- Reference crossings may lie outside the queried interval or be absent.
  Preserve this censoring; do not extrapolate a crossing or convert a
  lower bound into an exact endpoint advantage.
- The first-crossing rule can react to nonmonotone sample variation.
  Neither implementation enforces monotonicity. The direct script also
  smooths by array index, so missing steps would change the effective
  support; complete rows have already been qualified for the new pairs.
- Same-step training batches are paired, but a horizontal comparison
  necessarily compares losses at different corpus offsets. New batches
  are not independent, identically distributed population-loss estimates
  by construction. Smoothing reduces visible variation without establishing
  uncertainty or eliminating ordered corpus effects.
- Stable horizontal lead would support a translated-curve description;
  changing lead describes relative progress in that window. Neither proves
  equal full optimizer state, a unique causal clock, or a constant underlying
  optimization rate. Identical current beta/LR does not erase different
  histories, radii, clipping, or adaptive statistics.

The nonduplicate task is therefore a **bounded, explicit common-regime
horizontal-offset description of the three already qualified pairs**, with
all crossing/support failures retained. Replotting smoothed NLL gaps alone
would duplicate main coverage. Any new interval or smoothing sensitivity
must be labeled retrospective; the original endpoint success criteria remain
unchanged.
