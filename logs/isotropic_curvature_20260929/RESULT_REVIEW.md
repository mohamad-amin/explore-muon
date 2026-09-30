# Independent completed-results review

2026-09-29. `finish_status.json`, producer status, and all three analysis
results were complete before this review. No failure was present. I derived
the observations below from the complete numeric artifacts before reading
`REPORT.md`. No model or analysis pipeline was executed or changed.

Scope: inspected all 18 actual-direction rows, all transmission metrics and
left contrasts, raw-versus-whitened right controls, geometry radius moments,
and all six joint states. Independent NumPy reductions of the original
per-token arrays reproduced the 18 mean H/GN values and six joint curvature
ratios. All 630 aggregate curvature rows have positive mean H. This is a
results/interpretation review, not a second execution of the measurement model.

## Strongest supported findings

1. **Orientation dependence survives at every sampled state/depth.** Actual H
   divided by the average of the two norm-preserving left-control H values is
   4.163–20.141 early, 4.758–9.289 at update 500, and 4.938–8.988 at update 1300.
   This conditional difference cannot be assigned to the input kick-radius
   distribution, which is identical for each left pair.

2. **Local transmission reduces, but does not remove, the GN contrast.** In
   all 36 four-context actual/left comparisons, GN per preactivation energy
   has contrast 4.925–11.536. The MLP tangent-energy factor is 1.726–5.627;
   the remaining residual-normalized contrast is 1.072–3.692. Both separate
   banks also have transmission and remaining factors above one. These are
   repeated descriptive comparisons, not independent statistical trials.
   Early block 8 is the closest to normalization removing the contrast
   (remaining factors 1.07–1.16); late block 8 retains 3.13–3.42. A fixed
   architectural factor is therefore informative but insufficient throughout
   training. The residual normalization matches pooled energy, not each
   token's energy or its association with downstream curvature.

3. **Mean H approaching GN is substantially cancellation.** Actual H/GN is
   6.132–10.658 early and 0.857–0.996 at the later checkpoints. Yet later
   mean token abs(H−GN)/mean GN is 0.495–1.995. This is not evidence that
   individual model-curvature terms vanish or that logits become locally
   linear. H−GN is itself second order.

4. **Actual-write finite departures are small and distinct from that
   cancellation.** Four-context R(+1)/Q is 0.99063–1.00242. At 16 times the
   write, the even remainder/Q is 0.98372–1.03152, while odd/even is
   −0.14883 to 0.03773. The sizable early asymmetry does not establish a
   superquadratic radial law. Conversely, the near-quadratic actual curves
   should not be generalized to all controls: the largest raw-right even
   departure at 16 times is +18.3%.

5. **Selected cross-layer terms are positive and material.** The joint
   H cost/sum of individual H costs is 1.395–1.714 across the six states;
   every off-diagonal entry in their mean 3×3 H matrices is positive.
   Actual-step interaction is 98.1–99.8% of its H cross-term prediction.
   At ±16 times, that ratio spans 0.745–1.362. This concerns the three
   selected up matrices, not a complete optimizer update or a training-rate
   explanation.

## Exceptions and qualifications that should remain visible

- **Right-control energy changes cannot be ignored.** White/raw GN per
  preactivation energy is above one in all 18 panels (1.044–1.802). White
  total GN exceeds raw in all nine Muon panels; it is lower in eight PD
  panels, with late PD block 8 the exception (ratio 1.049). The directions
  preserve different metrics and have different natural activation radii.
  These contrasts do not rank optimizers or, alone, establish better isotropy.
- **Whitening does not establish a spherical input law.** PD's angular
  relative-variance ratio to the spherical prediction changes from
  13.3–127.0 in raw coordinates to 0.34–11.65 in whitened coordinates.
  Its angular mean approaches the prediction in every panel, but discrepancies
  remain. Large ratios for late Muon can divide by a tiny spherical variance:
  inspect absolute variances too. A nearly flat tall write can give nearly
  constant kick norms even with non-spherical inputs.
- **Actual slopes are not uniformly better than each rotation.** They are
  more negative in 35 of the 36 pooled left comparisons. At PD update 500,
  block 8, left1 has slope −0.0002051 versus actual −0.0001703. The report's
  specific early-Muon example is valid; a universal first-order claim would
  not be. Lower curvature alone remains insufficient for direction quality.
- Four contexts, two fixed signed-permutation controls, three checkpoints,
  changing direction spans, differing recipe/hardware settings, and
  early-layer sequence coupling limit population and causal conclusions.
  These results constrain direct use of a shared radial cost along retained
  trained writes; they do not falsify the paper's distributional model or
  its conditional theorems.

## Report check

After deriving the numbers above, I read the completed draft of `REPORT.md`.
Its quantitative claims and interpretation boundaries agree with the checked
artifacts. No material scientific correction is required. The report already
distinguishes H−GN from nonquadratic loss behavior, restricted from full
Hessians, architectural decomposition from causal benefit, and conditional
observations from population claims.
