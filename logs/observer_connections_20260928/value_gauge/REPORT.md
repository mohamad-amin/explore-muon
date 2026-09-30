# Output projection weakens the amplitude story but retains the perturbation difference

2026-09-28. CPU-only retrospective at all 32 original 1M-batch states (four
optimizers, eight retained checkpoints, eight layers each). Two threads,
3.00 seconds; no model calls, forwards, GPUs, jobs, or main-file changes.

## Identification result

V-only route norms are not invariant under Wv'=S Wv, Wo'=Wo S^-1, for invertible
per-head block-diagonal S. The latter preserves attention output because the
same within-head linear transform commutes with each head's attention matrix.
Therefore a larger V-coordinate route cannot by itself identify a larger
functional residual-branch route. This calculation follows the pre-analysis
`DECISION.md` and measures

    c = Wo Wv μ,                 m = Wo μ_z,

where μ is the saved mean normalized attention input and μ_z is the measured
mean after attention/before O. Both survive the internal V/O gauge. The
selection residual is r=m−c, and ||m||²=||c||²+||r||²+2c·r; the signed cross term
is retained and the identity verified. Residual-stream global scale remains a
separate ambiguity, so absolute c energy is not itself gauge-independent under
all model symmetries or a causal preservation criterion.

The larger learned V-coordinate route does **not** carry through O with the
same ranking or size. In particular SOAP-PD's step 500 constant branch component
is smaller than Muon's despite its much larger V-coordinate mean fraction.

| Step 500, sum of layer energies or median angle | Muon | PD | SOAP-Muon | SOAP-PD |
|---|---:|---:|---:|---:|
| Sum ||c||² | 93.960 | 101.244 | 122.436 | 74.976 |
| Sum ||m||² | 90.909 | 116.174 | 148.995 | 92.650 |
| Median cosine(c,m) | .803 | .867 | .951 | .928 |
| Median ||m−c||/||m|| | .651 | .509 | .338 | .376 |
| Median ||c||/||m|| | 1.064 | .970 | .959 | .948 |

At step 500 PD/SOAP-Muon/SOAP-PD exceed Muon's c energy in only 5/6/4 of eight
layers. Thus the original universal V-coordinate learned-route ordering is
not an invariant functional ordering. These sums are descriptive across layers;
they are not a network output norm (cross-layer propagation and interactions
are absent).

The better alignment of the constant component with the actual branch mean
does survive the output projection, especially late:

| Final checkpoint 1469 | Muon | PD | SOAP-Muon | SOAP-PD |
|---|---:|---:|---:|---:|
| Sum ||c||² | 651.900 | 1195.227 | 1160.410 | 806.901 |
| Sum ||m||² | 657.878 | 1387.803 | 1351.577 | 992.556 |
| Median cosine(c,m) | .840 | .959 | .964 | .974 |
| Median ||m−c||/||m|| | .581 | .284 | .272 | .232 |
| Median ||c||/||m|| | 1.122 | .941 | .957 | .928 |

At final, PD/SOAP-Muon/SOAP-PD c energy exceeds Muon's in 8/7/6 layers, while
cosine(c,m) exceeds Muon's in 7/8/8. SOAP-PD has the highest alignment, but not
the largest constant-route energy. Neither fact establishes useful descent.
The norm ratios and cosines are invariant under scalar residual rescaling;
absolute energies are not. Full O-input covariance was not saved, so there is
**no estimate of mean-versus-centered residual output energy** here.

## Actual V/O displacement at step 500→501

At fixed base-state μ, the full constant-route displacement is exactly

    Δ(WoWv)μ = Wo Dv μ + Do Wv μ + Do Dv μ.

This includes simultaneous changes to O and V and their finite-step product.
It does not include the changing μ, attention selection, upstream activation,
or changes elsewhere in the network. The full product change survives even
independent V/O gauges at the two endpoints. Its V-only and O-only pieces
survive a common fixed V/O gauge, but not arbitrary different gauges at each
endpoint; the realized saved parameterization specifies that attribution.

| Sum of energies over eight layers | Muon | PD | SOAP-Muon | SOAP-PD |
|---|---:|---:|---:|---:|
| V-only ||Wo Dv μ||² | .036513 | .005591 | .021837 | .001889 |
| O-only ||Do Wv μ||² | .003911 | .001728 | .007080 | .001136 |
| Finite-step product ||Do Dv μ||² | .00000156 | .00000030 | .00000231 | .00000035 |
| Exact joint ||Δ(WoWv)μ||² | .052025 | .009995 | .038925 | .004023 |
| Twice V/O dot product / joint energy | .223 | .267 | .256 | .247 |
| Median ||joint||/||c|| | .03120 | .00991 | .01863 | .00720 |
| c·joint positive, layer count | 4/8 | 8/8 | 7/8 | 8/8 |

The smaller shared-route perturbation under the input-whitened methods survives
O and the joint product: PD's joint energy is 5.2× below Muon's, SOAP-PD's is 9.7×
below SOAP-Muon's. Ratios of relative perturbation norms also retain the pattern.
O changes reinforce the V change rather than cancelling it on average; the V/O
cross term contributes 22–27% of joint energy. Discarding this term would be a
material mismeasurement even though the DoDv term itself is tiny.

This is a correction to the learned-amplitude interpretation, not a reversal
of the need for functional scoring. Distinct current route fluctuations remain
an observation worth scoring. The V-only local loss and GN responses have now
been measured in `../value_split/REPORT.md`; the full joint V/O path's loss
usefulness, selection interaction and GN cross terms remain unmeasured. Prior mean-only whitening/centering failures still
rule out treating this descriptive difference as an intervention endorsement.
One seed, different recipes, and co-adapted trajectories remain the scope.

## Artifacts and integrity

`layers.csv` retains all 256 layer/state readings. `step500_layers.csv` retains
all 32 joint displacements and signed cross terms. `summary.csv`,
`step500_summary.csv` and `result.json` retain medians and sums.
`result.json` includes every accessed Wv/Wo and nextweight tensor hash, μ/μ_z
hashes, source size/mtime, step verification, analysis source hash, and runtime.
Original accessed weight hashes and file metadata were rechecked after use.
The preserved source C/means come from 2048 held-out sequences per state, with
no independent-split uncertainty estimate. The script mmap-loads checkpoints
without traversing embeddings or optimizer states.

    PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python logs/observer_connections_20260928/value_gauge/analyze.py

Execution session 19341 completed exit 0. All generated files remain in this
observer subdirectory.
